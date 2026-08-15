#include "foliage/policy_engine.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace foliage {
namespace {

void append_unique(std::vector<Interlock>& destination, const Interlock interlock) {
    if (std::find(destination.begin(), destination.end(), interlock) == destination.end()) {
        destination.push_back(interlock);
    }
}

void append_unique(std::vector<Interlock>& destination,
                   const std::vector<Interlock>& interlocks) {
    for (const auto interlock : interlocks) {
        append_unique(destination, interlock);
    }
}

TimestampMs saturating_add(const TimestampMs left, const TimestampMs right) noexcept {
    constexpr auto maximum = std::numeric_limits<TimestampMs>::max();
    return right > maximum - left ? maximum : left + right;
}

bool valid_score(const double score) noexcept {
    return std::isfinite(score) && score >= 0.0 && score <= 1.0;
}

}  // namespace

PolicyEngine::PolicyEngine(PolicyConfig config, IActuator& actuator)
    : config_(std::move(config)), actuator_(actuator), next_command_id_(config_.initial_command_id) {
    if (!validate_config()) {
        throw std::invalid_argument("invalid policy configuration");
    }
}

DecisionOutput PolicyEngine::step(const PolicyInput& input) {
    DecisionOutput output{
        .at_ms = input.now_ms,
        .state_before = state_,
    };

    if (last_now_ms_.has_value() && input.now_ms < *last_now_ms_) {
        append_unique(output.interlocks, Interlock::NonMonotonicTime);
        fault(Interlock::NonMonotonicTime, TransitionReason::InvalidInput,
              "policy timestamps must be monotonic", output);
        finish_output(output);
        return output;
    }
    last_now_ms_ = input.now_ms;

    cache_inputs(input, output);
    if (perception_invalid_ || safety_invalid_) {
        const auto interlock = safety_invalid_ ? Interlock::InvalidSafetyInput
                                               : Interlock::InvalidPerception;
        fault(interlock, TransitionReason::InvalidInput,
              "rejected a malformed or future-dated input", output);
        finish_output(output);
        return output;
    }

    if (input.control == ControlRequest::EmergencyStop) {
        const auto result = execute(ActionType::EmergencyStop, input.now_ms, output);
        transition_to(PolicyState::Fault, TransitionReason::EmergencyStop, output);
        append_unique(output.interlocks, Interlock::EmergencyStopActive);
        if (!result.has_value() || !result->acknowledged()) {
            append_unique(output.interlocks, Interlock::ActuatorFault);
        }
        output.decision = DecisionCode::Faulted;
        output.detail = result.has_value() && result->acknowledged()
                            ? "operator emergency stop acknowledged"
                            : "operator emergency stop was not acknowledged";
        reset_candidates();
        finish_output(output);
        return output;
    }

    if (input.control == ControlRequest::Disarm) {
        const auto actuator_status = actuator_.status();
        if (state_ == PolicyState::Disarmed && actuator_status.connected &&
            !actuator_status.armed) {
            output.decision = DecisionCode::Disarmed;
            output.detail = "already disarmed";
            finish_output(output);
            return output;
        }
        const auto result = execute(ActionType::Disarm, input.now_ms, output);
        if (result.has_value() && result->acknowledged()) {
            transition_to(PolicyState::Disarmed, TransitionReason::OperatorDisarm, output);
            output.decision = DecisionCode::Disarmed;
            output.detail = "operator disarm acknowledged";
            reset_candidates();
            aim_acknowledged_ = false;
            burst_acknowledged_ = false;
        } else {
            fault(Interlock::ActuatorFault, TransitionReason::ActionFailure,
                  "disarm was not acknowledged", output);
        }
        finish_output(output);
        return output;
    }

    if (state_ == PolicyState::Disarmed && actuator_.status().armed) {
        fault(Interlock::ActuatorUnexpectedlyArmed, TransitionReason::SafetyFault,
              "actuator reported armed while the policy was disarmed", output);
        finish_output(output);
        return output;
    }

    if (input.control == ControlRequest::Arm && state_ == PolicyState::Fault) {
        output.decision = DecisionCode::Hold;
        output.detail = "disarm is required before re-arming after a fault";
        finish_output(output);
        return output;
    }

    if (input.control == ControlRequest::Arm && state_ == PolicyState::Disarmed) {
        const auto interlocks = hard_interlocks(input.now_ms);
        if (!interlocks.empty()) {
            append_unique(output.interlocks, interlocks);
            output.decision = DecisionCode::Hold;
            output.detail = "arm blocked by a safety interlock";
            finish_output(output);
            return output;
        }

        const auto result = execute(ActionType::Arm, input.now_ms, output);
        if (result.has_value() && result->acknowledged()) {
            transition_to(PolicyState::Monitoring, TransitionReason::ArmAcknowledged, output);
            output.decision = DecisionCode::Armed;
            output.detail = "armed and monitoring";
        } else {
            fault(Interlock::ActuatorFault, TransitionReason::ActionFailure,
                  "arm was not acknowledged", output);
        }
        finish_output(output);
        return output;
    }

    if (state_ == PolicyState::Disarmed || state_ == PolicyState::Fault) {
        output.detail = state_ == PolicyState::Disarmed ? "operator arm required"
                                                       : "fault latched; operator disarm required";
        finish_output(output);
        return output;
    }

    const auto hard = hard_interlocks(input.now_ms);
    if (!hard.empty()) {
        append_unique(output.interlocks, hard);
        fault(hard.front(), TransitionReason::SafetyFault,
              "active policy entered fault due to a hard safety interlock", output);
        finish_output(output);
        return output;
    }

    if (state_ == PolicyState::Burst) {
        if (!burst_acknowledged_) {
            fault(Interlock::ActuatorFault, TransitionReason::ActionFailure,
                  "burst completed without an acknowledgement", output);
        } else {
            transition_to(PolicyState::Cooldown, TransitionReason::BurstAcknowledged, output);
            output.decision = DecisionCode::Cooldown;
            output.detail = "burst acknowledged; cooldown active";
        }
        finish_output(output);
        return output;
    }

    if (state_ == PolicyState::Cooldown) {
        if (input.now_ms < cooldown_until_ms_) {
            append_unique(output.interlocks, Interlock::CooldownActive);
            output.decision = DecisionCode::Hold;
            output.detail = "cooldown active";
        } else {
            transition_to(PolicyState::Monitoring, TransitionReason::CooldownElapsed, output);
            output.decision = DecisionCode::Cooldown;
            output.detail = event_.active
                                ? "cooldown elapsed; continuous event remains suppressed"
                                : "cooldown elapsed; monitoring for a new event";
            reset_candidates();
        }
        finish_output(output);
        return output;
    }

    auto perception_blocks = perception_interlocks(input.now_ms);
    if (!perception_blocks.empty()) {
        append_unique(output.interlocks, perception_blocks);
        if (state_ == PolicyState::Aiming || state_ == PolicyState::Ready) {
            abort_aiming(input.now_ms, perception_blocks, output);
        } else {
            if (state_ == PolicyState::Tracking || state_ == PolicyState::Confirming) {
                transition_to(PolicyState::Monitoring, TransitionReason::InterlockHold, output);
            }
            reset_candidates();
            output.decision = DecisionCode::Hold;
            output.detail = "perception safety interlock is holding action";
        }
        finish_output(output);
        return output;
    }

    if (!incoming_frame_is_new_) {
        append_unique(output.interlocks, Interlock::AwaitingNewFrame);
        output.decision = DecisionCode::Hold;
        output.detail = "a distinct perception frame is required to advance";
        finish_output(output);
        return output;
    }

    if (!latest_perception_.has_value()) {
        fault(Interlock::MissingPerception, TransitionReason::InvalidInput,
              "perception disappeared after interlock evaluation", output);
        finish_output(output);
        return output;
    }
    const auto& perception = *latest_perception_;

    switch (state_) {
        case PolicyState::Monitoring: {
            begin_tracking(perception);
            transition_to(PolicyState::Tracking, TransitionReason::CatEnteredProtectedZone, output);
            output.decision = DecisionCode::TrackingStarted;
            output.detail = "tracking a single cat in the protected zone";
            break;
        }

        case PolicyState::Tracking: {
            if (same_tracking_candidate(perception)) {
                advance_tracking(perception);
            } else {
                begin_tracking(perception);
                output.decision = DecisionCode::TrackingStarted;
                output.detail = "track changed; tracking persistence restarted";
                break;
            }

            std::vector<Interlock> evidence_blocks{};
            const bool harmful = harmful_evidence(perception, evidence_blocks);
            if (tracking_is_persistent() && harmful) {
                begin_confirmation(perception);
                transition_to(PolicyState::Confirming, TransitionReason::TrackingPersistent, output);
                output.decision = DecisionCode::ConfirmationStarted;
                output.detail = "stable track has harmful behavior evidence";
            } else if (!harmful) {
                append_unique(output.interlocks, evidence_blocks);
                output.decision = DecisionCode::Hold;
                output.detail = "tracking; harmful behavior evidence is not sufficient";
            } else {
                output.detail = "tracking persistence not yet satisfied";
            }
            break;
        }

        case PolicyState::Confirming: {
            std::vector<Interlock> evidence_blocks{};
            const bool harmful = harmful_evidence(perception, evidence_blocks);
            if (!same_confirmation_candidate(perception) || !harmful) {
                append_unique(output.interlocks, evidence_blocks);
                begin_tracking(perception);
                has_confirmation_candidate_ = false;
                transition_to(PolicyState::Tracking, TransitionReason::InterlockHold, output);
                output.decision = DecisionCode::Hold;
                output.detail = "confirmation evidence broke; returned to tracking";
                break;
            }

            advance_confirmation(perception);
            if (!confirmation_is_persistent()) {
                output.detail = "harmful behavior confirmation is accumulating";
                break;
            }
            if (!event_.active || event_.burst_attempted) {
                append_unique(output.interlocks, Interlock::EventAlreadyActed);
                begin_tracking(perception);
                has_confirmation_candidate_ = false;
                transition_to(PolicyState::Tracking, TransitionReason::InterlockHold, output);
                output.decision = DecisionCode::Hold;
                output.detail = "continuous event has already received its single burst attempt";
                break;
            }

            transition_to(PolicyState::Aiming, TransitionReason::HarmfulBehaviorPersistent, output);
            const auto result = execute(ActionType::GotoPreset, input.now_ms, output,
                                        perception.safe_aim_preset.value_or(std::string{}));
            if (result.has_value() && result->acknowledged()) {
                aim_acknowledged_ = true;
                aim_acknowledged_at_ms_ = input.now_ms;
                output.decision = DecisionCode::AimRequested;
                output.detail = "safe aim preset acknowledged";
            } else {
                fault(Interlock::ActuatorFault, TransitionReason::ActionFailure,
                      "aim command was not acknowledged", output);
            }
            break;
        }

        case PolicyState::Aiming: {
            std::vector<Interlock> evidence_blocks{};
            const bool harmful = harmful_evidence(perception, evidence_blocks);
            const bool same_track = has_confirmation_candidate_ &&
                                    perception.primary_track_id.has_value() &&
                                    *perception.primary_track_id == confirmation_candidate_.track_id;
            if (!aim_acknowledged_ || !same_track || !harmful || !event_.active ||
                event_.burst_attempted) {
                append_unique(output.interlocks, evidence_blocks);
                if (event_.burst_attempted) {
                    append_unique(output.interlocks, Interlock::EventAlreadyActed);
                }
                abort_aiming(input.now_ms, output.interlocks, output);
                break;
            }
            if (input.now_ms < saturating_add(aim_acknowledged_at_ms_, config_.aim_settle_ms)) {
                output.decision = DecisionCode::Hold;
                output.detail = "waiting for the acknowledged aim preset to settle";
                break;
            }
            ready_after_frame_id_ = perception.frame_id;
            transition_to(PolicyState::Ready, TransitionReason::AimAcknowledged, output);
            output.decision = DecisionCode::Ready;
            output.detail = "aim is settled; another fresh safe frame is required before burst";
            break;
        }

        case PolicyState::Ready: {
            std::vector<Interlock> evidence_blocks{};
            const bool harmful = harmful_evidence(perception, evidence_blocks);
            const bool same_track = has_confirmation_candidate_ &&
                                    perception.primary_track_id.has_value() &&
                                    *perception.primary_track_id == confirmation_candidate_.track_id;
            if (perception.frame_id <= ready_after_frame_id_) {
                append_unique(output.interlocks, Interlock::AwaitingNewFrame);
                output.decision = DecisionCode::Hold;
                output.detail = "ready but waiting for a post-ready frame";
                break;
            }
            if (!same_track || !harmful || !event_.active || event_.burst_attempted) {
                append_unique(output.interlocks, evidence_blocks);
                if (event_.burst_attempted) {
                    append_unique(output.interlocks, Interlock::EventAlreadyActed);
                }
                abort_aiming(input.now_ms, output.interlocks, output);
                break;
            }

            // Latch before touching the actuator. A timeout is an unknown physical outcome and
            // must never result in an automatic retry.
            event_.burst_attempted = true;
            cooldown_until_ms_ = saturating_add(input.now_ms, config_.cooldown_ms);
            transition_to(PolicyState::Burst, TransitionReason::BurstAttempted, output);
            const auto result = execute(ActionType::Burst, input.now_ms, output, {},
                                        config_.burst_duration_ms);
            output.decision = DecisionCode::BurstAttempted;
            if (result.has_value() && result->acknowledged()) {
                burst_acknowledged_ = true;
                output.detail = "single burst acknowledged";
            } else {
                burst_acknowledged_ = false;
                fault(Interlock::ActuatorFault, TransitionReason::ActionFailure,
                      "burst was attempted but not acknowledged; automatic retry is forbidden",
                      output);
            }
            break;
        }

        case PolicyState::Disarmed:
        case PolicyState::Burst:
        case PolicyState::Cooldown:
        case PolicyState::Fault:
            break;
    }

    finish_output(output);
    return output;
}

PolicyState PolicyEngine::state() const noexcept {
    return state_;
}

const PolicyConfig& PolicyEngine::config() const noexcept {
    return config_;
}

std::optional<EventId> PolicyEngine::current_event_id() const noexcept {
    return event_.active ? std::optional<EventId>{event_.id} : std::nullopt;
}

bool PolicyEngine::current_event_burst_attempted() const noexcept {
    return event_.active && event_.burst_attempted;
}

bool PolicyEngine::validate_config() const noexcept {
    return config_.minimum_tracking_frames > 0 && config_.minimum_confirmation_frames > 0 &&
           config_.minimum_event_clear_frames > 0 &&
           valid_score(config_.minimum_track_quality) &&
           valid_score(config_.minimum_behavior_confidence) &&
           valid_score(config_.minimum_region_evidence) && config_.burst_duration_ms > 0 &&
           config_.hard_max_burst_duration_ms > 0 &&
           config_.burst_duration_ms <= config_.hard_max_burst_duration_ms &&
           config_.initial_command_id != 0;
}

void PolicyEngine::cache_inputs(const PolicyInput& input, DecisionOutput& output) {
    incoming_frame_is_new_ = false;
    perception_invalid_ = false;
    safety_invalid_ = false;
    perception_out_of_order_ = false;

    if (input.safety.has_value()) {
        const auto& safety = *input.safety;
        const bool older_than_cached = latest_safety_.has_value() &&
                                       safety.observed_at_ms < latest_safety_->observed_at_ms;
        if (safety.observed_at_ms > input.now_ms || older_than_cached) {
            safety_invalid_ = true;
            append_unique(output.interlocks, Interlock::InvalidSafetyInput);
        } else {
            latest_safety_ = safety;
        }
    }

    if (!input.perception.has_value()) {
        return;
    }

    const auto& perception = *input.perception;
    if (perception.observed_at_ms > input.now_ms || perception.frame_id == 0 ||
        !valid_score(perception.behavior_confidence) ||
        !valid_score(perception.region_evidence) || !valid_score(perception.track_quality)) {
        perception_invalid_ = true;
        append_unique(output.interlocks, Interlock::InvalidPerception);
        return;
    }

    if (latest_perception_.has_value()) {
        if (perception.frame_id < latest_perception_->frame_id ||
            perception.observed_at_ms < latest_perception_->observed_at_ms ||
            (perception.frame_id == latest_perception_->frame_id &&
             perception.observed_at_ms != latest_perception_->observed_at_ms)) {
            perception_out_of_order_ = true;
            append_unique(output.interlocks, Interlock::OutOfOrderPerception);
            return;
        }
        if (perception.frame_id == latest_perception_->frame_id) {
            return;
        }
    }

    latest_perception_ = perception;
    incoming_frame_is_new_ = true;
    update_event_latch(perception);
}

void PolicyEngine::update_event_latch(const PerceptionInput& perception) {
    if (perception.cats_in_protected_zone > 0) {
        if (!event_.active) {
            event_ = EventLatch{
                .id = next_event_id_++,
                .active = true,
                .burst_attempted = false,
                .started_at_ms = perception.observed_at_ms,
            };
        }
        event_.clear_started_at_ms.reset();
        event_.clear_frame_count = 0;
        event_.last_clear_frame_id = 0;
        return;
    }

    if (!event_.active) {
        return;
    }
    if (!event_.clear_started_at_ms.has_value()) {
        event_.clear_started_at_ms = perception.observed_at_ms;
        event_.clear_frame_count = 1;
        event_.last_clear_frame_id = perception.frame_id;
        return;
    }
    if (perception.frame_id != event_.last_clear_frame_id) {
        ++event_.clear_frame_count;
        event_.last_clear_frame_id = perception.frame_id;
    }
    const bool persisted = perception.observed_at_ms >= *event_.clear_started_at_ms &&
                           perception.observed_at_ms - *event_.clear_started_at_ms >=
                               config_.event_clear_persistence_ms;
    if (persisted && event_.clear_frame_count >= config_.minimum_event_clear_frames) {
        event_.active = false;
        event_.burst_attempted = false;
        event_.clear_started_at_ms.reset();
        event_.clear_frame_count = 0;
    }
}

std::vector<Interlock> PolicyEngine::hard_interlocks(const TimestampMs now_ms) const {
    std::vector<Interlock> result{};
    if (!latest_safety_.has_value()) {
        result.push_back(Interlock::MissingSafetyInput);
    } else {
        const auto& safety = *latest_safety_;
        if (now_ms - safety.observed_at_ms > config_.safety_stale_after_ms) {
            result.push_back(Interlock::StaleSafetyInput);
        }
        if (!safety.hardware_ready) {
            result.push_back(Interlock::HardwareNotReady);
        }
        if (!safety.watchdog_healthy) {
            result.push_back(Interlock::WatchdogUnhealthy);
        }
        if (!safety.calibration_valid) {
            result.push_back(Interlock::CalibrationInvalid);
        }
        if (safety.emergency_stop) {
            result.push_back(Interlock::EmergencyStopActive);
        }
        if (safety.actuator_fault) {
            result.push_back(Interlock::ActuatorFault);
        }
    }

    const auto status = actuator_.status();
    if (!status.connected) {
        append_unique(result, Interlock::ActuatorDisconnected);
    }
    if (!status.ready) {
        append_unique(result, Interlock::HardwareNotReady);
    }
    if (status.emergency_stop) {
        append_unique(result, Interlock::EmergencyStopActive);
    }
    if (status.fault) {
        append_unique(result, Interlock::ActuatorFault);
    }
    if (state_ != PolicyState::Disarmed && !status.armed) {
        append_unique(result, Interlock::ActuatorNotArmed);
    }
    if (command_ids_exhausted_) {
        append_unique(result, Interlock::CommandIdExhausted);
    }
    return result;
}

std::vector<Interlock> PolicyEngine::perception_interlocks(const TimestampMs now_ms) const {
    std::vector<Interlock> result{};
    if (perception_out_of_order_) {
        result.push_back(Interlock::OutOfOrderPerception);
    }
    if (!latest_perception_.has_value()) {
        result.push_back(Interlock::MissingPerception);
        return result;
    }

    const auto& perception = *latest_perception_;
    if (now_ms - perception.observed_at_ms > config_.perception_stale_after_ms) {
        result.push_back(Interlock::StalePerception);
    }
    if (perception.person_present) {
        result.push_back(Interlock::PersonPresent);
    }
    if (perception.cats_in_protected_zone == 0) {
        result.push_back(Interlock::OutsideProtectedZone);
    } else if (perception.cats_in_protected_zone > 1) {
        result.push_back(Interlock::MultipleCats);
    }
    if (perception.cats_ambiguous) {
        result.push_back(Interlock::AmbiguousCats);
    }
    if (!perception.primary_track_id.has_value()) {
        result.push_back(Interlock::MissingTrack);
    }
    if (perception.track_quality < config_.minimum_track_quality) {
        result.push_back(Interlock::PoorTracking);
    }
    if (perception.no_fire_intersection) {
        result.push_back(Interlock::NoFireIntersection);
    }
    if (!perception.safe_aim_preset.has_value() || perception.safe_aim_preset->empty()) {
        result.push_back(Interlock::MissingAimPreset);
    }
    if (event_.active && event_.burst_attempted) {
        result.push_back(Interlock::EventAlreadyActed);
    }
    return result;
}

bool PolicyEngine::harmful_evidence(const PerceptionInput& perception,
                                    std::vector<Interlock>& interlocks) const {
    if (perception.behavior == Behavior::Unknown) {
        append_unique(interlocks, Interlock::UnknownBehavior);
    } else if (perception.behavior == Behavior::Clear) {
        append_unique(interlocks, Interlock::BehaviorNotHarmful);
    }
    if (perception.behavior_confidence < config_.minimum_behavior_confidence) {
        append_unique(interlocks, Interlock::BehaviorBelowThreshold);
    }
    if (perception.region_evidence < config_.minimum_region_evidence) {
        append_unique(interlocks, Interlock::RegionEvidenceBelowThreshold);
    }
    return interlocks.empty();
}

void PolicyEngine::reset_candidates() noexcept {
    has_tracking_candidate_ = false;
    has_confirmation_candidate_ = false;
    tracking_candidate_ = Candidate{};
    confirmation_candidate_ = Candidate{};
    aim_acknowledged_ = false;
}

void PolicyEngine::transition_to(const PolicyState next, const TransitionReason reason,
                                 DecisionOutput& output) {
    if (state_ == next) {
        return;
    }
    output.transitions.push_back(TransitionRecord{
        .from = state_,
        .to = next,
        .reason = reason,
        .at_ms = output.at_ms,
    });
    state_ = next;
}

void PolicyEngine::fault(const Interlock interlock, const TransitionReason reason,
                         std::string detail, DecisionOutput& output) {
    append_unique(output.interlocks, interlock);
    const auto status = actuator_.status();
    if (status.connected && !status.emergency_stop && !command_ids_exhausted_) {
        // ESTOP is a separate safe command, never a retry of the command whose outcome faulted.
        // Use the last accepted timestamp if this fault was caused by a clock regression.
        const auto command_time = last_now_ms_.value_or(output.at_ms);
        (void)execute(ActionType::EmergencyStop, command_time, output);
    }
    transition_to(PolicyState::Fault, reason, output);
    output.decision = DecisionCode::Faulted;
    output.detail = std::move(detail);
    reset_candidates();
}

std::optional<ActionResult> PolicyEngine::execute(const ActionType type, const TimestampMs now_ms,
                                                  DecisionOutput& output, std::string target,
                                                  const std::uint32_t duration_ms) {
    if (command_ids_exhausted_) {
        append_unique(output.interlocks, Interlock::CommandIdExhausted);
        return std::nullopt;
    }
    if (next_command_id_ == std::numeric_limits<std::uint64_t>::max() &&
        type != ActionType::Disarm && type != ActionType::EmergencyStop) {
        // Keep the last representable ID available for a fail-closed command.
        append_unique(output.interlocks, Interlock::CommandIdExhausted);
        return std::nullopt;
    }

    const auto command_id = next_command_id_;
    if (next_command_id_ == std::numeric_limits<std::uint64_t>::max()) {
        command_ids_exhausted_ = true;
    } else {
        ++next_command_id_;
    }

    ActionCommand command{
        .type = type,
        .target = std::move(target),
        .amount = 0.0F,
        .duration_ms = duration_ms,
        .command_id = command_id,
        .issued_at_ms = now_ms,
        .event_id = event_.active ? std::optional<EventId>{event_.id} : std::nullopt,
    };

    ActionResult result{};
    try {
        result = actuator_.execute(command);
    } catch (const std::exception& exception) {
        result = ActionResult{
            .command_id = command.command_id,
            .status = ActionStatus::Failed,
            .completed_at_ms = now_ms,
            .reason = std::string{"ACTUATOR_EXCEPTION: "} + exception.what(),
        };
    } catch (...) {
        result = ActionResult{
            .command_id = command.command_id,
            .status = ActionStatus::Failed,
            .completed_at_ms = now_ms,
            .reason = "ACTUATOR_EXCEPTION",
        };
    }

    if (result.command_id != command.command_id) {
        result.status = ActionStatus::Failed;
        result.reason = "RESULT_COMMAND_ID_MISMATCH";
        result.command_id = command.command_id;
    }
    if (result.completed_at_ms < command.issued_at_ms || result.completed_at_ms > now_ms) {
        result.status = ActionStatus::Failed;
        result.reason = "INVALID_RESULT_TIMESTAMP";
    }
    output.actions.push_back(ActionRecord{std::move(command), result});
    return result;
}

void PolicyEngine::abort_aiming(const TimestampMs now_ms,
                                const std::vector<Interlock>& interlocks,
                                DecisionOutput& output) {
    append_unique(output.interlocks, interlocks);
    const auto result = execute(ActionType::Hold, now_ms, output);
    if (!result.has_value() || !result->acknowledged()) {
        fault(Interlock::ActuatorFault, TransitionReason::ActionFailure,
              "hold command was not acknowledged while aborting aim", output);
        return;
    }
    reset_candidates();
    transition_to(PolicyState::Monitoring, TransitionReason::InterlockHold, output);
    output.decision = DecisionCode::Hold;
    output.detail = "aim aborted and hold acknowledged";
}

bool PolicyEngine::same_tracking_candidate(const PerceptionInput& perception) const noexcept {
    return has_tracking_candidate_ && perception.primary_track_id.has_value() &&
           *perception.primary_track_id == tracking_candidate_.track_id;
}

bool PolicyEngine::same_confirmation_candidate(const PerceptionInput& perception) const noexcept {
    return has_confirmation_candidate_ && perception.primary_track_id.has_value() &&
           *perception.primary_track_id == confirmation_candidate_.track_id &&
           perception.behavior == confirmation_candidate_.behavior;
}

void PolicyEngine::begin_tracking(const PerceptionInput& perception) {
    tracking_candidate_ = Candidate{
        .track_id = perception.primary_track_id.value_or(0),
        .behavior = Behavior::Unknown,
        .started_at_ms = perception.observed_at_ms,
        .latest_at_ms = perception.observed_at_ms,
        .frame_count = 1,
        .last_frame_id = perception.frame_id,
    };
    has_tracking_candidate_ = true;
}

void PolicyEngine::advance_tracking(const PerceptionInput& perception) {
    if (perception.frame_id != tracking_candidate_.last_frame_id) {
        ++tracking_candidate_.frame_count;
        tracking_candidate_.last_frame_id = perception.frame_id;
    }
    tracking_candidate_.latest_at_ms = perception.observed_at_ms;
}

void PolicyEngine::begin_confirmation(const PerceptionInput& perception) {
    confirmation_candidate_ = Candidate{
        .track_id = perception.primary_track_id.value_or(0),
        .behavior = perception.behavior,
        .started_at_ms = perception.observed_at_ms,
        .latest_at_ms = perception.observed_at_ms,
        .frame_count = 1,
        .last_frame_id = perception.frame_id,
    };
    has_confirmation_candidate_ = true;
}

void PolicyEngine::advance_confirmation(const PerceptionInput& perception) {
    if (perception.frame_id != confirmation_candidate_.last_frame_id) {
        ++confirmation_candidate_.frame_count;
        confirmation_candidate_.last_frame_id = perception.frame_id;
    }
    confirmation_candidate_.latest_at_ms = perception.observed_at_ms;
}

bool PolicyEngine::tracking_is_persistent() const noexcept {
    return has_tracking_candidate_ &&
           tracking_candidate_.latest_at_ms >= tracking_candidate_.started_at_ms &&
           tracking_candidate_.latest_at_ms - tracking_candidate_.started_at_ms >=
               config_.tracking_persistence_ms &&
           tracking_candidate_.frame_count >= config_.minimum_tracking_frames;
}

bool PolicyEngine::confirmation_is_persistent() const noexcept {
    return has_confirmation_candidate_ &&
           confirmation_candidate_.latest_at_ms >= confirmation_candidate_.started_at_ms &&
           confirmation_candidate_.latest_at_ms - confirmation_candidate_.started_at_ms >=
               config_.confirmation_persistence_ms &&
           confirmation_candidate_.frame_count >= config_.minimum_confirmation_frames;
}

void PolicyEngine::finish_output(DecisionOutput& output) const {
    output.state_after = state_;
    output.event_id = event_.active ? std::optional<EventId>{event_.id} : std::nullopt;
    output.event_active = event_.active;
    output.event_burst_attempted = event_.active && event_.burst_attempted;
}

}  // namespace foliage
