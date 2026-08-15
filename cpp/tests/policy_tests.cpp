#include "foliage/mock_actuator.hpp"
#include "foliage/policy_engine.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace foliage;

class TestFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

#define REQUIRE(condition)                                                                     \
    do {                                                                                       \
        if (!(condition)) {                                                                    \
            throw TestFailure(std::string{__FILE__} + ":" + std::to_string(__LINE__) +        \
                              ": REQUIRE(" #condition ") failed");                           \
        }                                                                                      \
    } while (false)

PolicyConfig fast_config() {
    return PolicyConfig{
        .tracking_persistence_ms = 20,
        .minimum_tracking_frames = 2,
        .confirmation_persistence_ms = 20,
        .minimum_confirmation_frames = 3,
        .aim_settle_ms = 10,
        .event_clear_persistence_ms = 20,
        .minimum_event_clear_frames = 3,
        .perception_stale_after_ms = 50,
        .safety_stale_after_ms = 100,
        .cooldown_ms = 100,
        .minimum_track_quality = 0.70,
        .minimum_behavior_confidence = 0.80,
        .minimum_region_evidence = 0.60,
        .burst_duration_ms = 50,
        .hard_max_burst_duration_ms = 75,
        .initial_command_id = 100,
    };
}

SafetyInput safe_at(const TimestampMs at_ms) {
    return SafetyInput{
        .observed_at_ms = at_ms,
        .hardware_ready = true,
        .watchdog_healthy = true,
        .calibration_valid = true,
        .emergency_stop = false,
        .actuator_fault = false,
    };
}

PerceptionInput harmful_frame(const FrameId frame_id, const TimestampMs at_ms,
                              const TrackId track_id = 7) {
    return PerceptionInput{
        .observed_at_ms = at_ms,
        .frame_id = frame_id,
        .cats_in_protected_zone = 1,
        .cats_ambiguous = false,
        .person_present = false,
        .primary_track_id = track_id,
        .behavior = Behavior::Eating,
        .behavior_confidence = 0.95,
        .region_evidence = 0.90,
        .track_quality = 0.95,
        .no_fire_intersection = false,
        .safe_aim_preset = "pot_1_safe",
    };
}

DecisionOutput arm(PolicyEngine& policy, const TimestampMs at_ms = 0) {
    return policy.step(PolicyInput{
        .now_ms = at_ms,
        .safety = safe_at(at_ms),
        .control = ControlRequest::Arm,
    });
}

DecisionOutput frame(PolicyEngine& policy, PerceptionInput perception) {
    const auto at_ms = perception.observed_at_ms;
    return policy.step(PolicyInput{
        .now_ms = at_ms,
        .perception = std::move(perception),
        .safety = safe_at(at_ms),
    });
}

DecisionOutput drive_to_aiming(PolicyEngine& policy, FrameId first_frame = 1,
                               TimestampMs first_time = 10) {
    auto output = frame(policy, harmful_frame(first_frame, first_time));
    REQUIRE(policy.state() == PolicyState::Tracking);
    output = frame(policy, harmful_frame(first_frame + 1, first_time + 20));
    REQUIRE(policy.state() == PolicyState::Confirming);
    output = frame(policy, harmful_frame(first_frame + 2, first_time + 30));
    REQUIRE(policy.state() == PolicyState::Confirming);
    output = frame(policy, harmful_frame(first_frame + 3, first_time + 40));
    REQUIRE(policy.state() == PolicyState::Aiming);
    return output;
}

DecisionOutput drive_to_burst(PolicyEngine& policy, FrameId first_frame = 1,
                              TimestampMs first_time = 10) {
    (void)drive_to_aiming(policy, first_frame, first_time);
    auto output = frame(policy, harmful_frame(first_frame + 4, first_time + 50));
    REQUIRE(policy.state() == PolicyState::Ready);
    output = frame(policy, harmful_frame(first_frame + 5, first_time + 60));
    return output;
}

bool has_interlock(const DecisionOutput& output, const Interlock expected) {
    return std::find(output.interlocks.begin(), output.interlocks.end(), expected) !=
           output.interlocks.end();
}

void startup_requires_explicit_arm() {
    MockActuator actuator;
    PolicyEngine policy(fast_config(), actuator);

    REQUIRE(policy.state() == PolicyState::Disarmed);
    const auto ignored = frame(policy, harmful_frame(1, 10));
    REQUIRE(ignored.state_after == PolicyState::Disarmed);
    REQUIRE(actuator.count(ActionType::Burst) == 0);

    const auto armed = arm(policy, 11);
    REQUIRE(armed.state_before == PolicyState::Disarmed);
    REQUIRE(armed.state_after == PolicyState::Monitoring);
    REQUIRE(armed.decision == DecisionCode::Armed);
    REQUIRE(armed.transitions.size() == 1);
    REQUIRE(armed.actions.size() == 1);
    REQUIRE(armed.actions.front().command.type == ActionType::Arm);
    REQUIRE(actuator.status().armed);
}

void unexpected_physical_arm_state_is_failed_closed() {
    auto status = MockActuator::healthy_status();
    status.armed = true;
    MockActuator actuator(status);
    PolicyEngine policy(fast_config(), actuator);

    const auto output = policy.step(PolicyInput{
        .now_ms = 0,
        .safety = safe_at(0),
    });
    REQUIRE(output.state_after == PolicyState::Fault);
    REQUIRE(has_interlock(output, Interlock::ActuatorUnexpectedlyArmed));
    REQUIRE(actuator.count(ActionType::EmergencyStop) == 1);
    REQUIRE(!actuator.status().armed);
}

void deterministic_happy_path_is_auditable() {
    MockActuator actuator;
    PolicyEngine policy(fast_config(), actuator);
    REQUIRE(arm(policy).state_after == PolicyState::Monitoring);

    const auto aiming = drive_to_aiming(policy);
    REQUIRE(aiming.decision == DecisionCode::AimRequested);
    REQUIRE(aiming.transitions.front().from == PolicyState::Confirming);
    REQUIRE(aiming.transitions.front().to == PolicyState::Aiming);
    REQUIRE(aiming.actions.size() == 1);
    REQUIRE(aiming.actions.front().command.type == ActionType::GotoPreset);
    REQUIRE(aiming.actions.front().command.target == "pot_1_safe");

    const auto ready = frame(policy, harmful_frame(5, 60));
    REQUIRE(ready.state_after == PolicyState::Ready);
    REQUIRE(ready.decision == DecisionCode::Ready);

    const auto burst = frame(policy, harmful_frame(6, 70));
    REQUIRE(burst.state_after == PolicyState::Burst);
    REQUIRE(burst.decision == DecisionCode::BurstAttempted);
    REQUIRE(burst.actions.size() == 1);
    REQUIRE(burst.actions.front().command.type == ActionType::Burst);
    REQUIRE(burst.actions.front().command.duration_ms == 50);
    REQUIRE(burst.actions.front().command.event_id == burst.event_id);
    REQUIRE(burst.event_burst_attempted);

    const auto cooldown = policy.step(PolicyInput{
        .now_ms = 71,
        .safety = safe_at(71),
    });
    REQUIRE(cooldown.state_after == PolicyState::Cooldown);
    REQUIRE(cooldown.decision == DecisionCode::Cooldown);

    REQUIRE(actuator.unique_commands().size() == 3);
    REQUIRE(actuator.unique_commands()[0].command_id == 100);
    REQUIRE(actuator.unique_commands()[1].command_id == 101);
    REQUIRE(actuator.unique_commands()[2].command_id == 102);
    REQUIRE(to_string(PolicyState::Burst) == "BURST");
    REQUIRE(to_string(Interlock::PersonPresent) == "PERSON_PRESENT");
}

void continuous_event_gets_at_most_one_burst() {
    MockActuator actuator;
    PolicyEngine policy(fast_config(), actuator);
    (void)arm(policy);
    (void)drive_to_burst(policy);
    (void)policy.step(PolicyInput{.now_ms = 71, .safety = safe_at(71)});

    const auto elapsed = policy.step(PolicyInput{.now_ms = 170, .safety = safe_at(170)});
    REQUIRE(elapsed.state_after == PolicyState::Monitoring);
    REQUIRE(elapsed.event_burst_attempted);

    for (std::uint64_t index = 0; index < 20; ++index) {
        const auto at_ms = static_cast<TimestampMs>(171 + index * 20);
        const auto output = frame(policy, harmful_frame(7 + index, at_ms));
        REQUIRE(has_interlock(output, Interlock::EventAlreadyActed));
    }
    REQUIRE(actuator.count(ActionType::Burst) == 1);
}

void cleared_event_can_trigger_after_cooldown() {
    MockActuator actuator;
    PolicyEngine policy(fast_config(), actuator);
    (void)arm(policy);
    (void)drive_to_burst(policy);
    (void)policy.step(PolicyInput{.now_ms = 71, .safety = safe_at(71)});
    (void)policy.step(PolicyInput{.now_ms = 170, .safety = safe_at(170)});

    for (std::uint64_t index = 0; index < 3; ++index) {
        auto absent = harmful_frame(7 + index, 180 + index * 10);
        absent.cats_in_protected_zone = 0;
        absent.primary_track_id.reset();
        absent.behavior = Behavior::Clear;
        absent.behavior_confidence = 1.0;
        absent.region_evidence = 0.0;
        absent.track_quality = 0.0;
        (void)frame(policy, std::move(absent));
    }
    REQUIRE(!policy.current_event_id().has_value());

    const auto second_burst = drive_to_burst(policy, 10, 220);
    REQUIRE(second_burst.state_after == PolicyState::Burst);
    REQUIRE(actuator.count(ActionType::Burst) == 2);
}

void transient_interlock_aborts_aim_with_hold() {
    MockActuator actuator;
    PolicyEngine policy(fast_config(), actuator);
    (void)arm(policy);
    (void)drive_to_aiming(policy);

    auto person = harmful_frame(5, 60);
    person.person_present = true;
    const auto held = frame(policy, std::move(person));
    REQUIRE(held.state_after == PolicyState::Monitoring);
    REQUIRE(held.decision == DecisionCode::Hold);
    REQUIRE(has_interlock(held, Interlock::PersonPresent));
    REQUIRE(held.actions.size() == 1);
    REQUIRE(held.actions.front().command.type == ActionType::Hold);
    REQUIRE(actuator.count(ActionType::Burst) == 0);
}

void all_required_perception_interlocks_fail_closed() {
    using Mutation = std::function<void(PerceptionInput&)>;
    const std::vector<std::pair<Interlock, Mutation>> cases{
        {Interlock::PersonPresent, [](PerceptionInput& input) { input.person_present = true; }},
        {Interlock::MultipleCats,
         [](PerceptionInput& input) { input.cats_in_protected_zone = 2; }},
        {Interlock::AmbiguousCats, [](PerceptionInput& input) { input.cats_ambiguous = true; }},
        {Interlock::PoorTracking, [](PerceptionInput& input) { input.track_quality = 0.1; }},
        {Interlock::NoFireIntersection,
         [](PerceptionInput& input) { input.no_fire_intersection = true; }},
        {Interlock::MissingAimPreset,
         [](PerceptionInput& input) { input.safe_aim_preset.reset(); }},
    };

    for (const auto& [expected, mutate] : cases) {
        MockActuator actuator;
        PolicyEngine policy(fast_config(), actuator);
        (void)arm(policy);
        auto input = harmful_frame(1, 10);
        mutate(input);
        const auto output = frame(policy, std::move(input));
        REQUIRE(has_interlock(output, expected));
        REQUIRE(output.state_after == PolicyState::Monitoring);
        REQUIRE(actuator.count(ActionType::Burst) == 0);
    }
}

void stale_and_duplicate_frames_cannot_confirm() {
    MockActuator actuator;
    auto config = fast_config();
    config.perception_stale_after_ms = 25;
    PolicyEngine policy(config, actuator);
    (void)arm(policy);

    const auto first = frame(policy, harmful_frame(1, 10));
    REQUIRE(first.state_after == PolicyState::Tracking);
    for (TimestampMs at_ms : {15U, 20U, 25U}) {
        const auto duplicate = policy.step(PolicyInput{
            .now_ms = at_ms,
            .perception = harmful_frame(1, 10),
            .safety = safe_at(at_ms),
        });
        REQUIRE(has_interlock(duplicate, Interlock::AwaitingNewFrame));
        REQUIRE(policy.state() == PolicyState::Tracking);
    }

    const auto stale = policy.step(PolicyInput{
        .now_ms = 40,
        .safety = safe_at(40),
    });
    REQUIRE(has_interlock(stale, Interlock::StalePerception));
    REQUIRE(stale.state_after == PolicyState::Monitoring);
    REQUIRE(actuator.count(ActionType::GotoPreset) == 0);
    REQUIRE(actuator.count(ActionType::Burst) == 0);
}

void hardware_not_ready_latches_fault_and_estop() {
    MockActuator actuator;
    PolicyEngine policy(fast_config(), actuator);
    (void)arm(policy);

    auto unsafe = safe_at(10);
    unsafe.hardware_ready = false;
    const auto output = policy.step(PolicyInput{
        .now_ms = 10,
        .perception = harmful_frame(1, 10),
        .safety = unsafe,
    });
    REQUIRE(output.state_after == PolicyState::Fault);
    REQUIRE(has_interlock(output, Interlock::HardwareNotReady));
    REQUIRE(actuator.count(ActionType::EmergencyStop) == 1);
    REQUIRE(!actuator.status().armed);
}

void actuator_must_remain_armed_while_policy_is_active() {
    MockActuator actuator;
    PolicyEngine policy(fast_config(), actuator);
    (void)arm(policy);
    actuator.set_status(MockActuator::healthy_status());

    const auto output = frame(policy, harmful_frame(1, 10));
    REQUIRE(output.state_after == PolicyState::Fault);
    REQUIRE(has_interlock(output, Interlock::ActuatorNotArmed));
    REQUIRE(actuator.count(ActionType::EmergencyStop) == 1);
    REQUIRE(actuator.count(ActionType::Burst) == 0);
}

void repeated_arm_request_cannot_bypass_safety_checks() {
    MockActuator actuator;
    PolicyEngine policy(fast_config(), actuator);
    (void)arm(policy);

    auto unsafe = safe_at(10);
    unsafe.watchdog_healthy = false;
    const auto output = policy.step(PolicyInput{
        .now_ms = 10,
        .safety = unsafe,
        .control = ControlRequest::Arm,
    });
    REQUIRE(output.state_after == PolicyState::Fault);
    REQUIRE(has_interlock(output, Interlock::WatchdogUnhealthy));
    REQUIRE(actuator.count(ActionType::Arm) == 1);
    REQUIRE(actuator.count(ActionType::EmergencyStop) == 1);
}

void behavior_and_region_uncertainty_never_reach_aiming() {
    using Mutation = std::function<void(PerceptionInput&)>;
    const std::vector<std::pair<Interlock, Mutation>> cases{
        {Interlock::UnknownBehavior,
         [](PerceptionInput& input) { input.behavior = Behavior::Unknown; }},
        {Interlock::BehaviorNotHarmful,
         [](PerceptionInput& input) { input.behavior = Behavior::Clear; }},
        {Interlock::BehaviorBelowThreshold,
         [](PerceptionInput& input) { input.behavior_confidence = 0.2; }},
        {Interlock::RegionEvidenceBelowThreshold,
         [](PerceptionInput& input) { input.region_evidence = 0.2; }},
    };

    for (const auto& [expected, mutate] : cases) {
        MockActuator actuator;
        PolicyEngine policy(fast_config(), actuator);
        (void)arm(policy);
        DecisionOutput output{};
        for (std::uint64_t index = 0; index < 8; ++index) {
            auto perception = harmful_frame(1 + index, 10 + index * 20);
            mutate(perception);
            output = frame(policy, std::move(perception));
        }
        REQUIRE(has_interlock(output, expected));
        REQUIRE(policy.state() == PolicyState::Tracking);
        REQUIRE(actuator.count(ActionType::GotoPreset) == 0);
        REQUIRE(actuator.count(ActionType::Burst) == 0);
    }
}

void stale_safety_snapshot_latches_fault() {
    MockActuator actuator;
    auto config = fast_config();
    config.safety_stale_after_ms = 25;
    PolicyEngine policy(config, actuator);
    (void)arm(policy);

    const auto output = policy.step(PolicyInput{
        .now_ms = 30,
        .perception = harmful_frame(1, 30),
    });
    REQUIRE(output.state_after == PolicyState::Fault);
    REQUIRE(has_interlock(output, Interlock::StaleSafetyInput));
    REQUIRE(actuator.count(ActionType::EmergencyStop) == 1);
}

void missing_burst_ack_is_never_retried() {
    MockActuator actuator;
    actuator.enqueue_result(ActionType::Burst, ActionStatus::TimedOut, "NO_RESPONSE");
    PolicyEngine policy(fast_config(), actuator);
    (void)arm(policy);

    const auto failed = drive_to_burst(policy);
    REQUIRE(failed.state_after == PolicyState::Fault);
    REQUIRE(failed.event_burst_attempted);
    REQUIRE(actuator.count(ActionType::Burst) == 1);

    for (std::uint64_t index = 0; index < 10; ++index) {
        const auto at_ms = static_cast<TimestampMs>(80 + index * 10);
        (void)frame(policy, harmful_frame(7 + index, at_ms));
    }
    REQUIRE(actuator.count(ActionType::Burst) == 1);

    const auto disarmed = policy.step(PolicyInput{
        .now_ms = 200,
        .safety = safe_at(200),
        .control = ControlRequest::Disarm,
    });
    REQUIRE(disarmed.state_after == PolicyState::Disarmed);
    REQUIRE(arm(policy, 201).state_after == PolicyState::Monitoring);
    const auto suppressed = frame(policy, harmful_frame(20, 202));
    REQUIRE(has_interlock(suppressed, Interlock::EventAlreadyActed));
    REQUIRE(actuator.count(ActionType::Burst) == 1);
}

void mock_actuator_deduplicates_command_ids() {
    MockActuator actuator;
    const ActionCommand command{
        .type = ActionType::Arm,
        .command_id = 42,
        .issued_at_ms = 5,
    };
    const auto first = actuator.execute(command);
    const auto second = actuator.execute(command);
    REQUIRE(first.acknowledged());
    REQUIRE(!first.duplicate);
    REQUIRE(second.acknowledged());
    REQUIRE(second.duplicate);
    REQUIRE(actuator.unique_commands().size() == 1);
    REQUIRE(actuator.invocations().size() == 2);

    auto collision = command;
    collision.type = ActionType::Hold;
    const auto denied = actuator.execute(collision);
    REQUIRE(denied.status == ActionStatus::Denied);
    REQUIRE(denied.duplicate);
    REQUIRE(denied.reason == "COMMAND_ID_COLLISION");
    REQUIRE(actuator.unique_commands().size() == 1);
}

void non_monotonic_time_faults_and_estops_at_last_good_time() {
    MockActuator actuator;
    PolicyEngine policy(fast_config(), actuator);
    (void)arm(policy, 100);
    const auto before = actuator.unique_commands().size();
    const auto output = policy.step(PolicyInput{
        .now_ms = 99,
        .safety = safe_at(99),
    });
    REQUIRE(output.state_after == PolicyState::Fault);
    REQUIRE(has_interlock(output, Interlock::NonMonotonicTime));
    REQUIRE(actuator.unique_commands().size() == before + 1);
    REQUIRE(actuator.unique_commands().back().type == ActionType::EmergencyStop);
    REQUIRE(actuator.unique_commands().back().issued_at_ms == 100);
}

void invalid_configuration_is_rejected() {
    MockActuator actuator;
    auto config = fast_config();
    config.burst_duration_ms = config.hard_max_burst_duration_ms + 1;
    bool threw = false;
    try {
        PolicyEngine policy(config, actuator);
        (void)policy;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    REQUIRE(threw);
}

void final_command_id_is_reserved_for_estop() {
    MockActuator actuator;
    auto config = fast_config();
    config.initial_command_id = std::numeric_limits<std::uint64_t>::max();
    PolicyEngine policy(config, actuator);

    const auto output = arm(policy);
    REQUIRE(output.state_after == PolicyState::Fault);
    REQUIRE(has_interlock(output, Interlock::CommandIdExhausted));
    REQUIRE(actuator.count(ActionType::Arm) == 0);
    REQUIRE(actuator.count(ActionType::EmergencyStop) == 1);
    REQUIRE(actuator.unique_commands().front().command_id ==
            std::numeric_limits<std::uint64_t>::max());
}

struct TestCase {
    std::string_view name;
    void (*run)();
};

}  // namespace

int main() {
    const std::vector<TestCase> tests{
        {"startup_requires_explicit_arm", startup_requires_explicit_arm},
        {"unexpected_physical_arm_state_is_failed_closed",
         unexpected_physical_arm_state_is_failed_closed},
        {"deterministic_happy_path_is_auditable", deterministic_happy_path_is_auditable},
        {"continuous_event_gets_at_most_one_burst", continuous_event_gets_at_most_one_burst},
        {"cleared_event_can_trigger_after_cooldown", cleared_event_can_trigger_after_cooldown},
        {"transient_interlock_aborts_aim_with_hold", transient_interlock_aborts_aim_with_hold},
        {"all_required_perception_interlocks_fail_closed",
         all_required_perception_interlocks_fail_closed},
        {"stale_and_duplicate_frames_cannot_confirm", stale_and_duplicate_frames_cannot_confirm},
        {"hardware_not_ready_latches_fault_and_estop",
         hardware_not_ready_latches_fault_and_estop},
        {"actuator_must_remain_armed_while_policy_is_active",
         actuator_must_remain_armed_while_policy_is_active},
        {"repeated_arm_request_cannot_bypass_safety_checks",
         repeated_arm_request_cannot_bypass_safety_checks},
        {"behavior_and_region_uncertainty_never_reach_aiming",
         behavior_and_region_uncertainty_never_reach_aiming},
        {"stale_safety_snapshot_latches_fault", stale_safety_snapshot_latches_fault},
        {"missing_burst_ack_is_never_retried", missing_burst_ack_is_never_retried},
        {"mock_actuator_deduplicates_command_ids", mock_actuator_deduplicates_command_ids},
        {"non_monotonic_time_faults_and_estops_at_last_good_time",
         non_monotonic_time_faults_and_estops_at_last_good_time},
        {"invalid_configuration_is_rejected", invalid_configuration_is_rejected},
        {"final_command_id_is_reserved_for_estop", final_command_id_is_reserved_for_estop},
    };

    std::size_t failures = 0;
    for (const auto& test : tests) {
        try {
            test.run();
            std::cout << "PASS " << test.name << '\n';
        } catch (const std::exception& exception) {
            ++failures;
            std::cerr << "FAIL " << test.name << ": " << exception.what() << '\n';
        }
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << tests.size() << " test(s) passed\n";
    return 0;
}
