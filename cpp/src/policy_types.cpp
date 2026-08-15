#include "foliage/actuator.hpp"
#include "foliage/policy_types.hpp"

namespace foliage {

std::string_view to_string(const PolicyState value) noexcept {
    switch (value) {
        case PolicyState::Disarmed: return "DISARMED";
        case PolicyState::Monitoring: return "MONITORING";
        case PolicyState::Tracking: return "TRACKING";
        case PolicyState::Confirming: return "CONFIRMING";
        case PolicyState::Aiming: return "AIMING";
        case PolicyState::Ready: return "READY";
        case PolicyState::Burst: return "BURST";
        case PolicyState::Cooldown: return "COOLDOWN";
        case PolicyState::Fault: return "FAULT";
    }
    return "INVALID_POLICY_STATE";
}

std::string_view to_string(const Behavior value) noexcept {
    switch (value) {
        case Behavior::Clear: return "CLEAR";
        case Behavior::Eating: return "EATING";
        case Behavior::Digging: return "DIGGING";
        case Behavior::Unknown: return "UNKNOWN";
    }
    return "INVALID_BEHAVIOR";
}

std::string_view to_string(const Interlock value) noexcept {
    switch (value) {
        case Interlock::None: return "NONE";
        case Interlock::MissingSafetyInput: return "MISSING_SAFETY_INPUT";
        case Interlock::StaleSafetyInput: return "STALE_SAFETY_INPUT";
        case Interlock::InvalidSafetyInput: return "INVALID_SAFETY_INPUT";
        case Interlock::HardwareNotReady: return "HARDWARE_NOT_READY";
        case Interlock::WatchdogUnhealthy: return "WATCHDOG_UNHEALTHY";
        case Interlock::CalibrationInvalid: return "CALIBRATION_INVALID";
        case Interlock::EmergencyStopActive: return "EMERGENCY_STOP_ACTIVE";
        case Interlock::ActuatorFault: return "ACTUATOR_FAULT";
        case Interlock::ActuatorDisconnected: return "ACTUATOR_DISCONNECTED";
        case Interlock::ActuatorNotArmed: return "ACTUATOR_NOT_ARMED";
        case Interlock::ActuatorUnexpectedlyArmed: return "ACTUATOR_UNEXPECTEDLY_ARMED";
        case Interlock::MissingPerception: return "MISSING_PERCEPTION";
        case Interlock::StalePerception: return "STALE_PERCEPTION";
        case Interlock::OutOfOrderPerception: return "OUT_OF_ORDER_PERCEPTION";
        case Interlock::InvalidPerception: return "INVALID_PERCEPTION";
        case Interlock::PersonPresent: return "PERSON_PRESENT";
        case Interlock::MultipleCats: return "MULTIPLE_CATS";
        case Interlock::AmbiguousCats: return "AMBIGUOUS_CATS";
        case Interlock::MissingTrack: return "MISSING_TRACK";
        case Interlock::OutsideProtectedZone: return "OUTSIDE_PROTECTED_ZONE";
        case Interlock::PoorTracking: return "POOR_TRACKING";
        case Interlock::UnknownBehavior: return "UNKNOWN_BEHAVIOR";
        case Interlock::BehaviorNotHarmful: return "BEHAVIOR_NOT_HARMFUL";
        case Interlock::BehaviorBelowThreshold: return "BEHAVIOR_BELOW_THRESHOLD";
        case Interlock::RegionEvidenceBelowThreshold: return "REGION_EVIDENCE_BELOW_THRESHOLD";
        case Interlock::NoFireIntersection: return "NO_FIRE_INTERSECTION";
        case Interlock::MissingAimPreset: return "MISSING_AIM_PRESET";
        case Interlock::EventAlreadyActed: return "EVENT_ALREADY_ACTED";
        case Interlock::AwaitingNewFrame: return "AWAITING_NEW_FRAME";
        case Interlock::CooldownActive: return "COOLDOWN_ACTIVE";
        case Interlock::NonMonotonicTime: return "NON_MONOTONIC_TIME";
        case Interlock::CommandIdExhausted: return "COMMAND_ID_EXHAUSTED";
    }
    return "INVALID_INTERLOCK";
}

std::string_view to_string(const DecisionCode value) noexcept {
    switch (value) {
        case DecisionCode::NoChange: return "NO_CHANGE";
        case DecisionCode::Armed: return "ARMED";
        case DecisionCode::Disarmed: return "DISARMED";
        case DecisionCode::TrackingStarted: return "TRACKING_STARTED";
        case DecisionCode::ConfirmationStarted: return "CONFIRMATION_STARTED";
        case DecisionCode::Hold: return "HOLD";
        case DecisionCode::AimRequested: return "AIM_REQUESTED";
        case DecisionCode::Ready: return "READY";
        case DecisionCode::BurstAttempted: return "BURST_ATTEMPTED";
        case DecisionCode::Cooldown: return "COOLDOWN";
        case DecisionCode::Faulted: return "FAULTED";
    }
    return "INVALID_DECISION";
}

std::string_view to_string(const TransitionReason value) noexcept {
    switch (value) {
        case TransitionReason::ArmAcknowledged: return "ARM_ACKNOWLEDGED";
        case TransitionReason::OperatorDisarm: return "OPERATOR_DISARM";
        case TransitionReason::EmergencyStop: return "EMERGENCY_STOP";
        case TransitionReason::CatEnteredProtectedZone: return "CAT_ENTERED_PROTECTED_ZONE";
        case TransitionReason::TrackingPersistent: return "TRACKING_PERSISTENT";
        case TransitionReason::HarmfulBehaviorPersistent: return "HARMFUL_BEHAVIOR_PERSISTENT";
        case TransitionReason::AimAcknowledged: return "AIM_ACKNOWLEDGED";
        case TransitionReason::BurstAttempted: return "BURST_ATTEMPTED";
        case TransitionReason::BurstAcknowledged: return "BURST_ACKNOWLEDGED";
        case TransitionReason::CooldownElapsed: return "COOLDOWN_ELAPSED";
        case TransitionReason::EventCleared: return "EVENT_CLEARED";
        case TransitionReason::InterlockHold: return "INTERLOCK_HOLD";
        case TransitionReason::SafetyFault: return "SAFETY_FAULT";
        case TransitionReason::ActionFailure: return "ACTION_FAILURE";
        case TransitionReason::InvalidInput: return "INVALID_INPUT";
    }
    return "INVALID_TRANSITION_REASON";
}

std::string_view to_string(const ActionType value) noexcept {
    switch (value) {
        case ActionType::Arm: return "ARM";
        case ActionType::Disarm: return "DISARM";
        case ActionType::Home: return "HOME";
        case ActionType::GotoPreset: return "GOTO_PRESET";
        case ActionType::PanLeft: return "PAN_LEFT";
        case ActionType::PanRight: return "PAN_RIGHT";
        case ActionType::TiltUp: return "TILT_UP";
        case ActionType::TiltDown: return "TILT_DOWN";
        case ActionType::Hold: return "HOLD";
        case ActionType::Burst: return "BURST";
        case ActionType::EmergencyStop: return "ESTOP";
        case ActionType::Status: return "STATUS";
    }
    return "INVALID_ACTION_TYPE";
}

std::string_view to_string(const ActionStatus value) noexcept {
    switch (value) {
        case ActionStatus::Acknowledged: return "ACKNOWLEDGED";
        case ActionStatus::Denied: return "DENIED";
        case ActionStatus::Failed: return "FAILED";
        case ActionStatus::TimedOut: return "TIMED_OUT";
    }
    return "INVALID_ACTION_STATUS";
}

}  // namespace foliage
