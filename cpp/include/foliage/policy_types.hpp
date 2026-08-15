#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace foliage {

using TimestampMs = std::uint64_t;
using FrameId = std::uint64_t;
using TrackId = std::uint64_t;
using EventId = std::uint64_t;

enum class PolicyState {
    Disarmed,
    Monitoring,
    Tracking,
    Confirming,
    Aiming,
    Ready,
    Burst,
    Cooldown,
    Fault,
};

enum class Behavior {
    Clear,
    Eating,
    Digging,
    Unknown,
};

enum class ControlRequest {
    None,
    Arm,
    Disarm,
    EmergencyStop,
};

enum class Interlock {
    None,
    MissingSafetyInput,
    StaleSafetyInput,
    InvalidSafetyInput,
    HardwareNotReady,
    WatchdogUnhealthy,
    CalibrationInvalid,
    EmergencyStopActive,
    ActuatorFault,
    ActuatorDisconnected,
    ActuatorNotArmed,
    ActuatorUnexpectedlyArmed,
    MissingPerception,
    StalePerception,
    OutOfOrderPerception,
    InvalidPerception,
    PersonPresent,
    MultipleCats,
    AmbiguousCats,
    MissingTrack,
    OutsideProtectedZone,
    PoorTracking,
    UnknownBehavior,
    BehaviorNotHarmful,
    BehaviorBelowThreshold,
    RegionEvidenceBelowThreshold,
    NoFireIntersection,
    MissingAimPreset,
    EventAlreadyActed,
    AwaitingNewFrame,
    CooldownActive,
    NonMonotonicTime,
    CommandIdExhausted,
};

enum class DecisionCode {
    NoChange,
    Armed,
    Disarmed,
    TrackingStarted,
    ConfirmationStarted,
    Hold,
    AimRequested,
    Ready,
    BurstAttempted,
    Cooldown,
    Faulted,
};

enum class TransitionReason {
    ArmAcknowledged,
    OperatorDisarm,
    EmergencyStop,
    CatEnteredProtectedZone,
    TrackingPersistent,
    HarmfulBehaviorPersistent,
    AimAcknowledged,
    BurstAttempted,
    BurstAcknowledged,
    CooldownElapsed,
    EventCleared,
    InterlockHold,
    SafetyFault,
    ActionFailure,
    InvalidInput,
};

struct PerceptionInput {
    TimestampMs observed_at_ms{0};
    FrameId frame_id{0};
    std::uint32_t cats_in_protected_zone{0};
    bool cats_ambiguous{false};
    bool person_present{false};
    std::optional<TrackId> primary_track_id{};
    Behavior behavior{Behavior::Unknown};
    double behavior_confidence{0.0};
    double region_evidence{0.0};
    double track_quality{0.0};
    bool no_fire_intersection{true};
    std::optional<std::string> safe_aim_preset{};
};

struct SafetyInput {
    TimestampMs observed_at_ms{0};
    bool hardware_ready{false};
    bool watchdog_healthy{false};
    bool calibration_valid{false};
    bool emergency_stop{false};
    bool actuator_fault{true};
};

struct PolicyInput {
    TimestampMs now_ms{0};
    std::optional<PerceptionInput> perception{};
    std::optional<SafetyInput> safety{};
    ControlRequest control{ControlRequest::None};
};

struct PolicyConfig {
    TimestampMs tracking_persistence_ms{300};
    std::uint32_t minimum_tracking_frames{3};
    TimestampMs confirmation_persistence_ms{1'200};
    std::uint32_t minimum_confirmation_frames{4};
    TimestampMs aim_settle_ms{100};
    TimestampMs event_clear_persistence_ms{1'500};
    std::uint32_t minimum_event_clear_frames{3};
    TimestampMs perception_stale_after_ms{350};
    TimestampMs safety_stale_after_ms{1'000};
    TimestampMs cooldown_ms{30'000};
    double minimum_track_quality{0.70};
    double minimum_behavior_confidence{0.85};
    double minimum_region_evidence{0.65};
    std::uint32_t burst_duration_ms{75};
    std::uint32_t hard_max_burst_duration_ms{100};
    std::uint64_t initial_command_id{1};
};

struct TransitionRecord {
    PolicyState from{PolicyState::Disarmed};
    PolicyState to{PolicyState::Disarmed};
    TransitionReason reason{TransitionReason::InvalidInput};
    TimestampMs at_ms{0};
};

[[nodiscard]] std::string_view to_string(PolicyState value) noexcept;
[[nodiscard]] std::string_view to_string(Behavior value) noexcept;
[[nodiscard]] std::string_view to_string(Interlock value) noexcept;
[[nodiscard]] std::string_view to_string(DecisionCode value) noexcept;
[[nodiscard]] std::string_view to_string(TransitionReason value) noexcept;

}  // namespace foliage
