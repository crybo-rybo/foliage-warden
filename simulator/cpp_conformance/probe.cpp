#include "foliage/mock_actuator.hpp"
#include "foliage/policy_engine.hpp"

#include <algorithm>
#include <iostream>
#include <utility>

namespace {

using namespace foliage;

PolicyConfig config() {
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

SafetyInput safety(const TimestampMs at_ms) {
    return SafetyInput{
        .observed_at_ms = at_ms,
        .hardware_ready = true,
        .watchdog_healthy = true,
        .calibration_valid = true,
        .emergency_stop = false,
        .actuator_fault = false,
    };
}

PerceptionInput harmful(const FrameId frame_id, const TimestampMs at_ms) {
    return PerceptionInput{
        .observed_at_ms = at_ms,
        .frame_id = frame_id,
        .cats_in_protected_zone = 1,
        .cats_ambiguous = false,
        .person_present = false,
        .primary_track_id = 7,
        .behavior = Behavior::Eating,
        .behavior_confidence = 0.95,
        .region_evidence = 0.90,
        .track_quality = 0.95,
        .no_fire_intersection = false,
        .safe_aim_preset = "pot-1-safe-nearby",
    };
}

DecisionOutput arm(PolicyEngine& policy) {
    return policy.step(PolicyInput{
        .now_ms = 0,
        .safety = safety(0),
        .control = ControlRequest::Arm,
    });
}

DecisionOutput frame(PolicyEngine& policy, PerceptionInput perception) {
    const auto at_ms = perception.observed_at_ms;
    return policy.step(PolicyInput{
        .now_ms = at_ms,
        .perception = std::move(perception),
        .safety = safety(at_ms),
    });
}

bool contains(const DecisionOutput& output, const Interlock expected) {
    return std::find(output.interlocks.begin(), output.interlocks.end(), expected) !=
           output.interlocks.end();
}

}  // namespace

int main() {
    MockActuator person_actuator;
    PolicyEngine person_policy(config(), person_actuator);
    const auto startup = person_policy.state();
    const auto armed = arm(person_policy);
    auto person = harmful(1, 10);
    person.person_present = true;
    const auto person_result = frame(person_policy, std::move(person));

    MockActuator action_actuator;
    PolicyEngine action_policy(config(), action_actuator);
    (void)arm(action_policy);
    (void)frame(action_policy, harmful(1, 10));
    (void)frame(action_policy, harmful(2, 30));
    (void)frame(action_policy, harmful(3, 40));
    (void)frame(action_policy, harmful(4, 50));
    (void)frame(action_policy, harmful(5, 60));
    (void)frame(action_policy, harmful(6, 70));
    (void)action_policy.step(PolicyInput{.now_ms = 71, .safety = safety(71)});
    const auto original_burst = action_actuator.unique_commands().back();
    const auto duplicate = action_actuator.execute(original_burst);

    const bool passed =
        startup == PolicyState::Disarmed && armed.state_after == PolicyState::Monitoring &&
        person_result.state_after == PolicyState::Monitoring &&
        contains(person_result, Interlock::PersonPresent) &&
        action_actuator.count(ActionType::Burst) == 1 && duplicate.duplicate;

    std::cout << "{\"after_arm\":\"" << to_string(armed.state_after)
              << "\",\"burst_count\":" << action_actuator.count(ActionType::Burst)
              << ",\"duplicate_suppressed\":"
              << (duplicate.duplicate ? "true" : "false")
              << ",\"person_interlock\":"
              << (contains(person_result, Interlock::PersonPresent) ? "true" : "false")
              << ",\"person_state\":\"" << to_string(person_result.state_after)
              << "\",\"startup\":\"" << to_string(startup) << "\"}\n";
    return passed ? 0 : 1;
}
