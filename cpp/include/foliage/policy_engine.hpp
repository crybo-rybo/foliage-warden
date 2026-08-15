#pragma once

#include "foliage/actuator.hpp"
#include "foliage/policy_types.hpp"

#include <optional>
#include <vector>

namespace foliage {

struct DecisionOutput {
    TimestampMs at_ms{0};
    PolicyState state_before{PolicyState::Disarmed};
    PolicyState state_after{PolicyState::Disarmed};
    DecisionCode decision{DecisionCode::NoChange};
    std::vector<Interlock> interlocks{};
    std::vector<TransitionRecord> transitions{};
    std::vector<ActionRecord> actions{};
    std::optional<EventId> event_id{};
    bool event_active{false};
    bool event_burst_attempted{false};
    std::string detail{};
};

class PolicyEngine {
public:
    PolicyEngine(PolicyConfig config, IActuator& actuator);
    PolicyEngine(const PolicyEngine&) = delete;
    PolicyEngine& operator=(const PolicyEngine&) = delete;
    PolicyEngine(PolicyEngine&&) = delete;
    PolicyEngine& operator=(PolicyEngine&&) = delete;

    [[nodiscard]] DecisionOutput step(const PolicyInput& input);

    [[nodiscard]] PolicyState state() const noexcept;
    [[nodiscard]] const PolicyConfig& config() const noexcept;
    [[nodiscard]] std::optional<EventId> current_event_id() const noexcept;
    [[nodiscard]] bool current_event_burst_attempted() const noexcept;

private:
    struct Candidate {
        TrackId track_id{0};
        Behavior behavior{Behavior::Unknown};
        TimestampMs started_at_ms{0};
        TimestampMs latest_at_ms{0};
        std::uint32_t frame_count{0};
        FrameId last_frame_id{0};
    };

    struct EventLatch {
        EventId id{0};
        bool active{false};
        bool burst_attempted{false};
        TimestampMs started_at_ms{0};
        std::optional<TimestampMs> clear_started_at_ms{};
        std::uint32_t clear_frame_count{0};
        FrameId last_clear_frame_id{0};
    };

    [[nodiscard]] bool validate_config() const noexcept;
    void cache_inputs(const PolicyInput& input, DecisionOutput& output);
    void update_event_latch(const PerceptionInput& perception);
    [[nodiscard]] std::vector<Interlock> hard_interlocks(TimestampMs now_ms) const;
    [[nodiscard]] std::vector<Interlock> perception_interlocks(TimestampMs now_ms) const;
    [[nodiscard]] bool harmful_evidence(const PerceptionInput& perception,
                                        std::vector<Interlock>& interlocks) const;
    void reset_candidates() noexcept;
    void transition_to(PolicyState next, TransitionReason reason, DecisionOutput& output);
    void fault(Interlock interlock, TransitionReason reason, std::string detail,
               DecisionOutput& output);
    [[nodiscard]] std::optional<ActionResult> execute(ActionType type, TimestampMs now_ms,
                                                      DecisionOutput& output,
                                                      std::string target = {},
                                                      std::uint32_t duration_ms = 0);
    void abort_aiming(TimestampMs now_ms, const std::vector<Interlock>& interlocks,
                      DecisionOutput& output);
    [[nodiscard]] bool same_tracking_candidate(const PerceptionInput& perception) const noexcept;
    [[nodiscard]] bool same_confirmation_candidate(const PerceptionInput& perception) const noexcept;
    void begin_tracking(const PerceptionInput& perception);
    void advance_tracking(const PerceptionInput& perception);
    void begin_confirmation(const PerceptionInput& perception);
    void advance_confirmation(const PerceptionInput& perception);
    [[nodiscard]] bool tracking_is_persistent() const noexcept;
    [[nodiscard]] bool confirmation_is_persistent() const noexcept;
    void finish_output(DecisionOutput& output) const;

    PolicyConfig config_{};
    IActuator& actuator_;
    PolicyState state_{PolicyState::Disarmed};
    std::optional<TimestampMs> last_now_ms_{};
    std::optional<PerceptionInput> latest_perception_{};
    std::optional<SafetyInput> latest_safety_{};
    bool incoming_frame_is_new_{false};
    bool perception_invalid_{false};
    bool safety_invalid_{false};
    bool perception_out_of_order_{false};
    Candidate tracking_candidate_{};
    Candidate confirmation_candidate_{};
    bool has_tracking_candidate_{false};
    bool has_confirmation_candidate_{false};
    EventLatch event_{};
    EventId next_event_id_{1};
    std::uint64_t next_command_id_{1};
    bool command_ids_exhausted_{false};
    bool aim_acknowledged_{false};
    TimestampMs aim_acknowledged_at_ms_{0};
    FrameId ready_after_frame_id_{0};
    bool burst_acknowledged_{false};
    TimestampMs cooldown_until_ms_{0};
};

}  // namespace foliage
