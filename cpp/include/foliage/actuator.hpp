#pragma once

#include "foliage/policy_types.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace foliage {

enum class ActionType {
    Arm,
    Disarm,
    Home,
    GotoPreset,
    PanLeft,
    PanRight,
    TiltUp,
    TiltDown,
    Hold,
    Burst,
    EmergencyStop,
    Status,
};

enum class ActionStatus {
    Acknowledged,
    Denied,
    Failed,
    TimedOut,
};

struct ActionCommand {
    ActionType type{ActionType::Hold};
    std::string target{};
    float amount{0.0F};
    std::uint32_t duration_ms{0};
    std::uint64_t command_id{0};
    TimestampMs issued_at_ms{0};
    std::optional<EventId> event_id{};

    [[nodiscard]] bool operator==(const ActionCommand&) const = default;
};

struct ActionResult {
    std::uint64_t command_id{0};
    ActionStatus status{ActionStatus::Failed};
    TimestampMs completed_at_ms{0};
    std::string reason{};
    bool duplicate{false};

    [[nodiscard]] bool acknowledged() const noexcept {
        return status == ActionStatus::Acknowledged;
    }
};

struct ActuatorStatus {
    TimestampMs observed_at_ms{0};
    bool connected{false};
    bool ready{false};
    bool armed{false};
    bool emergency_stop{false};
    bool fault{true};
};

struct ActionRecord {
    ActionCommand command{};
    ActionResult result{};
};

class IActuator {
public:
    virtual ~IActuator() = default;
    virtual ActionResult execute(const ActionCommand& command) = 0;
    [[nodiscard]] virtual ActuatorStatus status() const noexcept = 0;
};

[[nodiscard]] std::string_view to_string(ActionType value) noexcept;
[[nodiscard]] std::string_view to_string(ActionStatus value) noexcept;

}  // namespace foliage
