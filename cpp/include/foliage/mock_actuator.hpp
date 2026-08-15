#pragma once

#include "foliage/actuator.hpp"

#include <deque>
#include <map>
#include <unordered_map>
#include <vector>

namespace foliage {

class MockActuator final : public IActuator {
public:
    explicit MockActuator(ActuatorStatus initial_status = healthy_status());

    ActionResult execute(const ActionCommand& command) override;
    [[nodiscard]] ActuatorStatus status() const noexcept override;

    void set_status(const ActuatorStatus& status);
    void enqueue_result(ActionType type, ActionStatus status, std::string reason = {});

    [[nodiscard]] const std::vector<ActionCommand>& invocations() const noexcept;
    [[nodiscard]] const std::vector<ActionCommand>& unique_commands() const noexcept;
    [[nodiscard]] std::size_t count(ActionType type) const noexcept;

    [[nodiscard]] static ActuatorStatus healthy_status() noexcept;

private:
    struct ScriptedResult {
        ActionStatus status{ActionStatus::Acknowledged};
        std::string reason{};
    };

    struct CachedExecution {
        ActionCommand command{};
        ActionResult result{};
    };

    [[nodiscard]] ActionResult default_result(const ActionCommand& command) const;
    void apply_acknowledged_effect(const ActionCommand& command);

    ActuatorStatus status_{};
    std::map<ActionType, std::deque<ScriptedResult>> scripted_results_{};
    std::unordered_map<std::uint64_t, CachedExecution> executions_{};
    std::vector<ActionCommand> invocations_{};
    std::vector<ActionCommand> unique_commands_{};
};

}  // namespace foliage
