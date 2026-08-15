#include "foliage/mock_actuator.hpp"

#include <algorithm>
#include <utility>

namespace foliage {

MockActuator::MockActuator(const ActuatorStatus initial_status) : status_(initial_status) {}

ActionResult MockActuator::execute(const ActionCommand& command) {
    invocations_.push_back(command);

    if (const auto found = executions_.find(command.command_id); found != executions_.end()) {
        if (found->second.command == command) {
            auto duplicate = found->second.result;
            duplicate.duplicate = true;
            return duplicate;
        }
        return ActionResult{
            .command_id = command.command_id,
            .status = ActionStatus::Denied,
            .completed_at_ms = command.issued_at_ms,
            .reason = "COMMAND_ID_COLLISION",
            .duplicate = true,
        };
    }

    ActionResult result{};
    if (auto& queue = scripted_results_[command.type]; !queue.empty()) {
        auto scripted = std::move(queue.front());
        queue.pop_front();
        result = ActionResult{
            .command_id = command.command_id,
            .status = scripted.status,
            .completed_at_ms = command.issued_at_ms,
            .reason = std::move(scripted.reason),
            .duplicate = false,
        };
    } else {
        result = default_result(command);
    }

    unique_commands_.push_back(command);
    executions_.emplace(command.command_id, CachedExecution{command, result});
    if (result.acknowledged()) {
        apply_acknowledged_effect(command);
    }
    return result;
}

ActuatorStatus MockActuator::status() const noexcept {
    return status_;
}

void MockActuator::set_status(const ActuatorStatus& status) {
    status_ = status;
}

void MockActuator::enqueue_result(const ActionType type, const ActionStatus status,
                                  std::string reason) {
    scripted_results_[type].push_back(ScriptedResult{status, std::move(reason)});
}

const std::vector<ActionCommand>& MockActuator::invocations() const noexcept {
    return invocations_;
}

const std::vector<ActionCommand>& MockActuator::unique_commands() const noexcept {
    return unique_commands_;
}

std::size_t MockActuator::count(const ActionType type) const noexcept {
    return static_cast<std::size_t>(std::count_if(
        unique_commands_.begin(), unique_commands_.end(),
        [type](const ActionCommand& command) { return command.type == type; }));
}

ActuatorStatus MockActuator::healthy_status() noexcept {
    return ActuatorStatus{
        .observed_at_ms = 0,
        .connected = true,
        .ready = true,
        .armed = false,
        .emergency_stop = false,
        .fault = false,
    };
}

ActionResult MockActuator::default_result(const ActionCommand& command) const {
    ActionStatus result = ActionStatus::Acknowledged;
    std::string reason{};

    if (!status_.connected) {
        result = ActionStatus::TimedOut;
        reason = "DISCONNECTED";
    } else if (status_.fault && command.type != ActionType::Disarm &&
               command.type != ActionType::EmergencyStop) {
        result = ActionStatus::Failed;
        reason = "ACTUATOR_FAULT";
    } else if (!status_.ready && command.type != ActionType::Disarm &&
               command.type != ActionType::EmergencyStop) {
        result = ActionStatus::Denied;
        reason = "NOT_READY";
    } else if (status_.emergency_stop && command.type != ActionType::Disarm &&
               command.type != ActionType::EmergencyStop) {
        result = ActionStatus::Denied;
        reason = "ESTOP_ACTIVE";
    } else if ((command.type == ActionType::GotoPreset || command.type == ActionType::Burst) &&
               !status_.armed) {
        result = ActionStatus::Denied;
        reason = "NOT_ARMED";
    } else if (command.type == ActionType::GotoPreset && command.target.empty()) {
        result = ActionStatus::Denied;
        reason = "MISSING_TARGET";
    } else if (command.type == ActionType::Burst && command.duration_ms == 0) {
        result = ActionStatus::Denied;
        reason = "INVALID_DURATION";
    }

    return ActionResult{
        .command_id = command.command_id,
        .status = result,
        .completed_at_ms = command.issued_at_ms,
        .reason = std::move(reason),
        .duplicate = false,
    };
}

void MockActuator::apply_acknowledged_effect(const ActionCommand& command) {
    status_.observed_at_ms = command.issued_at_ms;
    switch (command.type) {
        case ActionType::Arm:
            status_.armed = true;
            break;
        case ActionType::Disarm:
            status_.armed = false;
            status_.emergency_stop = false;
            break;
        case ActionType::EmergencyStop:
            status_.armed = false;
            status_.emergency_stop = true;
            break;
        default:
            break;
    }
}

}  // namespace foliage
