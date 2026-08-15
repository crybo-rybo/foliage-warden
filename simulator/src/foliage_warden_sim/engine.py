"""Virtual-time policy, incident, and mock-transport reference implementation.

This module is intentionally incapable of physical actuation.  Its only adapter is
an in-memory scripted mock whose outcomes are scheduled on the virtual clock.
"""

from __future__ import annotations

import heapq
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from .types import (
    Assertion,
    Incident,
    InternalWork,
    JsonObject,
    PendingCommand,
    Persistence,
)
from .validation import MAX_SAFE_INTEGER, LoadedContracts, load_contracts

COUNT_NAMES = (
    "ready_transitions",
    "would_burst_decisions",
    "burst_commands_issued",
    "burst_commands_acked",
    "physical_bursts",
    "automatic_retries",
    "duplicate_commands_suppressed",
    "fault_transitions",
)


class SimulationError(RuntimeError):
    """The validated scenario cannot be executed deterministically."""


@dataclass(slots=True)
class RunResult:
    scenario_id: str
    config_id: str
    config_sha256: str
    final_state: str
    final_clock_ms: int
    counts: JsonObject
    visited_states: tuple[str, ...]
    reason_codes: tuple[str, ...]
    action_sequence: tuple[JsonObject, ...]
    assertions: tuple[Assertion, ...]
    event_records: tuple[JsonObject, ...]
    audit_records: tuple[JsonObject, ...]
    evaluator_records: tuple[JsonObject, ...]
    deterministic_replay_verified: bool = False

    @property
    def passed(self) -> bool:
        return all(assertion.passed for assertion in self.assertions)

    def summary(self) -> JsonObject:
        return {
            "action_sequence": list(self.action_sequence),
            "assertions": [assertion.to_dict() for assertion in self.assertions],
            "config_id": self.config_id,
            "config_sha256": self.config_sha256,
            "counts": self.counts,
            "deterministic_replay_verified": self.deterministic_replay_verified,
            "final_clock_ms": self.final_clock_ms,
            "final_state": self.final_state,
            "passed": self.passed,
            "reason_codes": list(self.reason_codes),
            "scenario_id": self.scenario_id,
            "schema_version": 1,
            "visited_states": list(self.visited_states),
        }

    def deterministic_signature(self) -> str:
        value = {
            "action_sequence": self.action_sequence,
            "audit_records": self.audit_records,
            "counts": self.counts,
            "event_records": self.event_records,
            "evaluator_records": self.evaluator_records,
            "final_clock_ms": self.final_clock_ms,
            "final_state": self.final_state,
            "reason_codes": self.reason_codes,
            "visited_states": self.visited_states,
        }
        encoded = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return sha256(encoded).hexdigest()


class Simulator:
    """Execute one validated scenario against the pure in-memory reference policy."""

    def __init__(self, contracts: LoadedContracts):
        self.contracts = contracts
        self.scenario = contracts.scenario
        self.config = contracts.config
        initial = self.scenario["initial_conditions"]
        self.clock_ms = 0
        self.state = "DISARMED"
        self.armed = False
        self.camera_status = initial["camera_status"]
        self.actuator_status = initial["actuator_status"]
        self.next_command_id = initial["command_id_seed"]
        self.epoch = 0
        self.camera_generation = 0
        self.camera_disconnected_at_ms: int | None = None

        self.counts: JsonObject = {name: 0 for name in COUNT_NAMES}
        self.visited_states: list[str] = ["DISARMED"]
        self.reason_codes: list[str] = []
        self.action_sequence: list[JsonObject] = []
        self.command_attempts: list[JsonObject] = []
        self.command_ledger: dict[int, JsonObject] = {}
        self.command_occurrences: Counter[str] = Counter()
        self.pending_commands: dict[int, PendingCommand] = {}
        self.persistence: dict[tuple[str, str, str], Persistence] = {}
        self.open_incidents: dict[tuple[str, str, str], Incident] = {}
        self.incidents: list[Incident] = []
        self.incident_counter = 0
        self.cooldown_until_ms: int | None = None
        self.last_observation: JsonObject | None = None
        self.last_observation_event_id: str | None = None
        self.last_observation_delivery_ms: int | None = None
        self.last_interlocks: JsonObject = {}
        self.restart_times: list[int] = []
        self.arm_times: list[int] = []

        self.internal: list[InternalWork] = []
        self.internal_order = 0
        self.record_sequence = 0
        self.audit_counter = 0
        self.fault_counter = 0
        self.event_records: list[JsonObject] = []
        self.audit_records: list[JsonObject] = []
        self.evaluator_actions: list[JsonObject] = []

        run_candidate = f"sim-{self.scenario['scenario_id']}"
        if len(run_candidate) > 128:
            run_candidate = f"sim-{sha256(run_candidate.encode()).hexdigest()[:24]}"
        self.run_id = run_candidate
        self._reason("STARTUP_DISARMED")
        self._event(
            "RUN_STARTED",
            "RUNTIME",
            {"reason_codes": ["STARTUP_DISARMED"], "summary": "virtual replay started"},
        )
        if self.camera_status == "DISCONNECTED":
            self._start_camera_disconnect_timer()

    def run(self) -> RunResult:
        inputs = self.contracts.inputs
        index = 0
        while index < len(inputs):
            at_ms = inputs[index].at_ms
            self._drain_internal_before(at_ms)
            self.clock_ms = at_ms
            while index < len(inputs) and inputs[index].at_ms == at_ms:
                self._apply_external(inputs[index].payload)
                index += 1
            self._drain_internal_at(at_ms)

        self._event(
            "RUN_STOPPED",
            "RUNTIME",
            {
                "reason_codes": list(self.reason_codes),
                "summary": f"virtual replay stopped in {self.state}",
            },
        )
        assertions = self._assert_expectations()
        evaluator_records = tuple(self._build_evaluator_records())
        return RunResult(
            scenario_id=self.scenario["scenario_id"],
            config_id=self.config["config_id"],
            config_sha256=self.contracts.config_sha256,
            final_state=self.state,
            final_clock_ms=self.clock_ms,
            counts=dict(self.counts),
            visited_states=tuple(self.visited_states),
            reason_codes=tuple(self.reason_codes),
            action_sequence=tuple(self.action_sequence),
            assertions=tuple(assertions),
            event_records=tuple(self.event_records),
            audit_records=tuple(self.audit_records),
            evaluator_records=evaluator_records,
        )

    # -- scheduling -----------------------------------------------------

    def _schedule(self, due_ms: int, kind: str, payload: JsonObject) -> None:
        if not self.clock_ms <= due_ms <= MAX_SAFE_INTEGER:
            raise SimulationError(
                f"invalid internal deadline {due_ms} at {self.clock_ms}"
            )
        self.internal_order += 1
        heapq.heappush(
            self.internal,
            InternalWork(due_ms, self.internal_order, kind, payload),
        )

    def _drain_internal_before(self, target_ms: int) -> None:
        while self.internal and self.internal[0].due_ms < target_ms:
            work = heapq.heappop(self.internal)
            self.clock_ms = work.due_ms
            self._apply_internal(work)

    def _drain_internal_at(self, target_ms: int) -> None:
        while self.internal and self.internal[0].due_ms == target_ms:
            work = heapq.heappop(self.internal)
            self.clock_ms = work.due_ms
            self._apply_internal(work)

    def _apply_internal(self, work: InternalWork) -> None:
        if work.kind == "CAMERA_TIMEOUT":
            if (
                work.payload["epoch"] == self.epoch
                and work.payload["generation"] == self.camera_generation
                and self.camera_status == "DISCONNECTED"
            ):
                self._reason("CAMERA_TIMEOUT")
                self._fault("CAMERA_TIMEOUT", "camera disconnect deadline elapsed")
            return

        command_id = work.payload["command_id"]
        pending = self.pending_commands.get(command_id)
        if (
            pending is None
            or pending.finished
            or pending.epoch != work.payload["epoch"]
        ):
            return
        if work.kind == "COMMAND_RESPONSE":
            self._finish_command(pending, work.payload["response"])
        elif work.kind == "COMMAND_TIMEOUT":
            self._finish_command(pending, "TIMEOUT")
        else:  # pragma: no cover - an internal programming error
            raise SimulationError(f"unknown internal work kind {work.kind!r}")

    # -- external inputs ------------------------------------------------

    def _apply_external(self, event: JsonObject) -> None:
        kind = event["type"]
        if kind == "CONTROL":
            self._control(event)
        elif kind == "OBSERVATION":
            self._observation(event)
        elif kind == "CAMERA_STATUS":
            self._camera_status(event)
        elif kind == "ACTUATOR_STATUS":
            self._actuator_status(event)
        elif kind == "PROCESS_RESTART":
            self._restart(event)
        elif kind == "INJECT_ACTION":
            self._input_record(event, accepted=True, reasons=[])
            self._dispatch(
                event["action"],
                trigger_event_id=event["event_id"],
                incident=None,
                observation=None,
                state_before=self.state,
                state_after=self.state,
            )
        elif kind == "TICK":
            self._input_record(event, accepted=True, reasons=[])
            self._settle_cooldown()
        else:  # schema validation should make this unreachable
            raise SimulationError(f"unknown external input type {kind!r}")

    def _control(self, event: JsonObject) -> None:
        operation = event["operation"]
        reasons: list[str] = []
        if operation == "ARM":
            if self.state == "DISARMED" and self.camera_status == "CONNECTED":
                self.armed = True
                self.arm_times.append(self.clock_ms)
                self._transition("MONITORING", "MANUAL_ARM", event["event_id"])
                reasons.append("MANUAL_ARM")
            elif self.camera_status != "CONNECTED":
                self._reason("CAMERA_DISCONNECTED")
                reasons.append("CAMERA_DISCONNECTED")
        elif operation == "DISARM":
            self.armed = False
            self.epoch += 1
            self.pending_commands.clear()
            self.persistence.clear()
            self.open_incidents.clear()
            self.cooldown_until_ms = None
            self._transition("DISARMED", "MANUAL_DISARM", event["event_id"])
            reasons.append("MANUAL_DISARM")
        elif operation == "ESTOP":
            self.armed = False
            self._reason("ESTOP")
            self._fault("ESTOP", "operator emergency stop", event["event_id"])
            reasons.append("ESTOP")
        self._input_record(event, accepted=True, reasons=reasons)

    def _camera_status(self, event: JsonObject) -> None:
        old = self.camera_status
        new = event["status"]
        self.camera_status = new
        reasons: list[str] = []
        self.camera_generation += 1
        if new == "DISCONNECTED":
            self._reason("CAMERA_DISCONNECTED")
            reasons.append("CAMERA_DISCONNECTED")
            self._start_camera_disconnect_timer()
            self._break_confirmation()
        elif new == "ERROR":
            self._reason("CAMERA_DISCONNECTED")
            reasons.append("CAMERA_DISCONNECTED")
            self._fault(
                "CAMERA_DISCONNECTED", "camera reported ERROR", event["event_id"]
            )
        else:
            self.camera_disconnected_at_ms = None
        self._event(
            "HEALTH_CHANGED",
            "CAMERA",
            {
                "component": "CAMERA",
                "from": "DISCONNECTED" if old == "ERROR" else old,
                "to": "DISCONNECTED" if new == "ERROR" else new,
                "reason_codes": reasons,
            },
        )
        self._input_record(event, accepted=True, reasons=reasons)

    def _start_camera_disconnect_timer(self) -> None:
        self.camera_disconnected_at_ms = self.clock_ms
        due = self.clock_ms + self.config["camera"]["disconnect_fault_after_ms"]
        self._schedule(
            due,
            "CAMERA_TIMEOUT",
            {"epoch": self.epoch, "generation": self.camera_generation},
        )

    def _actuator_status(self, event: JsonObject) -> None:
        old = self.actuator_status
        new = event["status"]
        self.actuator_status = new
        reasons: list[str] = []
        if new != "READY":
            self._reason("HARDWARE_NOT_READY")
            reasons.append("HARDWARE_NOT_READY")
            self._break_confirmation()
        if new == "FAULT":
            self._fault(
                "HARDWARE_NOT_READY", "actuator reported FAULT", event["event_id"]
            )
        self._event(
            "HEALTH_CHANGED",
            "ACTUATOR",
            {
                "component": "ACTUATOR",
                "from": old,
                "to": new,
                "reason_codes": reasons,
            },
        )
        self._input_record(event, accepted=True, reasons=reasons)

    def _restart(self, event: JsonObject) -> None:
        previous_state = self.state
        self.epoch += 1
        self.restart_times.append(self.clock_ms)
        self.armed = False
        self.pending_commands.clear()
        self.persistence.clear()
        self.open_incidents.clear()
        self.cooldown_until_ms = None
        self.last_observation = None
        self.last_observation_event_id = None
        self.last_observation_delivery_ms = None
        self._reason("PROCESS_RESTART")
        self._reason("STARTUP_DISARMED")
        self._transition("DISARMED", "PROCESS_RESTART", event["event_id"])
        if previous_state == "FAULT":
            self._event(
                "FAULT_CLEARED",
                "RUNTIME",
                {
                    "fault_id": f"fault-{self.fault_counter:06d}",
                    "reason_codes": ["PROCESS_RESTART"],
                    "latched": False,
                    "detail": "new process boundary",
                },
            )
        self.camera_generation += 1
        if self.camera_status == "DISCONNECTED":
            self._start_camera_disconnect_timer()
        self._input_record(
            event,
            accepted=True,
            reasons=["PROCESS_RESTART", "STARTUP_DISARMED"],
        )

    # -- policy observation fusion -------------------------------------

    def _observation(self, event: JsonObject) -> None:
        observation = event["observation"]
        age_ms = self.clock_ms - observation["captured_at_ms"]
        if age_ms < 0:
            raise SimulationError("validated observation became future-dated")

        if self.state == "DISARMED" or not self.armed:
            self._input_record(event, accepted=True, reasons=["STARTUP_DISARMED"])
            self._record_suppression(
                event, ["STARTUP_DISARMED"], observation, self.state, self.state
            )
            return
        if self.state == "FAULT":
            self._input_record(event, accepted=True, reasons=[])
            reason = (
                self.reason_codes[-1] if self.reason_codes else "CONFIGURATION_ERROR"
            )
            self._record_suppression(
                event, [reason], observation, self.state, self.state
            )
            return
        if self.camera_status != "CONNECTED":
            self._block(event, "CAMERA_DISCONNECTED")
            return
        if self.actuator_status != "READY":
            self._block(event, "HARDWARE_NOT_READY")
            return
        if (
            not self.config["actuator"]["enabled"]
            or not self.config["actuator"]["burst"]["enabled"]
        ):
            self._block(event, "ACTUATION_DISABLED")
            return
        if age_ms > self.config["camera"]["max_frame_age_ms"]:
            state_before = self.state
            self._reason("FRAME_STALE")
            self._break_confirmation()
            self._input_record(event, accepted=False, reasons=["FRAME_STALE"])
            self._record_suppression(
                event, ["FRAME_STALE"], observation, state_before, self.state
            )
            return

        self.last_observation = observation
        self.last_observation_event_id = event["event_id"]
        self.last_observation_delivery_ms = self.clock_ms
        self._settle_cooldown()
        tracks = observation["tracks"]
        detection = self.config["perception"]["detection"]
        people = [
            track
            for track in tracks
            if track["class"] == "PERSON"
            and track["detection_confidence"] >= detection["person_min_confidence"]
        ]
        cats = [
            track
            for track in tracks
            if track["class"] == "CAT"
            and track["detection_confidence"] >= detection["cat_min_confidence"]
        ]
        if people:
            self._block(event, "PERSON_PRESENT")
            return
        if len(cats) > 1:
            self._block(event, "MULTIPLE_CATS")
            return
        if any(track["ambiguous"] for track in people + cats):
            self._block(event, "AMBIGUOUS_TRACK")
            return
        if not cats:
            self._non_harmful(event, "NO_CAT_IN_ZONE", explicit_clear=True)
            return

        cat = cats[0]
        tracking = self.config["perception"]["tracking"]
        if cat["track_age_ms"] < tracking["min_track_age_ms"]:
            self._block(event, "TRACK_TOO_YOUNG")
            return
        if cat["track_quality"] < tracking["min_track_quality"]:
            self._block(event, "POOR_TRACK")
            return

        zone_id = cat["zone_id"]
        approach_ids = {
            zone["id"]
            for zone in self.config["scene"]["zones"]
            if zone["type"] == "approach"
        }
        if (
            zone_id not in approach_ids
            or cat["region_evidence"]["approach_overlap"] <= 0
        ):
            self._non_harmful(event, "NO_CAT_IN_ZONE", explicit_clear=True)
            return

        behavior = cat["behavior"]
        scores = behavior["scores"]
        limits = self.config["perception"]["behavior"]
        if (
            behavior["label"] == "UNKNOWN"
            or scores["UNKNOWN"] > limits["unknown_max_probability"]
        ):
            self._non_harmful(event, "BEHAVIOR_UNKNOWN", explicit_clear=True)
            return
        if behavior["label"] == "CLEAR":
            self._non_harmful(event, "BEHAVIOR_CLEAR", explicit_clear=True)
            return

        harmful = behavior["label"]
        qualifies = False
        if harmful == "EATING":
            qualifies = (
                scores["EATING"] >= limits["eating_min_probability"]
                and cat["region_evidence"]["foliage_overlap"]
                >= limits["eating_min_foliage_overlap"]
            )
        elif harmful == "DIGGING":
            qualifies = (
                scores["DIGGING"] >= limits["digging_min_probability"]
                and cat["region_evidence"]["soil_overlap"]
                >= limits["digging_min_soil_overlap"]
                and cat["region_evidence"]["motion_score"]
                >= limits["digging_min_motion_score"]
            )
        if not qualifies:
            self._non_harmful(event, "BEHAVIOR_NOT_PERSISTENT", explicit_clear=True)
            return
        if cat["no_fire_intersection"]:
            self._block(event, "NO_FIRE_INTERSECTION")
            return
        presets = {
            preset["id"]: preset for preset in self.config["scene"]["aim_presets"]
        }
        preset = presets.get(cat["aim_preset_id"])
        if preset is None or preset["zone_id"] != zone_id:
            self._block(event, "NO_SAFE_PRESET")
            return

        self._input_record(event, accepted=True, reasons=[])
        self._qualifying_observation(event, cat)

    def _block(self, event: JsonObject, reason: str) -> None:
        state_before = self.state
        self._reason(reason)
        self._break_confirmation()
        self._input_record(event, accepted=True, reasons=[reason])
        self._record_suppression(
            event,
            [reason],
            event.get("observation"),
            state_before,
            self.state,
        )

    def _non_harmful(
        self, event: JsonObject, reason: str, *, explicit_clear: bool
    ) -> None:
        state_before = self.state
        self._reason(reason)
        if self.state == "MONITORING":
            self._transition("TRACKING", reason, event["event_id"])
        if self.state in {"TRACKING", "CONFIRMING", "AIMING", "READY"}:
            self._transition("MONITORING", reason, event["event_id"])
        self.persistence.clear()
        if explicit_clear:
            self._note_incidents_clear()
        self._settle_cooldown()
        self._input_record(event, accepted=True, reasons=[reason])
        self._record_suppression(
            event,
            [reason],
            event.get("observation"),
            state_before,
            self.state,
        )

    def _break_confirmation(self) -> None:
        self.persistence.clear()
        if self.state in {"TRACKING", "CONFIRMING", "AIMING", "READY"}:
            self._transition(
                "MONITORING",
                self.reason_codes[-1]
                if self.reason_codes
                else "BEHAVIOR_NOT_PERSISTENT",
                self.last_observation_event_id or "internal-hold",
            )

    def _qualifying_observation(self, event: JsonObject, cat: JsonObject) -> None:
        state_before = self.state
        key = (cat["track_id"], cat["zone_id"], cat["behavior"]["label"])
        policy = self.config["policy"]
        tracking = self.config["perception"]["tracking"]
        accumulator = self.persistence.setdefault(key, Persistence())
        if (
            accumulator.last_seen_ms is not None
            and self.clock_ms - accumulator.last_seen_ms > tracking["max_track_gap_ms"]
        ):
            accumulator.supporting_ms.clear()
        cutoff = self.clock_ms - policy["confirmation_window_ms"]
        accumulator.supporting_ms[:] = [
            value for value in accumulator.supporting_ms if value >= cutoff
        ]
        accumulator.supporting_ms.append(self.clock_ms)
        accumulator.last_seen_ms = self.clock_ms

        incident = self.open_incidents.get(key)
        if incident is None:
            self.incident_counter += 1
            incident = Incident(
                incident_id=f"incident-{self.incident_counter:06d}",
                track_id=cat["track_id"],
                zone_id=cat["zone_id"],
                behavior=cat["behavior"]["label"],
                started_ms=self.clock_ms,
                last_harmful_ms=self.clock_ms,
                latest_observation=self.last_observation or {},
            )
            self.open_incidents[key] = incident
            self.incidents.append(incident)
        else:
            incident.last_harmful_ms = self.clock_ms
            incident.latest_observation = (
                self.last_observation or incident.latest_observation
            )
            incident.clear_since_ms = None

        if incident.action_latched:
            self._reason("INCIDENT_ALREADY_ACTIONED")
            reasons = ["INCIDENT_ALREADY_ACTIONED"]
            if self.cooldown_until_ms is not None:
                self._reason("COOLDOWN_ACTIVE")
                reasons.append("COOLDOWN_ACTIVE")
            self._record_suppression(
                event,
                reasons,
                self.last_observation,
                state_before,
                self.state,
                incident,
            )
            return
        if self.cooldown_until_ms is not None:
            self._reason("COOLDOWN_ACTIVE")
            self._record_suppression(
                event,
                ["COOLDOWN_ACTIVE"],
                self.last_observation,
                state_before,
                self.state,
                incident,
            )
            return
        if self.state == "MONITORING":
            self._transition(
                "TRACKING", "BEHAVIOR_NOT_PERSISTENT", event["event_id"], incident
            )
            self._transition(
                "CONFIRMING", "BEHAVIOR_NOT_PERSISTENT", event["event_id"], incident
            )
        elif self.state == "TRACKING":
            self._transition(
                "CONFIRMING", "BEHAVIOR_NOT_PERSISTENT", event["event_id"], incident
            )

        enough_count = (
            len(accumulator.supporting_ms) >= policy["min_supporting_observations"]
        )
        enough_time = (
            accumulator.supporting_ms[-1] - accumulator.supporting_ms[0]
            >= policy["harmful_persistence_ms"]
        )
        if not (enough_count and enough_time):
            self._reason("BEHAVIOR_NOT_PERSISTENT")
            self._record_suppression(
                event,
                ["BEHAVIOR_NOT_PERSISTENT"],
                self.last_observation,
                state_before,
                self.state,
                incident,
            )
            return
        if self.state in {"AIMING", "READY", "BURST"}:
            return

        confirmed_reason = (
            "EATING_CONFIRMED" if incident.behavior == "EATING" else "DIGGING_CONFIRMED"
        )
        self._reason(confirmed_reason)
        state_before = self.state
        self._transition("AIMING", confirmed_reason, event["event_id"], incident)
        command = {
            "command_id": self._allocate_command_id(),
            "command": "GOTO_PRESET",
            "target": cat["aim_preset_id"],
        }
        self._dispatch(
            command,
            trigger_event_id=event["event_id"],
            incident=incident,
            observation=self.last_observation,
            state_before=state_before,
            state_after="AIMING",
        )

    def _note_incidents_clear(self) -> None:
        for key, incident in list(self.open_incidents.items()):
            if incident.clear_since_ms is None:
                incident.clear_since_ms = self.clock_ms
            if (
                self.clock_ms - incident.clear_since_ms
                >= self.config["policy"]["incident_clear_ms"]
            ):
                incident.ended_ms = self.clock_ms
                del self.open_incidents[key]

    def _settle_cooldown(self) -> None:
        if self.state != "COOLDOWN" or self.cooldown_until_ms is None:
            return
        if self.clock_ms < self.cooldown_until_ms:
            self._reason("COOLDOWN_ACTIVE")
            return
        if any(incident.action_latched for incident in self.open_incidents.values()):
            self._reason("INCIDENT_ALREADY_ACTIONED")
            return
        self.cooldown_until_ms = None
        self._transition("MONITORING", "BEHAVIOR_CLEAR", "cooldown-settled")

    # -- mock transport -------------------------------------------------

    def _allocate_command_id(self) -> int:
        if self.next_command_id > MAX_SAFE_INTEGER:
            self._fault("CONFIGURATION_ERROR", "command ID exhausted")
            raise SimulationError("command ID exhausted")
        command_id = self.next_command_id
        self.next_command_id += 1
        return command_id

    def _dispatch(
        self,
        command: JsonObject,
        *,
        trigger_event_id: str,
        incident: Incident | None,
        observation: JsonObject | None,
        state_before: str,
        state_after: str,
    ) -> None:
        command = dict(command)
        command_id = command["command_id"]
        attempt: JsonObject = {
            "command": command,
            "incident_id": incident.incident_id if incident else None,
            "issued_at_ms": self.clock_ms,
            "state_after": state_after,
            "state_before": state_before,
            "trigger_event_id": trigger_event_id,
        }
        self.command_attempts.append(attempt)
        evidence_snapshot = self._audit_evidence(observation, self.clock_ms)
        safety_snapshot = self._safety_snapshot(observation, self.clock_ms, incident)
        if command_id in self.command_ledger:
            self.counts["duplicate_commands_suppressed"] += 1
            self._reason("DUPLICATE_COMMAND_ID")
            self._record_action_result(
                command,
                "DUPLICATE",
                self.clock_ms,
                trigger_event_id,
                incident,
                observation,
                state_before,
                state_after,
                self.clock_ms,
                evidence_snapshot,
                safety_snapshot,
            )
            if incident is not None:
                self._fault(
                    "DUPLICATE_COMMAND_ID",
                    "policy command ID collided with the mock deduplication ledger",
                    trigger_event_id,
                )
            return

        self.command_ledger[command_id] = command
        self.command_occurrences[command["command"]] += 1
        if incident is not None:
            self._record_pending_action(
                command,
                trigger_event_id,
                incident,
                state_before,
                state_after,
                evidence_snapshot,
                safety_snapshot,
            )
        if command["command"] == "BURST":
            self.counts["burst_commands_issued"] += 1
            if incident is not None:
                incident.action_latched = True
                self.cooldown_until_ms = (
                    self.clock_ms + self.config["policy"]["cooldown_ms"]
                )

        adapter_rejection = self._adapter_rejection(command)
        if adapter_rejection is None:
            response, delay_ms = self._scripted_response(command["command"])
        else:
            response = "DENIED"
            delay_ms = self.scenario["actuator_script"]["ack_delay_ms"]
        pending = PendingCommand(
            command=command,
            issued_at_ms=self.clock_ms,
            state_before=state_before,
            state_after=state_after,
            trigger_event_id=trigger_event_id,
            incident_id=incident.incident_id if incident else None,
            observation=observation,
            evidence_snapshot=evidence_snapshot,
            safety_snapshot=safety_snapshot,
            response=response,
            epoch=self.epoch,
        )
        self.pending_commands[command_id] = pending
        if response != "DROP":
            self._schedule(
                self.clock_ms + delay_ms,
                "COMMAND_RESPONSE",
                {"command_id": command_id, "epoch": self.epoch, "response": response},
            )
        self._schedule(
            self.clock_ms + self.config["actuator"]["ack_timeout_ms"],
            "COMMAND_TIMEOUT",
            {"command_id": command_id, "epoch": self.epoch},
        )

    def _scripted_response(self, command_name: str) -> tuple[str, int]:
        script = self.scenario["actuator_script"]
        occurrence = self.command_occurrences[command_name]
        for override in script["overrides"]:
            if (
                override["command"] == command_name
                and override["occurrence"] == occurrence
            ):
                return override["response"], override.get(
                    "delay_ms", script["ack_delay_ms"]
                )
        return script["default_response"], script["ack_delay_ms"]

    def _adapter_rejection(self, command: JsonObject) -> str | None:
        if not self.config["actuator"]["enabled"] or self.actuator_status != "READY":
            return "ACTUATOR_UNAVAILABLE"
        if command["command"] == "GOTO_PRESET":
            presets = {item["id"] for item in self.config["scene"]["aim_presets"]}
            if command.get("target") not in presets:
                return "UNKNOWN_PRESET"
        if command["command"] == "BURST":
            burst = self.config["actuator"]["burst"]
            if not burst["enabled"]:
                return "BURST_DISABLED"
            if command.get("duration_ms", 0) > burst["hardware_max_duration_ms"]:
                return "DURATION_EXCEEDS_CLAMP"
        return None

    def _finish_command(self, pending: PendingCommand, result: str) -> None:
        pending.finished = True
        command = pending.command
        incident = next(
            (
                item
                for item in self.incidents
                if item.incident_id == pending.incident_id
            ),
            None,
        )
        normalized = {
            "ACK": "ACK",
            "DENIED": "DENIED",
            "TRANSPORT_ERROR": "TRANSPORT_ERROR",
            "TIMEOUT": "TIMEOUT",
        }[result]
        self._record_action_result(
            command,
            normalized,
            self.clock_ms,
            pending.trigger_event_id,
            incident,
            pending.observation,
            pending.state_before,
            pending.state_after,
            pending.issued_at_ms,
            pending.evidence_snapshot,
            pending.safety_snapshot,
        )

        if normalized == "ACK":
            self._reason("COMMAND_ACKNOWLEDGED")
            if command["command"] == "GOTO_PRESET":
                self._goto_acknowledged(pending, incident)
            elif command["command"] == "BURST":
                self.counts["burst_commands_acked"] += 1
                if self.state == "BURST":
                    self._transition(
                        "COOLDOWN",
                        "COMMAND_ACKNOWLEDGED",
                        pending.trigger_event_id,
                        incident,
                    )
            return

        if normalized == "TIMEOUT":
            reason = "COMMAND_ACK_TIMEOUT"
        elif normalized == "DENIED":
            reason = "COMMAND_DENIED"
        else:
            reason = "TRANSPORT_ERROR"
        self._reason(reason)
        self._fault(
            reason,
            f"{command['command']} completed as {normalized}",
            pending.trigger_event_id,
        )

    def _goto_acknowledged(
        self, pending: PendingCommand, incident: Incident | None
    ) -> None:
        if incident is None or self.state != "AIMING":
            return
        reason = self._recheck_reason(incident)
        if reason is not None:
            self._reason(reason)
            self.persistence.clear()
            self._transition("MONITORING", reason, pending.trigger_event_id, incident)
            return

        confirmed_reason = (
            "EATING_CONFIRMED" if incident.behavior == "EATING" else "DIGGING_CONFIRMED"
        )
        self._transition("READY", confirmed_reason, pending.trigger_event_id, incident)
        self.counts["ready_transitions"] += 1
        self.counts["would_burst_decisions"] += 1
        incident.ready_ms = self.clock_ms
        self._transition("BURST", confirmed_reason, pending.trigger_event_id, incident)
        burst = {
            "command_id": self._allocate_command_id(),
            "command": "BURST",
            "duration_ms": self.config["actuator"]["burst"]["duration_ms"],
        }
        self._dispatch(
            burst,
            trigger_event_id=pending.trigger_event_id,
            incident=incident,
            observation=self.last_observation,
            state_before="READY",
            state_after="BURST",
        )

    def _recheck_reason(self, incident: Incident) -> str | None:
        if not self.armed:
            return "STARTUP_DISARMED"
        if self.camera_status != "CONNECTED":
            return "CAMERA_DISCONNECTED"
        if self.actuator_status != "READY":
            return "HARDWARE_NOT_READY"
        if (
            not self.config["actuator"]["enabled"]
            or not self.config["actuator"]["burst"]["enabled"]
        ):
            return "ACTUATION_DISABLED"
        observation = self.last_observation
        delivery = self.last_observation_delivery_ms
        if observation is None or delivery is None:
            return "FRAME_STALE"
        if (
            self.clock_ms - observation["captured_at_ms"]
            > self.config["camera"]["max_frame_age_ms"]
        ):
            return "FRAME_STALE"
        detection = self.config["perception"]["detection"]
        people = [
            track
            for track in observation["tracks"]
            if track["class"] == "PERSON"
            and track["detection_confidence"] >= detection["person_min_confidence"]
        ]
        cats = [
            track
            for track in observation["tracks"]
            if track["class"] == "CAT"
            and track["detection_confidence"] >= detection["cat_min_confidence"]
        ]
        if people:
            return "PERSON_PRESENT"
        if len(cats) != 1:
            return "MULTIPLE_CATS" if len(cats) > 1 else "NO_CAT_IN_ZONE"
        cat = cats[0]
        if cat["ambiguous"]:
            return "AMBIGUOUS_TRACK"
        if cat["track_id"] != incident.track_id or cat["zone_id"] != incident.zone_id:
            return "NO_CAT_IN_ZONE"
        tracking = self.config["perception"]["tracking"]
        if cat["track_age_ms"] < tracking["min_track_age_ms"]:
            return "TRACK_TOO_YOUNG"
        if cat["track_quality"] < tracking["min_track_quality"]:
            return "POOR_TRACK"
        if cat["behavior"]["label"] != incident.behavior:
            return "BEHAVIOR_NOT_PERSISTENT"
        if cat["no_fire_intersection"]:
            return "NO_FIRE_INTERSECTION"
        preset = next(
            (
                item
                for item in self.config["scene"]["aim_presets"]
                if item["id"] == cat["aim_preset_id"]
            ),
            None,
        )
        if preset is None or preset["zone_id"] != incident.zone_id:
            return "NO_SAFE_PRESET"
        return None

    # -- records, state, and assertions --------------------------------

    def _reason(self, reason: str) -> None:
        if reason not in self.reason_codes:
            self.reason_codes.append(reason)

    def _transition(
        self,
        target: str,
        reason: str,
        trigger_event_id: str,
        incident: Incident | None = None,
    ) -> None:
        if self.state == target:
            return
        source = self.state
        self.state = target
        self.visited_states.append(target)
        self._reason(reason)
        if target == "FAULT":
            self.counts["fault_transitions"] += 1
        payload: JsonObject = {
            "from": source,
            "to": target,
            "trigger_event_id": trigger_event_id,
            "reason_codes": [reason],
        }
        if incident is not None:
            payload["incident_id"] = incident.incident_id
        self._event("STATE_TRANSITION", "POLICY", payload)

    def _fault(
        self, reason: str, detail: str, trigger_event_id: str = "internal-deadline"
    ) -> None:
        self.armed = False
        if self.state != "FAULT":
            self._transition("FAULT", reason, trigger_event_id)
            self.fault_counter += 1
            self._event(
                "FAULT_RAISED",
                "POLICY",
                {
                    "fault_id": f"fault-{self.fault_counter:06d}",
                    "reason_codes": [reason],
                    "latched": True,
                    "detail": detail,
                },
            )
        self.persistence.clear()

    def _input_record(
        self, event: JsonObject, *, accepted: bool, reasons: list[str]
    ) -> None:
        payload: JsonObject = {
            "event_id": event["event_id"],
            "input_type": event["type"],
            "reason_codes": reasons,
        }
        if event["type"] == "OBSERVATION":
            payload["observation_id"] = event["observation"]["observation_id"]
        source = "PERCEPTION" if event["type"] == "OBSERVATION" else "REPLAY"
        self._event("INPUT_ACCEPTED" if accepted else "INPUT_REJECTED", source, payload)

    def _event(self, record_type: str, source: str, payload: JsonObject) -> None:
        record: JsonObject = {
            "schema_version": 1,
            "record_type": record_type,
            "sequence": self.record_sequence,
            "run_id": self.run_id,
            "monotonic_ms": self.clock_ms,
            "config_id": self.config["config_id"],
            "config_sha256": self.contracts.config_sha256,
            "scenario_id": self.scenario["scenario_id"],
            "source": source,
            "payload": payload,
        }
        self.record_sequence += 1
        self.event_records.append(record)

    def _record_action_result(
        self,
        command: JsonObject,
        result: str,
        completed_at_ms: int,
        trigger_event_id: str,
        incident: Incident | None,
        observation: JsonObject | None,
        state_before: str,
        state_after: str,
        issued_at_ms: int,
        evidence_snapshot: JsonObject,
        safety_snapshot: JsonObject,
    ) -> None:
        action: JsonObject = {
            "command_id": command["command_id"],
            "command": command["command"],
            "result": result,
        }
        if "target" in command:
            action["target"] = command["target"]
        self.action_sequence.append(action)

        self.audit_counter += 1
        audit_id = f"audit-{self.audit_counter:06d}"
        reason = {
            "ACK": "COMMAND_ACKNOWLEDGED",
            "DENIED": "COMMAND_DENIED",
            "TIMEOUT": "COMMAND_ACK_TIMEOUT",
            "DUPLICATE": "DUPLICATE_COMMAND_ID",
            "TRANSPORT_ERROR": "TRANSPORT_ERROR",
        }[result]
        audit: JsonObject = {
            "schema_version": 1,
            "audit_id": audit_id,
            "sequence": self.record_sequence,
            "run_id": self.run_id,
            "monotonic_ms": issued_at_ms,
            "config_id": self.config["config_id"],
            "config_sha256": self.contracts.config_sha256,
            "scenario_id": self.scenario["scenario_id"],
            "trigger_event_id": trigger_event_id,
            "state_before": state_before,
            "state_after": state_after,
            "decision": "DISPATCH",
            "reason_codes": [reason],
            "evidence": evidence_snapshot,
            "safety": safety_snapshot,
            "action": {
                "command": command,
                "dispatch_mode": "MOCK",
                "physical_effect_possible": False,
                "retry_count": 0,
                "automatic_retry_allowed": False,
            },
            "outcome": result,
        }
        if incident is not None:
            audit["incident_id"] = incident.incident_id
        if result == "ACK":
            audit["ack_monotonic_ms"] = completed_at_ms
        self.record_sequence += 1
        self.audit_records.append(audit)
        self._event("ACTION_AUDIT_REF", "ACTUATOR", {"audit_id": audit_id})

        if result != "DUPLICATE":
            ack_status = {
                "ACK": "ACK",
                "DENIED": "DENIED",
                "TIMEOUT": "TIMEOUT",
                "TRANSPORT_ERROR": "MISSING",
            }[result]
            self.evaluator_actions.append(
                {
                    "record_type": "action",
                    "schema_version": 1,
                    "action_id": audit_id,
                    "session_id": self.scenario["scenario_id"],
                    "timestamp_ms": next(
                        item["issued_at_ms"]
                        for item in reversed(self.command_attempts)
                        if item["command"] is command
                        or item["command"]["command_id"] == command["command_id"]
                    ),
                    "command_id": str(command["command_id"]),
                    "action": command["command"],
                    "state_before": state_before,
                    "state_after": state_after,
                    "incident_id": incident.incident_id if incident else None,
                    "ack_status": ack_status,
                    "is_retry": False,
                    "metadata": {
                        "mock_only": True,
                        "physical_effect_possible": False,
                        "transport_result": result,
                    },
                }
            )

    def _record_pending_action(
        self,
        command: JsonObject,
        trigger_event_id: str,
        incident: Incident,
        state_before: str,
        state_after: str,
        evidence_snapshot: JsonObject,
        safety_snapshot: JsonObject,
    ) -> None:
        """Flush authorization before the mock can acknowledge or the process can restart."""

        self.audit_counter += 1
        audit_id = f"audit-{self.audit_counter:06d}"
        reason = (
            "EATING_CONFIRMED" if incident.behavior == "EATING" else "DIGGING_CONFIRMED"
        )
        self.audit_records.append(
            {
                "schema_version": 1,
                "audit_id": audit_id,
                "sequence": self.record_sequence,
                "run_id": self.run_id,
                "monotonic_ms": self.clock_ms,
                "config_id": self.config["config_id"],
                "config_sha256": self.contracts.config_sha256,
                "scenario_id": self.scenario["scenario_id"],
                "incident_id": incident.incident_id,
                "trigger_event_id": trigger_event_id,
                "state_before": state_before,
                "state_after": state_after,
                "decision": "DISPATCH",
                "reason_codes": [reason],
                "evidence": evidence_snapshot,
                "safety": safety_snapshot,
                "action": {
                    "command": command,
                    "dispatch_mode": "MOCK",
                    "physical_effect_possible": False,
                    "retry_count": 0,
                    "automatic_retry_allowed": False,
                },
                "outcome": "PENDING",
            }
        )
        self.record_sequence += 1
        self._event("ACTION_AUDIT_REF", "ACTUATOR", {"audit_id": audit_id})

    def _record_suppression(
        self,
        event: JsonObject,
        reasons: list[str],
        observation: JsonObject | None,
        state_before: str,
        state_after: str,
        incident: Incident | None = None,
    ) -> None:
        self.audit_counter += 1
        audit_id = f"audit-{self.audit_counter:06d}"
        safety = self._safety_snapshot(observation, self.clock_ms, incident)
        safety["all_clear"] = False
        audit: JsonObject = {
            "schema_version": 1,
            "audit_id": audit_id,
            "sequence": self.record_sequence,
            "run_id": self.run_id,
            "monotonic_ms": self.clock_ms,
            "config_id": self.config["config_id"],
            "config_sha256": self.contracts.config_sha256,
            "scenario_id": self.scenario["scenario_id"],
            "trigger_event_id": event["event_id"],
            "state_before": state_before,
            "state_after": state_after,
            "decision": "SUPPRESS",
            "reason_codes": list(dict.fromkeys(reasons)),
            "evidence": self._audit_evidence(observation, self.clock_ms),
            "safety": safety,
            "outcome": "SUPPRESSED",
        }
        if incident is not None:
            audit["incident_id"] = incident.incident_id
        self.audit_records.append(audit)
        self.record_sequence += 1
        self._event("ACTION_AUDIT_REF", "POLICY", {"audit_id": audit_id})

    def _audit_evidence(
        self, observation: JsonObject | None, completed_at_ms: int
    ) -> JsonObject:
        if observation is None:
            return {
                "observation_id": "no-observation",
                "captured_at_ms": completed_at_ms,
                "age_ms": 0,
                "track_ids": [],
                "model_ids": [],
            }
        tracks = observation["tracks"]
        cats = [track for track in tracks if track["class"] == "CAT"]
        models = sorted(
            {
                track["behavior"]["model_id"]
                for track in cats
                if track["behavior"].get("model_id")
            }
        )
        result: JsonObject = {
            "observation_id": observation["observation_id"],
            "captured_at_ms": observation["captured_at_ms"],
            "age_ms": completed_at_ms - observation["captured_at_ms"],
            "track_ids": [track["track_id"] for track in tracks],
            "model_ids": models,
        }
        if cats:
            cat = cats[0]
            if cat["zone_id"] is not None:
                result["zone_id"] = cat["zone_id"]
            if cat["aim_preset_id"] is not None:
                result["aim_preset_id"] = cat["aim_preset_id"]
            result["behavior"] = cat["behavior"]
            result["region_evidence"] = cat["region_evidence"]
        return result

    def _safety_snapshot(
        self,
        observation: JsonObject | None,
        at_ms: int,
        incident: Incident | None,
    ) -> JsonObject:
        person_clear = True
        single_cat_clear = False
        frame_fresh = False
        track_quality_clear = False
        behavior_clear = False
        no_fire_clear = False
        safe_preset = False
        if observation is not None:
            detection = self.config["perception"]["detection"]
            people = [
                track
                for track in observation["tracks"]
                if track["class"] == "PERSON"
                and track["detection_confidence"] >= detection["person_min_confidence"]
            ]
            cats = [
                track
                for track in observation["tracks"]
                if track["class"] == "CAT"
                and track["detection_confidence"] >= detection["cat_min_confidence"]
            ]
            person_clear = not people
            single_cat_clear = len(cats) == 1
            frame_fresh = (
                at_ms - observation["captured_at_ms"]
                <= self.config["camera"]["max_frame_age_ms"]
            )
            if cats:
                cat = cats[0]
                track_quality_clear = (
                    cat["track_quality"]
                    >= self.config["perception"]["tracking"]["min_track_quality"]
                )
                behavior_clear = cat["behavior"]["label"] in {"EATING", "DIGGING"}
                no_fire_clear = not cat["no_fire_intersection"]
                safe_preset = cat["aim_preset_id"] is not None
        values = {
            "armed": self.armed,
            "person_clear": person_clear,
            "single_cat_clear": single_cat_clear,
            "frame_fresh": frame_fresh,
            "track_quality_clear": track_quality_clear,
            "behavior_clear": behavior_clear,
            "no_fire_clear": no_fire_clear,
            "safe_preset_available": safe_preset,
            "hardware_ready": self.actuator_status == "READY",
            "cooldown_clear": self.cooldown_until_ms is None,
            "incident_action_available": incident is None
            or not incident.action_latched,
        }
        values["all_clear"] = all(values.values())
        return values

    def _assert_expectations(self) -> list[Assertion]:
        expected = self.scenario["expectations"]
        assertions: list[Assertion] = []

        def check(name: str, wanted: Any, actual: Any) -> None:
            assertions.append(Assertion(name, wanted == actual, wanted, actual))

        check("final_state", expected["final_state"], self.state)
        for name, wanted in expected["exact_counts"].items():
            check(f"exact_counts.{name}", wanted, self.counts[name])
        visited = set(self.visited_states)
        for state in expected["required_states"]:
            assertions.append(
                Assertion(
                    f"required_state.{state}", state in visited, True, state in visited
                )
            )
        for state in expected["forbidden_states"]:
            assertions.append(
                Assertion(
                    f"forbidden_state.{state}",
                    state not in visited,
                    False,
                    state in visited,
                )
            )
        reasons = set(self.reason_codes)
        for reason in expected["required_reason_codes"]:
            assertions.append(
                Assertion(
                    f"required_reason.{reason}",
                    reason in reasons,
                    True,
                    reason in reasons,
                )
            )
        commands = {item["command"] for item in self.action_sequence}
        for command in expected["forbidden_commands"]:
            assertions.append(
                Assertion(
                    f"forbidden_command.{command}",
                    command not in commands,
                    False,
                    command in commands,
                )
            )
        wanted_actions = expected["expected_action_sequence"]
        assertions.append(
            Assertion(
                "expected_action_sequence.length",
                len(wanted_actions) == len(self.action_sequence),
                len(wanted_actions),
                len(self.action_sequence),
            )
        )
        for index, wanted in enumerate(wanted_actions):
            actual = (
                self.action_sequence[index] if index < len(self.action_sequence) else {}
            )
            matches = all(actual.get(key) == value for key, value in wanted.items())
            assertions.append(
                Assertion(f"expected_action_sequence.{index}", matches, wanted, actual)
            )
        for invariant in expected["invariants"]:
            assertions.append(self._assert_invariant(invariant))
        return assertions

    def _assert_invariant(self, invariant: str) -> Assertion:
        passed = True
        actual: Any = True
        if invariant == "STARTS_DISARMED":
            passed = self.visited_states[0] == "DISARMED"
            actual = self.visited_states[0]
        elif invariant == "NO_PHYSICAL_EFFECTS":
            passed = (
                self.counts["physical_bursts"] == 0
                and not self.config["actuator"]["allow_physical_effects"]
                and self.config["actuator"]["backend"] == "MOCK"
            )
        elif invariant == "RESTART_REQUIRES_REARM":
            passed = all(
                any(restart < arm for arm in self.arm_times)
                for restart in self.restart_times
                if any(
                    attempt["issued_at_ms"] > restart
                    for attempt in self.command_attempts
                )
            )
        elif invariant == "PERSON_BLOCKS_ACTION":
            passed = "PERSON_PRESENT" in self.reason_codes and not self.action_sequence
        elif invariant == "MULTIPLE_CATS_BLOCK_ACTION":
            passed = "MULTIPLE_CATS" in self.reason_codes and not self.action_sequence
        elif invariant == "STALE_FRAMES_BLOCK_ACTION":
            passed = "FRAME_STALE" in self.reason_codes and not self.action_sequence
        elif invariant == "POOR_TRACK_BLOCKS_ACTION":
            passed = "POOR_TRACK" in self.reason_codes and not self.action_sequence
        elif invariant == "NO_FIRE_BLOCKS_ACTION":
            passed = (
                "NO_FIRE_INTERSECTION" in self.reason_codes and not self.action_sequence
            )
        elif invariant == "HARDWARE_NOT_READY_BLOCKS_ACTION":
            passed = (
                "HARDWARE_NOT_READY" in self.reason_codes and not self.action_sequence
            )
        elif invariant == "ONE_BURST_PER_INCIDENT":
            burst_incidents = [
                attempt["incident_id"]
                for attempt in self.command_attempts
                if attempt["command"]["command"] == "BURST"
                and attempt["command"]["command_id"] in self.command_ledger
                and attempt["incident_id"] is not None
            ]
            passed = all(count <= 1 for count in Counter(burst_incidents).values())
        elif invariant == "BURST_NOT_RETRIED":
            passed = self.counts["automatic_retries"] == 0
        elif invariant == "COMMAND_IDS_DEDUPLICATED":
            unique_ids = [item["command_id"] for item in self.command_ledger.values()]
            passed = len(unique_ids) == len(set(unique_ids))
        elif invariant == "ACTIONS_AUDITED":
            passed = len(self.audit_records) >= len(self.command_attempts)
        elif invariant == "DETERMINISTIC_REPLAY":
            # run_scenario performs an actual second execution and replaces this marker.
            passed = False
            actual = "not verified by a single Simulator.run()"
        else:  # schema validation makes this unreachable
            passed = False
            actual = "unknown invariant"
        return Assertion(f"invariant.{invariant}", passed, True, actual)

    def _build_evaluator_records(self) -> list[JsonObject]:
        final_time = max(1, self.clock_ms)
        records: list[JsonObject] = [
            {
                "record_type": "session",
                "schema_version": 1,
                "session_id": self.scenario["scenario_id"],
                "monitored_duration_ms": final_time,
                "startup_state": "DISARMED",
                "final_state": self.state,
                "failure_reason": (
                    self.reason_codes[-1] if self.state == "FAULT" else None
                ),
                "metadata": {
                    "config_hash": self.contracts.config_sha256,
                    "reference_simulator": True,
                },
            }
        ]
        for incident in self.incidents:
            observation = incident.latest_observation
            cats = [
                track
                for track in observation.get("tracks", [])
                if track["class"] == "CAT"
            ]
            cat = next(
                (track for track in cats if track["track_id"] == incident.track_id),
                cats[0] if cats else None,
            )
            score = (
                cat["behavior"]["scores"][incident.behavior] if cat is not None else 0.0
            )
            end_ms = max(
                incident.started_ms + 1,
                incident.last_harmful_ms + 1,
                (incident.ready_ms or 0) + 1,
            )
            records.append(
                {
                    "record_type": "prediction_event",
                    "schema_version": 1,
                    "event_id": incident.incident_id,
                    "session_id": self.scenario["scenario_id"],
                    "behavior": incident.behavior,
                    "start_ms": incident.started_ms,
                    "end_ms": end_ms,
                    "score": score,
                    "would_action": incident.ready_ms is not None,
                    "ready_ms": incident.ready_ms,
                    "zone_id": incident.zone_id,
                    "incident_id": incident.incident_id,
                    "track_id": incident.track_id,
                    "person_present": False,
                    "cat_count": 1,
                    "stale_input": False,
                    "track_lost": False,
                    "hardware_ready": True,
                    "no_fire_intersection": False,
                    "model_id": (
                        cat["behavior"].get("model_id") if cat is not None else None
                    ),
                    "config_hash": self.contracts.config_sha256,
                    "metadata": {"reference_simulator": True},
                }
            )
        records.extend(self.evaluator_actions)
        return records


def run_scenario(
    scenario_path: str | Path,
    *,
    config_path: str | Path | None = None,
    schema_dir: str | Path | None = None,
    verify_determinism: bool = True,
) -> RunResult:
    """Load, validate, execute, assert, and optionally replay one scenario twice."""

    contracts = load_contracts(
        scenario_path, config_path=config_path, schema_dir=schema_dir
    )
    result = Simulator(contracts).run()
    if not verify_determinism:
        return result
    replay = Simulator(contracts).run()
    same = result.deterministic_signature() == replay.deterministic_signature()
    result.deterministic_replay_verified = same
    marker = "invariant.DETERMINISTIC_REPLAY"
    assertions = [
        assertion for assertion in result.assertions if assertion.name != marker
    ]
    assertions.append(
        Assertion(
            marker,
            same,
            result.deterministic_signature(),
            replay.deterministic_signature(),
        )
    )
    result.assertions = tuple(assertions)
    return result


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def write_jsonl(path: str | Path, records: Iterable[JsonObject]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(stable_json(record))
            stream.write("\n")
            stream.flush()
