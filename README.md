# Plant Protector

## Jetson Orin Nano Super Edge-AI Cat Deterrent

**Product brief and recommended software architecture**  
**Research checked:** 9 August 2026

> **Primary recommendation:** Build a modular, safety-gated perception-to-action system: cat/person detection and tracking, fixed plant/soil regions, a short-video behavior classifier for eating versus digging, a deterministic state machine, and a transport-agnostic actuator adapter. Treat a compact VLM or VLA as an optional research branch—not the component that directly controls the burst.

| Design dimension | Decision |
|---|---|
| Runtime platform | NVIDIA Jetson Orin Nano Super, local inference |
| Primary runtime | C++20 + GStreamer / DeepStream + TensorRT |
| Behavior understanding | Temporal classifier with region and motion evidence |
| Aiming strategy | Pre-calibrated safe plant-zone presets first; visual servoing later |
| Hardware boundary | Simple semantic commands with ACK, watchdog, and hard limits |
| Default safety posture | Startup disarmed; uncertainty and faults mean no action |
| VLA position | Research-only shadow policy after the modular baseline works |

> **Animal-safety boundary:** This system should be a secondary protective layer, not the only barrier between a cat and a toxic plant. Confirm every plant against a veterinary toxicology source and physically remove or isolate high-risk plants. Feline veterinary guidance discourages aversive training stimuli, so any actuator should be optional, low-energy, aimed at a pre-calibrated area near the plant rather than at the cat, limited to one brief interruption, and backed by positive reinforcement and environmental changes.

## 1. Executive recommendation

The best version-one architecture is not a monolithic Vision-Language-Action model. It is a **VLA-shaped system** with explicit, independently testable stages:

1. Vision detects and tracks a cat.
2. Temporal perception decides whether the cat is eating, digging, or merely passing or sniffing.
3. A deterministic policy confirms persistence and checks safety interlocks.
4. A narrow action adapter issues simple movement and burst commands.

The plants are stationary, so the software should not rediscover them every frame. Calibrate foliage, soil, protected zones, no-fire areas, and safe aim presets once. The safest and simplest version-one targeting strategy is to aim at a known area near a pot, not at the animal.

### Architecture choice

| Option | Fit | Recommendation |
|---|---|---|
| Single-frame detector plus rules | Fine for cat presence, weak for eating versus sniffing and digging versus ordinary paw motion | Observe-only prototype |
| Detector + tracker + temporal classifier | Efficient, local, explainable, replayable | **Recommended** |
| Small VLM verifier | Useful on saved clips, but slower and generative | Optional sidecar |
| SmolVLA-class compact VLA | Interesting but requires demonstrations and has a domain gap | Later shadow experiment |
| OpenVLA / GR00T-class model | Too large or mismatched for this Nano MVP | Exclude |

## 2. Product definition

### Goals

- Detect a cat entering a protected plant zone.
- Distinguish eating and digging from passing, sitting, sniffing, grooming, leaf movement, and human activity.
- Select a safe pre-calibrated aim point associated with the affected plant zone.
- Issue no more than one brief deterrent event per confirmed incident, followed by a long cooldown.
- Record enough evidence to explain every would-act and did-act decision.
- Run locally on the Jetson during normal operation.

### Non-goals

- A mobile apartment patrol robot.
- Cat identity recognition.
- General open-ended language planning.
- Direct end-to-end model control of motors or the burst actuator.
- Continuous pursuit or repeated firing.
- A guarantee against toxic-plant exposure; physical barriers remain necessary.

## 3. Recommended architecture

```text
camera
  -> cat/person detector
  -> persistent tracker
  -> protected-zone gate
  -> temporal clip buffer
  -> behavior classifier
  -> region/motion evidence fusion
  -> deterministic safety policy
  -> safe aim preset
  -> action API
  -> hardware adapter
```

Supporting components are a calibration tool, event recorder, review UI, replay evaluator, and reproducible model-build scripts. The optional VLM sidecar reads saved clips and never sits in the physical action path.

### Runtime data flow

1. Capture a hardware-decoded stream and keep a rolling pre-event buffer.
2. Detect cat and person classes and attach a stable track ID.
3. Skip behavior inference until the tracked cat intersects a protected zone.
4. Assemble a short temporal crop containing both the cat and relevant plant/pot context.
5. Classify behavior and calculate foliage overlap, soil overlap, local motion, and track quality.
6. Smooth the evidence over time and require persistence.
7. Select a safe plant-zone preset.
8. Pass every action through explicit safety interlocks.
9. Store clips, scores, transitions, command IDs, ACKs, model versions, and a configuration hash.

## 4. Perception and behavior recognition

### Scene calibration

Draw and version these regions:

- Protected approach zone
- Foliage polygons
- Soil/pot polygons
- No-fire polygons
- One or more safe aim presets per plant zone

### Detection and tracking

Start with a small COCO-pretrained detector containing `cat` and `person`. Keep it behind a swappable interface. Fine-tune only after real recordings show a local miss-rate problem. Use a DeepStream tracker such as NvSORT or NvDCF and reject young, stale, or low-quality tracks.

### Temporal classifier

Eating and digging are temporal interactions, so use roughly a one-to-three-second window rather than a single image. Recommended model order:

1. TSM + MobileNetV3
2. MoViNet-A0 or a streaming MoViNet
3. NVIDIA TAO ActionRecognitionNet
4. CNN features + a small GRU baseline
5. Head/paw localization only if precision plateaus

Use richer training labels—`PASSING`, `SNIFFING`, `EATING`, `DIGGING`, `OTHER/UNKNOWN`—then map them to operational outputs `CLEAR`, `EATING`, `DIGGING`, and `UNKNOWN`.

### Evidence fusion

```text
harm_score = behavior_probability
             x region_evidence
             x track_quality
             x safety_availability
```

A high eating probability without foliage interaction should not trigger. A high digging probability without repeated motion in the soil region should not trigger.

## 5. Decision policy and safety

States:

```text
DISARMED -> MONITORING -> TRACKING -> CONFIRMING
          -> AIMING -> READY -> BURST -> COOLDOWN

Any active state -> FAULT
```

Rules:

- Startup is always `DISARMED`.
- Only a human can `ARM` the system.
- One frame cannot authorize a burst.
- `UNKNOWN`, person present, multiple ambiguous cats, stale frames, poor tracking, hardware-not-ready, or no-fire intersection means `HOLD` or `FAULT`.
- `BURST` is allowed only from `READY`, once per event.
- A missing burst acknowledgement is never retried automatically.
- Use a long cooldown and never auto-escalate intensity.

### Targeting

Use `GOTO_PRESET` for a safe point near the affected pot. Do not aim at the face, eyes, ears, or body. Continuous visual servoing can be added later with incremental `PAN_LEFT`, `PAN_RIGHT`, `TILT_UP`, and `TILT_DOWN` commands.

## 6. Hardware action API

Recommended commands:

- `ARM`
- `DISARM`
- `HOME`
- `GOTO_PRESET`
- `PAN_LEFT`
- `PAN_RIGHT`
- `TILT_UP`
- `TILT_DOWN`
- `HOLD`
- `BURST`
- `ESTOP`
- `STATUS`

Minimal JSON Lines wire format:

```json
{"v":1,"id":101,"cmd":"GOTO_PRESET","target":"pot_1_front"}
{"v":1,"id":102,"cmd":"BURST","duration_ms":75}

{"id":101,"status":"ACK"}
{"id":102,"status":"DENIED","reason":"COOLDOWN"}
```

The duration is illustrative only. Hardware must clamp the pulse to a validated maximum. Use unique command IDs, deduplication, a heartbeat watchdog, and no automatic retry for `BURST`.

C++ boundary:

```cpp
enum class ActionType {
    Arm, Disarm, Home, GotoPreset,
    PanLeft, PanRight, TiltUp, TiltDown,
    Hold, Burst, EStop, Status
};

struct ActionCommand {
    ActionType type;
    std::string target;
    float amount = 0.0F;
    uint32_t duration_ms = 0;
    uint64_t command_id = 0;
};

class IActuator {
public:
    virtual ~IActuator() = default;
    virtual ActionResult execute(const ActionCommand&) = 0;
    virtual ActuatorStatus status() const = 0;
};
```

Implement `MockActuator`, `SerialActuator`, and `ReplayActuator`.

## 7. Software stack

| Layer | Recommendation |
|---|---|
| OS / SDK | JetPack 7.2 / Jetson Linux 39.2 |
| Video analytics | DeepStream 9.1 + GStreamer |
| Inference | TensorRT 10.16.2, FP16 first |
| Runtime | C++20 + CMake |
| Training | Python + PyTorch or TAO off-device |
| Calibration | OpenCV or a small local web UI |
| Configuration | YAML/TOML with schema validation and hash |
| Event store | SQLite metadata + local clips |
| Service | systemd first; containerize after the camera/model path is stable |

Export models to ONNX, build TensorRT engines on the target Jetson or an identical environment, pin versions, and retain memory headroom for video buffers and tracking. Benchmark in MAXN SUPER mode, then choose the deployed power profile from measured latency and thermals.

## 8. Data and evaluation

Run observe-only for one to two weeks with a mock actuator. Save clips around protected-zone entries and sample normal household activity. Do not stage chewing with a toxic plant; use veterinarian-confirmed cat-safe grass and a dedicated safe digging setup.

Dataset priorities:

- Diverse `EATING` and `DIGGING` clips
- Large hard-negative set: sniffing, looking, grooming, leaf motion, human watering, pawing toys, stepping near the pot
- Splits by day/session, not random frames
- Metadata for cat, lighting, plant zone, camera version, and staged-safe status
- An uncertainty queue for active learning

Primary metrics are event precision, event recall, false would-bursts per hour, time to `READY`, track-loss rate, `UNKNOWN` rate, and deterministic replay.

## 9. Roadmap

| Phase | Objective | Exit condition |
|---|---|---|
| 0 | Safety, camera view, no-fire zones, mock actuator | Boots disarmed; failure paths defined |
| 1 | Detector, tracker, ROI, recorder | Useful local clips; no hardware output |
| 2 | Temporal behavior model and evidence fusion | Measured performance on held-out sessions |
| 3 | State machine and aim-only integration | Correct movement commands, burst disabled |
| 4 | Shadow armed logic | No false would-bursts over an agreed window |
| 5 | Controlled supervised pilot | No duplicate or interlock-violating actions |
| 6 | Hardening | Watchdogs, fault injection, service and retention tested |
| 7 | Optional VLA experiment | Shadow-only comparison against modular baseline |

## 10. Proposed acceptance criteria

- Cat-zone recall: at least 98% on held-out local events.
- Harmful-behavior precision: at least 95% for would-act decisions.
- Harmful-behavior recall target: at least 90%, while preserving precision.
- Zero false would-bursts during 20–50 hours of shadow operation.
- All person-present tests block `READY` and `BURST`.
- Typical behavior-onset-to-`READY` below two seconds; command dispatch after `READY` below 250 ms.
- Exactly one burst command per continuous event.
- Camera loss, stale inference, process restart, MCU reset, limit fault, or heartbeat loss results in `DISARMED` or `FAULT`.
- Every action is replayable with clip, model IDs, configuration hash, and command log.

These are engineering targets, not performance claims.

## 11. Key risks

- **Eating versus sniffing:** temporal windows, foliage contact, hard negatives, optional head localization.
- **Digging versus paw motion:** repeated motion inside soil polygon plus classifier persistence.
- **Occlusion:** elevated diagonal camera, stronger tracker, expanded crops.
- **Multiple cats:** no action in the MVP.
- **Human presence:** detector, no-fire geometry, manual arm, physical kill switch.
- **Stress/injury:** low-energy near-plant interruption, one-shot limit, long cooldown, replaceable deterrent.
- **Habituation:** never auto-escalate; rely on barriers, enrichment, and positive reinforcement.
- **Toxic exposure before detection:** remove/isolate dangerous plants; consult a veterinarian or poison-control source after suspected ingestion.
- **Toolchain mismatch:** pin versions, build on target, run equivalence tests.
- **Linux stall:** microcontroller watchdog and fail-closed output design.

## 12. VLM/VLA research track

A small VLM can review saved clips, assist labeling, and generate event summaries. It should not authorize physical actions. Current Jetson catalogs include compact options such as Qwen3.5 0.8B and a 2B physical-reasoning VLM, but they still compete for memory and add latency.

SmolVLA is the most plausible true-VLA experiment because it is much smaller than OpenVLA. It still needs task-specific demonstrations and has a major domain gap. Collect human-selected action demonstrations from recorded observations and run the policy in shadow mode. OpenVLA 7B and GR00T-class models are not appropriate for the 8GB Nano MVP.

## 13. Open decisions

- Number and layout of plants
- Camera position and nighttime lighting
- Number of cats likely to be visible simultaneously
- Actuator geometry and homing method
- Exact safe deterrent mechanism
- Available off-device training compute
- UI expectations
- Event retention policy

## 14. References

1. [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
2. [NVIDIA JetPack 7.2 Downloads](https://developer.nvidia.com/embedded/jetpack/downloads)
3. [NVIDIA DeepStream 9.1 Overview](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Overview.html)
4. [DeepStream Reference Application](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_ref_app_deepstream.html)
5. [DeepStream Tracker](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html)
6. [DeepStream ROI Analytics](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdsanalytics.html)
7. [DeepStream ROI Preprocess](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvdspreprocess.html)
8. [TensorRT Support Matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
9. [NVIDIA TAO ActionRecognitionNet](https://docs.nvidia.com/tao/tao-toolkit/latest/text/cv_finetuning/pytorch/action_recognition_net.html)
10. [MoViNets](https://arxiv.org/abs/2103.11511)
11. [Temporal Shift Module](https://arxiv.org/abs/1811.08383)
12. [OpenVLA](https://arxiv.org/abs/2406.09246)
13. [SmolVLA](https://huggingface.co/blog/smolvla)
14. [LeRobot SmolVLA documentation](https://huggingface.co/docs/lerobot/smolvla)
15. [NVIDIA Isaac GR00T N1](https://developer.nvidia.com/blog/accelerate-generalist-humanoid-robot-development-with-nvidia-isaac-gr00t-n1/)
16. [Jetson AI Lab: Qwen3.5 0.8B](https://www.jetson-ai-lab.com/models/qwen3-5-0-8b/)
17. [Jetson AI Lab: Cosmos Reason 2 2B](https://www.jetson-ai-lab.com/models/cosmos-reason2-2b/)
18. [FelineVMA: How Cats Learn](https://catvets.com/resource/how-cats-learn/)
19. [FelineVMA: Positive Reinforcement Training Toolkit](https://catvets.com/resource/positive-reinforcement-training-educational-toolkit/)
20. [ASPCA Toxic and Non-Toxic Plants](https://www.aspca.org/pet-care/aspca-poison-control/toxic-and-non-toxic-plants)

> **Bottom line:** Build the observe-only dataset and replay harness first. The hardest technical question is behavior recognition in the real camera view—not motor control and not a foundation model. Once that pipeline is precise and explainable, the actuator becomes a small, replaceable endpoint and the VLA branch becomes a meaningful experiment rather than a risky dependency.
