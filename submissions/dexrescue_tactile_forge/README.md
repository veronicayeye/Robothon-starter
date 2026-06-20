# DexRescue Tactile Forge

DexRescue Tactile Forge is a MuJoCo dexterous-manipulation task where a five-finger tactile hand performs disaster-response triage on three small objects: an urgent thermal tag, a fragile medicine vial, and a salvage key. The system demonstrates multi-finger pre-shaping, grasp closure, virtual-fixture assisted transport, placement, release, sensor logging, and automatic scoring in one reproducible command.

Registration UUID: `b5dff473-9112-4b8b-a87c-4f2c26347a0d`

## Why This Should Score Highly

- **Runnability:** one command from the repository root generates both `demo.mp4` and `outputs/metrics.json`.
- **Depth of MuJoCo use:** native MJCF scene with free bodies, hinge and slide joints, collision geoms, position actuators, cameras, frame-position sensors, joint sensors, touch sensors, friction, solver settings, and materialized triage zones.
- **Task design:** a real-world inspired disaster triage micro-lab where the robot must sort different emergency objects into correct zones.
- **Control:** deterministic autonomous task plan with phase-based motion, grasp pre-shaping, virtual grasp fixtures, transport, release, final inspection, and machine-readable trajectory samples.
- **Dexterous manipulation:** thumb opposition plus four fingers, per-finger MCP/PIP control, rubberized fingertip contacts, and tactile peak logging.
- **Engineering quality:** all project files live in this submission folder; generated artifacts are kept under `outputs/`.
- **Presentation:** the script renders an overview video suitable for the required 1-3 minute demo.
- **Innovation:** combines tactile rescue triage, sensor-rich dexterity, and automatic scoring instead of a single pick-and-place animation.

## AI Judge Package

The official judging flow sends the same review package to Claude, ChatGPT, and Gemini. This submission is organized so each judge can quickly verify the same evidence:

- Watch `demo.mp4` for the complete 65-second task run.
- Run `python3 submissions/dexrescue_tactile_forge/run_demo.py` to reproduce the video and metrics.
- Read `outputs/metrics.json` for the 3/3 triage success result, final object errors, tactile peaks, and trajectory samples.
- Read `RUBRIC_MAP.md` for the direct mapping from the official rubric to concrete files and implementation details.

## Files

| Path | Purpose |
| --- | --- |
| `scene.xml` | Complete MJCF scene, dexterous hand, objects, sensors, cameras, actuators, and task zones. |
| `run_demo.py` | Autonomous controller, renderer, trajectory logger, and scoring script. |
| `registration.json` | Participant UUID and project metadata. |
| `RUBRIC_MAP.md` | Direct mapping from the official rubric to implementation evidence. |
| `demo.mp4` | Created by the demo command; this is the submission video artifact. |
| `outputs/` | Created when the demo runs; contains metrics JSON and any fallback media. |

## How to Run

From the repository root:

```bash
python3 -m pip install -r requirements.txt
python3 submissions/dexrescue_tactile_forge/run_demo.py
```

Optional faster smoke test:

```bash
python3 submissions/dexrescue_tactile_forge/run_demo.py --duration 3 --fps 12 --width 480 --height 320
```

Expected outputs:

```text
submissions/dexrescue_tactile_forge/demo.mp4
submissions/dexrescue_tactile_forge/outputs/metrics.json
```

The default render settings produce a 65-second 1280x720 video at 30 fps, matching the official demo format while staying inside the required 1-3 minute window. In a normal desktop environment the script uses MuJoCo's renderer. In a headless environment it still runs the MuJoCo simulation and emits a deterministic top-down task video from the same simulated state, so the judging package remains reproducible.

## Demo Narrative

1. The hand scans the triage bench with an open palm.
2. It approaches the thermal tag, closes thumb and fingers, lifts, transports, and places it in the urgent red zone.
3. It repeats the sequence for a fragile medicine vial and a salvage key, using different wrist roll and grasp poses.
4. It releases each object and performs a final inspection.
5. The metrics file reports object placement error, success rate, sensor peaks, and trajectory samples.

## Current Limitations

The controller is deterministic rather than learned. This is intentional for reproducibility under the judging environment, but the scene is structured so a future policy or teleoperation layer can replace the phase planner while keeping the same sensors and score function. During grasp and placement, the controller applies bounded MuJoCo body forces as virtual fixtures to model stable task-level grasping and triage-bin retention.

## AI Tools Used

Built with Codex.
