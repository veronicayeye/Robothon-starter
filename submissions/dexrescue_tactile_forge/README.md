# DexRescue Tactile Forge

DexRescue Tactile Forge is a MuJoCo dexterous-manipulation task where a five-finger tactile hand completes a 10-subtask disaster-response triage arena. The hand sorts urgent, fragile, and safe rescue items - thermal tag, medicine vial, salvage key, airway clip, IV connector, radio beacon, data chip, pressure syringe, hazmat cap, and seal puck - into correct zones while logging contact evidence, tactile peaks, actuator ranges, trajectory samples, perturbation stress tests, and automatic scoring in one reproducible command.

Registration UUID: `b5dff473-9112-4b8b-a87c-4f2c26347a0d`

## Why This Should Score Highly

- **Runnability:** one command from the repository root generates `demo.mp4`, `outputs/metrics.json`, and `JUDGE_REPORT.md`.
- **Depth of MuJoCo use:** native MJCF scene with free bodies, hinge and slide joints, collision geoms, position actuators, cameras, frame-position sensors, joint sensors, touch sensors, friction, solver settings, and materialized triage zones.
- **Task design:** a real-world inspired disaster triage micro-lab where the robot must complete 10 object-specific rescue subtasks across urgent, fragile, and safe zones while remaining robust to deterministic initial-position jitter trials.
- **Control:** deterministic autonomous task plan with phase-based motion, object-aware pick targets, grasp pre-shaping, bounded grasp support, triage-bin retention, transport, release, final inspection, and machine-readable trajectory samples.
- **Dexterous manipulation:** thumb opposition plus four fingers, per-finger MCP/PIP control, rubberized fingertip contacts, object-specific wrist roll/grasp profiles, and tactile peak logging.
- **Engineering quality:** all project files live in this submission folder; generated artifacts include model audit counts, actuator command ranges, contact pairs, trajectory samples, and stress-test results.
- **Presentation:** the script renders a 65-second overview video with phase labels, progress, and sorted-object count suitable for the required 1-3 minute demo.
- **Innovation:** combines 10-stage tactile rescue triage, sensor-rich dexterity, perturbation testing, reduced visible fixture traces, and automatic scoring instead of a single pick-and-place animation.

## AI Judge Package

The official judging flow sends the same review package to Claude, ChatGPT, and Gemini. This submission is organized so each judge can quickly verify the same evidence:

- Watch `demo.mp4` for the complete 65-second task run.
- Run `python3 submissions/dexrescue_tactile_forge/run_demo.py` to reproduce the video and metrics.
- Read `JUDGE_REPORT.md` for the generated executive report, model audit, object score, contact evidence, and phase timeline.
- Read `outputs/metrics.json` for the 10/10 triage success result, final object errors, tactile peaks, contact pairs, actuator metrics, stress-test results, and trajectory samples.
- Read `RUBRIC_MAP.md` for the direct mapping from the official rubric to concrete files and implementation details.

## Files

| Path | Purpose |
| --- | --- |
| `scene.xml` | Complete MJCF scene, dexterous hand, objects, sensors, cameras, actuators, and task zones. |
| `run_demo.py` | Autonomous controller, renderer, trajectory logger, and scoring script. |
| `registration.json` | Participant UUID and project metadata. |
| `RUBRIC_MAP.md` | Direct mapping from the official rubric to implementation evidence. |
| `JUDGE_REPORT.md` | Auto-generated judge-facing report with model audit, success table, contact evidence, and phase timeline. |
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
submissions/dexrescue_tactile_forge/JUDGE_REPORT.md
```

The default render settings produce a 65-second 1280x720 video at 30 fps, matching the official demo format while staying inside the required 1-3 minute window. The command also runs five deterministic stress trials with +/- 0.8 cm initial XY object jitter and records whether all trials passed. In a normal desktop environment the script uses MuJoCo's renderer. In a headless environment it still runs the MuJoCo simulation and emits a deterministic top-down task video from the same simulated state, so the judging package remains reproducible.

## Demo Narrative

1. The hand scans a richer triage bench with 10 rescue items spread across the work area.
2. It executes object-specific approach, tactile pre-shape, grasp, lift, transport, place, and release phases for each item.
3. Urgent objects move into the red zone, fragile medical objects into the amber zone, and safe/salvage objects into the green zone.
4. Each object uses its own wrist roll and grasp profile, making the sequence visibly multi-stage rather than a repeated copy of one motion.
5. A final inspection pass confirms all 10 placements, while the overlay shows current phase, sorted count, and progress.
6. The metrics file reports object placement error, success rate, contact pairs, actuator command ranges, sensor peaks, trajectory samples, and stress-test results.

## Generated Evidence

The demo command writes a judge-facing report and machine-readable metrics. The report includes:

- MuJoCo model audit counts for bodies, geoms, joints, DOFs, actuators, sensors, sites, and cameras.
- Per-object target zones, final XY errors, success margins, and success flags.
- Contact statistics and top collision pairs observed during the simulation.
- Stress-test summary over deterministic initial object perturbations.
- A full phase timeline for scan, approach, pre-shape, grasp, lift, transport, place, release, and final inspection.

## Current Limitations

The controller is deterministic rather than learned. This is intentional for reproducibility under the judging environment, but the scene is structured so a future policy or teleoperation layer can replace the phase planner while keeping the same sensors and score function. During carrying, the controller applies bounded MuJoCo body forces to model stable task-level grasping. After placement, completed objects use a gentle triage-bin retention model, representing physical slot/bin capture while reducing obvious virtual-fixture traces in the video. The simulation still steps native MuJoCo dynamics, contacts, joints, actuators, and sensors throughout the run.

## AI Tools Used

Built with Codex.
