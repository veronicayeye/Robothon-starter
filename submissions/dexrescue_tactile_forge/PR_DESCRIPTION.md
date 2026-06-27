Registration UUID: b5dff473-9112-4b8b-a87c-4f2c26347a0d

## Project

DexRescue Tactile Forge

## Summary

This submission is a MuJoCo dexterous-manipulation triage lab. A five-finger tactile hand autonomously completes 10 emergency-response subtasks - thermal tag, medicine vial, salvage key, airway clip, IV connector, radio beacon, data chip, pressure syringe, hazmat cap, and seal puck - while logging touch sensors, contact pairs, actuator commands, joint state, trajectory samples, rendered video, perturbation stress tests, and final placement metrics.

## Judge Evidence Summary

- 10/10 objects sorted in the nominal run.
- 5/5 deterministic initial-position jitter stress trials pass at 10/10.
- 74 DOF, 24 joints, 27 geoms, 14 actuators, 10 sensors, and 2 cameras in the MuJoCo model audit.
- All 14 actuators are exercised, with 196.7851 total command variation across gantry, wrist, thumb, and four independent fingers.
- 108 unique contact pairs are observed during simulation.
- Object-specific wrist roll and grasp profiles are logged for all 10 rescue objects.
- 7/10 carried objects produce nonzero fingertip-touch peaks in the submitted metrics.
- The 65-second 1280x720 demo uses a wrist-camera main view plus an overview inset, phase label, progress, and sorted count.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 submissions/dexrescue_tactile_forge/run_demo.py
```

## Outputs

- `submissions/dexrescue_tactile_forge/demo.mp4`
- `submissions/dexrescue_tactile_forge/outputs/metrics.json`
- `submissions/dexrescue_tactile_forge/JUDGE_REPORT.md`

The default demo is 65 seconds at 1280x720 / 30 fps and reports 10/10 successful triage placements. The command also emits a judge-facing report with MuJoCo model audit counts, phase timeline, tactile evidence, actuator coverage, contact evidence, and five deterministic initial-position jitter trials at +/- 0.8 cm. The updated demo uses a wrist-camera close view plus an overview inset, phase, progress, and sorted-object count for a clearer judge-facing narrative.

## Rubric Notes

The review package is organized for the three AI judges:

- `README.md` gives the project narrative and exact run command.
- `RUBRIC_MAP.md` maps each official scoring criterion to implementation evidence.
- `scene.xml` exposes MJCF bodies, collisions, joints, actuators, cameras, and sensors.
- `run_demo.py` reproduces the autonomous controller, video, trajectory samples, tactile peaks, stress trials, model audit, and final score.
- `JUDGE_REPORT.md` is generated from the submitted code so reviewers can quickly verify the key evidence.

## AI Tools Used

Codex
