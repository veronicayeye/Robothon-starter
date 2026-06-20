Registration UUID: b5dff473-9112-4b8b-a87c-4f2c26347a0d

## Project

DexRescue Tactile Forge

## Summary

This submission is a MuJoCo dexterous-manipulation triage lab. A five-finger tactile hand autonomously sorts three emergency-response objects into triage zones while logging touch sensors, joint state, trajectory samples, rendered video, and final placement metrics.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 submissions/dexrescue_tactile_forge/run_demo.py
```

## Outputs

- `submissions/dexrescue_tactile_forge/demo.mp4`
- `submissions/dexrescue_tactile_forge/outputs/metrics.json`

The default demo is 65 seconds at 1280x720 / 30 fps and reports 3/3 successful triage placements.

## Rubric Notes

The review package is organized for the three AI judges:

- `README.md` gives the project narrative and exact run command.
- `RUBRIC_MAP.md` maps each official scoring criterion to implementation evidence.
- `scene.xml` exposes MJCF bodies, collisions, joints, actuators, cameras, and sensors.
- `run_demo.py` reproduces the autonomous controller, video, trajectory samples, tactile peaks, and final score.

## AI Tools Used

Codex
