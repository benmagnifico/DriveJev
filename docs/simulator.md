# Simulator, executor and observation contract

DriveJev drives inside [JevPilot](https://github.com/standardagents/jevpilot), a three.js driving
playground with procedurally generated towns, cities and an interstate, scripted traffic and
pedestrians, traffic lights and stop signs. JevPilot is used unmodified (git submodule
`third_party/jevpilot`); everything DriveJev adds lives in `simulator/`.

| File | Role |
|---|---|
| `simulator/world.mjs` | `World` = JevPilot `Simulation` + seeded random routes over the whole grid, pre-allocated hazard agents and a `HazardDirector`, and a dynamic-state `clone()` used by the reference teacher |
| `simulator/executor.mjs` | Semantic executor (candidate generation, lane keeping, speed profiles, ACC, optional AEB), the student observation, evaluation counters, and the privileged reference teacher |
| `simulator/harness.mjs` | Browser page: model cameras (wide + tele), batch and real-time closed loops |
| `simulator/run.mjs` | Node runner: static server, model proxy, N headless Chrome workers, `episodes.jsonl` |

## Clock

Physics always steps at a fixed `DT = 0.05 s` (20 Hz). Every 5 steps (4 Hz) the harness renders the
cameras and may ask the model for a decision. The history frame is the frame captured exactly
10 steps (0.5 s) earlier. The observation is serialized as JSON text inside the prompt, so a
variable time step would feed the model numbers it never saw; displays interpolate instead.

* **Batch closed loop** (`runEpisode`): the world waits for every model answer (zero-latency upper bound,
  the same convention as synchronous CARLA evaluation).
* **Real-time closed loop** (`runRealtime`): physics follows the wall clock and never waits; at most one
  request is in flight; an answer is applied only if its observation is at most 0.5 s old and the chosen
  behaviour is still offered with the same target.

## Cameras

| Camera | Size | Vertical FOV | Pitch | Purpose |
|---|---|---|---|---|
| `front` (t−0.5 s and t) | 640×384 | 90° | +10° | scene, near signals, motion |
| `front_tele` (t) | 384×224 | 15° | +2° | traffic lights, pedestrians and vehicles 40–90 m ahead |

Both are rendered from a dedicated scene built with JevPilot's training render profile (pixel ratio 1,
no antialiasing, no shadows, low foliage) at 1.6 m height, with the ego car and all annotations hidden.
In the 90° wide view a traffic-light lens 50 m away covers about one pixel; in the tele view it covers
about five, which is what makes stopping from city speed (18 m/s) possible.

## Behaviours (candidates)

The executor offers 2–8 behaviours per decision. Candidate IDs never appear in the prompt.

| Behaviour | Offered when | What the executor does |
|---|---|---|
| `keep_route_cruise` | moving (≥ 0.5 m/s) | follow the route at the road/curve speed limit, keep the ACC gap |
| `stop_at_line` | a junction ahead, front bumper not past the line, not already resting at it | 3.5 m/s² profile to rest 0.5 m before the line (reports `stop_unreachable` if even 8 m/s² cannot) |
| `hold_stop` | stopped | stay at rest |
| `proceed_route` | stopped | start along the route (ACC applies) |
| `route_turn` | navigation turn within 50 m | enter and follow the turn |
| `yield_agent` | moving and the perception predictor reports a path conflict | comfortable stop short of the predicted conflict point |
| `continue_current` | a stop/turn/yield is active | keep the current behaviour and its target |
| `emergency_brake` | always | maximum braking (8 m/s²) |

Rules that remove known failure modes of the earlier prototype:

* feedback (`stop_unreachable`, `turn_completed`) belongs to the current behaviour and is cleared when it changes;
* `stop_at_line` is gated by the **front bumper**, not the car centre;
* a braking behaviour started less than 0.5 s ago is not cancelled by a moving behaviour (no stop–go flapping);
* ACC: JevPilot's standstill gap plus a 0.8 s time headway to the perceived lead vehicle;
* AEB (optional, off in the main evaluation): caps speed when the perception predictor finds a collision that braking can prevent.

## Student observation

Only these fields are serialized (the compiler rejects anything else, so simulator truth cannot leak):

```json
{
  "ego": {"speed_mps", "acceleration_mps2", "steering", "route_offset_m", "heading_error_deg", "stationary_s"},
  "nav": {"remaining_m", "next_turn", "turn_distance_m", "speed_limit_mps",
          "junction_control": "signal|stop|null", "stop_line_ahead_m", "stop_completed"},
  "maneuver": {"action_type", "target_id", "elapsed_s", "progress_m", "status", "feedback"},
  "recent_actions": [{"action_type", "age_s"}],
  "traffic": {
    "lead": {"gap_m", "speed_mps"} | null,
    "hazard": {"type", "in_s", "distance_m", "side"} | null,
    "junction": {"vehicles_inside", "cross_approaching", "pedestrians_crossing", "earlier_arrivals"} | null
  }
}
```

* `junction_control` is static map information; `stop_completed` and `stationary_s` are the ego car's own memory.
* `traffic.*` is computed only from objects reported by JevPilot's sensor scan (range ≥ 80 m, 360°, occluded by
  buildings). Other vehicles' planned routes are removed before prediction (constant-velocity extrapolation).
* **Signal colours are not in the observation** – the model has to see them.

## Reference teacher (privileged)

`teacher(executor)` reads JevPilot's rule state (signal phase, stop-sign service, junction reservations, first
arrival) and rolls the **whole world** forward 3.5 s for each candidate on a dynamic-state clone, so NPCs react to
the ego as they would in reality. Pending hazard triggers that the ego could not yet observe stay frozen inside
the rollout. It picks the most progressive behaviour that is safe (no at-fault contact, clearance > 0.25 m) and
legal, and spreads probability over behaviours whose rollouts are indistinguishable. It is an upper-bound
reference policy, not part of the model.
