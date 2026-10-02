# Simulator, executor and observation contract

DriveJev drives inside [JevPilot](https://github.com/standardagents/jevpilot), a three.js driving
playground with procedurally generated towns, cities and an interstate, scripted traffic and
pedestrians, traffic lights and stop signs. JevPilot is used unmodified (git submodule
`third_party/jevpilot`); everything DriveJev adds lives in `simulator/`.

| File | Role |
|---|---|
| `simulator/world.mjs` | `DriveWorld` = JevPilot `Simulation` + seeded random routes over the whole grid, pre-allocated hazard agents and a `HazardDirector`, and a dynamic-state `clone()` used by the teachers |
| `simulator/interactions.mjs` | `InteractionWorld` (larger agent pool) and `InteractionDirector` with the six multi-agent interaction scenarios ([scenarios](scenarios.md)) |
| `simulator/executor.mjs` | Semantic executor (candidate generation, lane keeping, speed profiles, ACC, optional AEB), the student observation, evaluation counters, and the base reference teacher |
| `simulator/interaction_executor.mjs` | `InteractionExecutor` (vehicle-occlusion perception, junction ETA fields) and the observable teacher |
| `simulator/preflight.mjs` | drops suite seeds whose initial state already overlaps an NPC |
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
| `front_tele` (t) | 384×224 | 15° | +2° | traffic lights, pedestrians and vehicles 40 to 90 m ahead |

Both are rendered from a dedicated scene built with JevPilot's training render profile (pixel ratio 1,
no antialiasing, no shadows, low foliage) at 1.6 m height, with the ego car and all annotations hidden.
In the 90° wide view a traffic-light lens 50 m away covers about one pixel; in the tele view it covers
about five, which is what makes stopping from city speed (18 m/s) possible.

## Behaviours (candidates)

The executor offers 2 to 8 behaviours per decision. Candidate IDs never appear in the prompt.

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

Executor rules:

* feedback (`stop_unreachable`, `turn_completed`) belongs to the current behaviour and is cleared when it changes;
* `stop_at_line` is gated by the **front bumper**, not the car centre;
* a braking behaviour started less than 0.5 s ago is not cancelled by a moving behaviour (no stop-and-go flapping);
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
    "junction": {"vehicles_inside", "cross_approaching", "pedestrians_crossing", "earlier_arrivals",
                 "oncoming_eta_s", "cross_eta_s"} | null
  }
}
```

* `junction_control` is static map information; `stop_completed` and `stationary_s` are the ego car's own memory.
* `traffic.*` is computed only from objects reported by JevPilot's sensor scan (range ≥ 80 m, 360°, occluded by
  buildings). Other vehicles' planned routes are removed before prediction (constant-velocity extrapolation).
* Perception also lets **vehicles occlude**: a pedestrian is hidden when the line of sight from the camera passes
  through a car's footprint, a vehicle when its centre and both ends are hidden; anything within 6 m is always
  perceived. A pedestrian behind a parked car is therefore missing from the state exactly as long as it is hidden
  in the images.
* `oncoming_eta_s` / `cross_eta_s`: the time until the nearest perceived oncoming / crossing vehicle that is closing on
  the junction reaches the junction box (constant velocity; `null` beyond 12 s).
* **Signal colours are not in the observation**: the model has to see them.

## Teachers (privileged)

### Observable teacher

`observableTeacher(executor)` labels the training data of DriveJev 1.0 and drives the privileged pilot of the demo. It
reads JevPilot's rule state and rolls the world forward for every offered behaviour, like the reference teacher below,
with four rules that keep every label explainable from the student's inputs:

1. **Only what the student could have seen.** A rollout contains only road users that the perception reported
   in the last second (or that are within 8 m), so a label never reacts to a car hidden behind a building or a
   pedestrian still hidden behind a parked car.
2. **Gap acceptance with a time margin.** A behaviour is safe only if the ego never enters the zone a moving vehicle
   will cover in the next 0.6 s (its footprint stretched forward), unless that vehicle is following the ego, so left
   turns are never accepted just in front of an oncoming car.
3. **Only rear-end contacts are excused.** A contact at low ego speed is excused only when another vehicle hits the
   ego from behind, so waiting inside an oncoming lane is never labelled safe.
4. **No braking into a near miss.** If every stopping behaviour collides while continuing only lacks margin, the
   teacher continues; if everything collides it minimises the impact speed. The horizon is 4.5 s (gap acceptance
   from standstill needs more than 3.5 s).

### Reference teacher

`referenceTeacher(executor)` reads JevPilot's rule state (signal phase, stop-sign service, junction reservations, first
arrival) and rolls the **whole world** forward 3.5 s for each candidate on a dynamic-state clone, so NPCs react to
the ego as they would in reality. Pending hazard triggers that the ego could not yet observe stay frozen inside
the rollout. It picks the most progressive behaviour that is safe (no at-fault contact, clearance > 0.25 m) and
legal, and spreads probability over behaviours whose rollouts are indistinguishable. It is the upper-bound reference
of the base test suite, not part of the model.
