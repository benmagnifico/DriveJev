# Evaluation

## Protocol

* **Suites.** `eval/suites/val.json` (15 episodes, seeds 311000–311014) is used for every model-selection decision;
  `eval/suites/test.json` (28 episodes, seeds 320000–320027) is run once per reported model. Both mix town and city
  random routes, hazard episodes and the interstate (see [scenarios.md](scenarios.md)). Training data never used
  these seeds.
* **Batch closed loop** (main tables): the world waits for each answer, i.e. a zero-latency upper bound, as in
  synchronous CARLA evaluation. **Real-time closed loop**: physics follows the wall clock and never waits;
  answers older than 0.5 s are dropped; we report the *applied* decision rate.
* **AEB** (collision-mitigation braking inside the executor) is **off** in the main tables. Runs with AEB on are
  reported separately and never mixed into the main numbers.
* Every episode ends at arrival, at a collision, or at the time limit (120 s; 100 s on validation; 110 s on the interstate).

## Metrics

| Metric | Definition |
|---|---|
| Success | arrived at the destination with no collision and no traffic violation |
| Collisions | episodes that ended in contact with a vehicle, pedestrian or building |
| Violations | JevPilot red-light and missed-stop-sign events (car centre crosses the line) **plus** front-bumper crossings while a stop was required (red, amber that could still be stopped for, unserved stop sign) |
| Route completion | fraction of the route length driven |
| Mean speed | route-averaged speed |
| Stuck time | seconds per episode spent stationary although the rule allowed motion and no vehicle stood within 4 m ahead |
| Hard braking | seconds per km with deceleration above 5.5 m/s² |
| 95 % CI | bootstrap over episodes (2,000 resamples) |

```bash
node simulator/run.mjs --jobs eval/suites/test.json --out runs/test-ours --workers 4 --model-url http://127.0.0.1:9031/predict
python eval/summarize.py runs/test-ours
```

## Test suite (28 episodes, batch closed loop unless stated)

| Policy | Inputs | Success ↑ | Collisions ↓ | Violations ↓ | Route done | Stuck s/ep ↓ | Mean speed m/s |
|---|---|---|---|---|---|---|---|
| Reference teacher (privileged, upper bound) | simulator truth + world rollouts | 96% <sub>[89, 100]</sub> (27/28) | 0 | 0 | 99% | 0.3 | 9.0 |
| Earlier prototype, no safety brake | 2 wide frames + small state | 4% <sub>[0, 11]</sub> (1/28) | 12 | 38 | 70% | 9.3 | 7.9 |
| Earlier prototype + JevPilot safety brake | 2 wide frames + small state | 14% <sub>[4, 29]</sub> (4/28) | 0 | 51 | 91% | 18.3 | 6.8 |
| State-only policy (no camera, same labels) | state JSON | 11% <sub>[0, 21]</sub> (3/28) | 4 | 51 | 88% | 8.1 | 7.1 |
| DriveJev, no DAgger (round-1 data only) | 3 frames + state | 57% <sub>[39, 75]</sub> (16/28) | 0 | 6 | 84% | 24.0 | 6.3 |
| DriveJev, linear pointer head | 3 frames + state | 86% <sub>[71, 96]</sub> (24/28) | 1 | 4 | 97% | 0.3 | 9.5 |
| **DriveJev (ours)** | 3 frames + state | 79% <sub>[61, 93]</sub> (22/28) | 3 | 8 | 94% | 0.5 | 9.3 |
| DriveJev (ours) + AEB | 3 frames + state | 82% <sub>[68, 96]</sub> (23/28) | 1 | 11 | 99% | 0.6 | 9.1 |

| Policy | town | city | town+hazards | city+hazards | highway |
|---|---|---|---|---|---|
| Reference teacher (privileged, upper bound) | 8/8 | 8/8 | 4/5 | 5/5 | 2/2 |
| Earlier prototype, no safety brake | 0/8 | 1/8 | 0/5 | 0/5 | 0/2 |
| Earlier prototype + JevPilot safety brake | 2/8 | 1/8 | 0/5 | 1/5 | 0/2 |
| State-only policy (no camera, same labels) | 0/8 | 1/8 | 0/5 | 0/5 | 2/2 |
| DriveJev, no DAgger (round-1 data only) | 5/8 | 5/8 | 1/5 | 3/5 | 2/2 |
| DriveJev, linear pointer head | 6/8 | 8/8 | 4/5 | 4/5 | 2/2 |
| **DriveJev (ours)** | 7/8 | 7/8 | 4/5 | 2/5 | 2/2 |
| DriveJev (ours) + AEB | 7/8 | 7/8 | 4/5 | 3/5 | 2/2 |

| Real-time closed loop (8 test episodes, 1 worker) | Success | Collisions | Violations | Applied decisions | Latency p50 / p95 |
|---|---|---|---|---|---|
| DriveJev (ours) | 6/8 | 0 | 5 | 3.94 Hz | 133 / 141 ms |

AEB was triggered 21 times over the 28 AEB-on episodes. "Earlier prototype + JevPilot safety brake" uses JevPilot's
own speed envelope (following distance and swept-path conflict braking), the configuration of the original demo.

## Validation suite (15 episodes) — the data used for every model decision

| Model | Data | Success | Collisions | Violations | Stuck s/ep | Arrival |
|---|---|---|---|---|---|---|
| Reference teacher | — | 12/15 | 0 | 1 | 0.3 | 87 % |
| round 1, `pointer_mlp` | teacher drives (30 k labels) | 9/15 | 0 | 4 | 17.0 | 67 % |
| round 1, `pointer` | same | 7/15 | 0 | 9 | 5.2 | 87 % |
| round 2, `pointer_mlp` | + hazard-dense + DAgger round 1 (53 k) | 10/15 | 2 | 3 | 0.9 | 73 % |
| round 2, `pointer_mlp`, group decision rule | same | 9/15 | 2 | 5 | 0.4 | 80 % |
| **round 3, `pointer_mlp` (released)** | + DAgger round 2 (60 k) | **10/15** | **1** | 4 | **0.3** | **87 %** |
| round 3, `pointer_mlp`, group decision rule | same | 9/15 | 1 | 5 | 0.4 | 87 % |
| state-only (no camera) | round-1 data | 2/15 | 3 | 32 | 0.1 | 73 % |

The validation timeout is 100 s, which is why even the teacher misses some long hazard episodes.

## Offline agreement

Fraction of held-out validation states (2,528, seeds 310000+) where the argmax is in the teacher's set of equivalent
behaviours: 98.1–98.5 % for every head and round, and 98.4 % for the state-only ablation. The current behaviour is
part of the state, so copying it is usually right; offline agreement does not predict closed-loop quality.

## Notes

* Batch closed-loop results are deterministic for a given seed and model; real-time results depend on latency.
* `HazardDirector` tries the hazard kinds of an episode in a fixed rotation and waits until the current kind can be
  placed (a junction runner needs a suitable junction), so hazard episodes contain 1–4 hazards depending on the route.

