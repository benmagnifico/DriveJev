# Evaluation

DriveJev 1.1 adds an **interaction benchmark** (six multi-agent scenarios on top of the three 1.0 hazards, see
[scenarios.md](scenarios.md)); the 1.0 suites are kept unchanged and replay exactly (`legacyHazards: true`).
Section 1 covers the interaction suites, section 2 the 1.0 suites.

## 1. Interaction suites (DriveJev 1.1)

### Protocol

* **Suites.** `eval/suites/interaction_val.json` (24 episodes, seeds 411000–411023) is used for every 1.1 model-selection
  decision; `eval/suites/interaction_test.json` (35 episodes, seeds 420000–420035 without 420018) is run once per reported
  configuration. Families: `interaction` (all nine hazard kinds, shuffled per seed, 9 s cooldown) in town and city,
  `storm` (4 s cooldown) and the interstate with cut-ins. Time limit 160 s (interstate 130 s).
* Same batch / real-time protocols, metrics and AEB convention as below. 1.0 heads are evaluated with
  `obs_schema: '1.0'`, i.e. exactly the observation they were trained on.
* **Hazard collisions**: episodes that ended in contact with a scripted hazard agent, per kind, against the number of
  activations of that kind (episodes end at the first collision, so later hazards of a crashed episode never activate).
* **Stuck (1.1)**: stationary although the rule allows motion, no vehicle within 4 m ahead *and* the perception
  predictor reports no conflict for moving on (waiting for an oncoming gap is not counted).

```bash
node simulator/run.mjs --jobs eval/suites/interaction_test.json --out runs/itest --workers 4 --model-url http://127.0.0.1:9031/predict
node simulator/run.mjs --jobs eval/suites/interaction_test.json --out runs/itest-teacher --set policy=teacher   # upper bound, no GPU
node simulator/run.mjs --jobs eval/suites/interaction_test.json --out runs/itest-v10 --set obs_schema=1.0      # a 1.0 head
python eval/summarize.py runs/itest runs/itest-teacher
```

<!-- INTERACTION_RESULTS -->

## 2. DriveJev 1.0 suites

### Protocol

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

### Metrics

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

### Test suite (28 episodes, batch closed loop unless stated)

| Policy | Inputs | Success ↑ | Collisions ↓ | Violations ↓ | Route done | Stuck s/ep ↓ | Mean speed m/s |
|---|---|---|---|---|---|---|---|
| Reference teacher (privileged, upper bound) | simulator truth + world rollouts | 96% <sub>[89, 100]</sub> (27/28) | 0 | 0 | 99% | 0.3 | 9.0 |
| Earlier prototype, no safety brake | 2 wide frames + small state | 4% <sub>[0, 11]</sub> (1/28) | 12 | 38 | 70% | 9.3 | 7.9 |
| Earlier prototype + JevPilot safety brake | 2 wide frames + small state | 14% <sub>[4, 29]</sub> (4/28) | 0 | 51 | 91% | 18.3 | 6.8 |
| State-only policy (no camera, same labels) | state JSON | 11% <sub>[0, 21]</sub> (3/28) | 4 | 51 | 88% | 8.1 | 7.1 |
| DriveJev round 1 (no DAgger) | 3 frames + state | 57% <sub>[39, 75]</sub> (16/28) | 0 | 6 | 84% | 24.0 | 6.3 |
| DriveJev round 3 (2 DAgger rounds) | 3 frames + state | 79% <sub>[61, 93]</sub> (22/28) | 3 | 8 | 94% | 0.5 | 9.3 |
| DriveJev round 3, linear pointer head | 3 frames + state | 86% <sub>[71, 96]</sub> (24/28) | 1 | 4 | 97% | 0.3 | 9.5 |
| DriveJev round 3 + LoRA r16 (not adopted) | 3 frames + state | 71% <sub>[54, 89]</sub> (20/28) | 4 | 4 | 90% | 0.2 | 9.4 |
| **DriveJev (ours, round 4)** | 3 frames + state | 79% <sub>[61, 93]</sub> (22/28) | 2 | 5 | 95% | 0.3 | 9.4 |
| DriveJev (ours) + AEB | 3 frames + state | 79% <sub>[61, 93]</sub> (22/28) | 1 | 6 | 99% | 0.2 | 9.3 |
| DriveJev round 3 + AEB | 3 frames + state | 82% <sub>[68, 96]</sub> (23/28) | 1 | 11 | 99% | 0.6 | 9.1 |

| Policy | town | city | town+hazards | city+hazards | highway |
|---|---|---|---|---|---|
| Reference teacher (privileged, upper bound) | 8/8 | 8/8 | 4/5 | 5/5 | 2/2 |
| Earlier prototype, no safety brake | 0/8 | 1/8 | 0/5 | 0/5 | 0/2 |
| Earlier prototype + JevPilot safety brake | 2/8 | 1/8 | 0/5 | 1/5 | 0/2 |
| State-only policy (no camera, same labels) | 0/8 | 1/8 | 0/5 | 0/5 | 2/2 |
| DriveJev round 1 (no DAgger) | 5/8 | 5/8 | 1/5 | 3/5 | 2/2 |
| DriveJev round 3 (2 DAgger rounds) | 7/8 | 7/8 | 4/5 | 2/5 | 2/2 |
| DriveJev round 3, linear pointer head | 6/8 | 8/8 | 4/5 | 4/5 | 2/2 |
| DriveJev round 3 + LoRA r16 (not adopted) | 7/8 | 6/8 | 2/5 | 3/5 | 2/2 |
| **DriveJev (ours, round 4)** | 5/8 | 8/8 | 3/5 | 4/5 | 2/2 |
| DriveJev (ours) + AEB | 6/8 | 7/8 | 4/5 | 3/5 | 2/2 |
| DriveJev round 3 + AEB | 7/8 | 7/8 | 4/5 | 3/5 | 2/2 |

| Real-time closed loop (8 test episodes, 1 worker) | Success | Collisions | Violations | Applied decisions | Latency p50 / p95 | Same 8 seeds, batch |
|---|---|---|---|---|---|---|
| DriveJev (ours, round 4) | 4/8 | 2 | 6 | 3.93 Hz | 132 / 140 ms | 5/8 |
| DriveJev round 3 | 6/8 | 0 | 5 | 3.94 Hz | 133 / 141 ms | 7/8 |

AEB was triggered 23 times over the 28 AEB-on episodes of the released model. "Earlier prototype + JevPilot safety
brake" uses JevPilot's own speed envelope (following distance and swept-path conflict braking), the configuration of
the original demo.

### Validation suite (15 episodes) — the data used for every model decision

| Model | Data | Success | Collisions | Violations | Stuck s/ep | Arrival |
|---|---|---|---|---|---|---|
| Reference teacher | — | 12/15 | 0 | 1 | 0.3 | 87 % |
| round 1, `pointer_mlp` | teacher drives (30 k labels) | 9/15 | 0 | 4 | 17.0 | 67 % |
| round 1, `pointer` | same | 7/15 | 0 | 9 | 5.2 | 87 % |
| round 2, `pointer_mlp` | + hazard-dense + DAgger round 1 (53 k) | 10/15 | 2 | 3 | 0.9 | 73 % |
| round 2, `pointer_mlp`, group decision rule | same | 9/15 | 2 | 5 | 0.4 | 80 % |
| round 3, `pointer_mlp` | + DAgger round 2 (60 k) | 10/15 | 1 | 4 | 0.3 | 87 % |
| round 3, `pointer_mlp`, group decision rule | same | 9/15 | 1 | 5 | 0.4 | 87 % |
| round 3 + LoRA r16 (language layers, 9 k decisions, from the released head) | same | 10/15 | 2 | 3 | 0.3 | 80 % |
| state-only (no camera) | round-1 data | 2/15 | 3 | 32 | 0.1 | 73 % |

30-episode validation suite (the 15 episodes above + seeds 311015–311029), used to choose the round-4 model:

| Model | Data | Success | Collisions | Violations |
|---|---|---|---|---|
| round 3, `pointer_mlp` | 60 k labels | 21/30 | 1 | 10 |
| **round 4, `pointer_mlp` (released)** | + DAgger round 3 (68 k) | **21/30** | **0** | **8** |
| round 4, `pointer` | same | 21/30 | 1 | 9 |

Rule fixed before the run: most successes, then fewest collisions + violations, then keep the incumbent.

The LoRA variant was to be adopted only if it beat round 3 on validation (more successes, or equal successes with
fewer collisions + violations); it tied (10/15, 5 vs 5), so it was not adopted. Its single test run is reported as an
ablation. The validation timeout is 100 s, which is why even the teacher misses some long hazard episodes.

### Offline agreement

Fraction of held-out validation states (2,528, seeds 310000+) where the argmax is in the teacher's set of equivalent
behaviours: 98.1–98.5 % for every head and round, and 98.4 % for the state-only ablation. The current behaviour is
part of the state, so copying it is usually right; offline agreement does not predict closed-loop quality.

### Notes

* Batch closed-loop results are deterministic for a given seed and model; real-time results depend on latency.
* `HazardDirector` tries the hazard kinds of an episode in a fixed rotation and waits until the current kind can be
  placed (a junction runner needs a suitable junction), so hazard episodes contain 1–4 hazards depending on the route.

