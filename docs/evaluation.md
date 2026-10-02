# Evaluation

DriveJev 1.0 is evaluated closed loop in JevPilot on two held-out test suites: the **interaction suite** (six multi-agent scenarios on top of the three base hazards, see [scenarios.md](scenarios.md)) and the **base suite** (random routes with the three base hazards `jaywalker` / `cross_runner` / `lead_brake`, which replays exactly with `legacyHazards: true`). The privileged teacher, which reads simulator truth, is reported as an upper bound.

## Protocol

* **Suites.** `eval/suites/interaction_test.json` (35 episodes, seeds 420000 to 420035 without 420018) and `eval/suites/test.json` (28 episodes, seeds 320000 to 320027) are run once per reported configuration. Model selection used only validation seeds (`interaction_val.json`, 411000 to 411023; `val.json`, 311000 to 311014, plus 311015 to 311029 for the base world). Training data never used any of these seeds.
* **Batch closed loop** (main tables): the world waits for each answer, as in synchronous CARLA evaluation. **Real-time closed loop**: physics follows the wall clock and never waits, at most one request is in flight, answers older than 0.5 s are dropped, and we report the *applied* decision rate.
* **AEB** (collision-mitigation braking inside the executor) is **off** in the main tables; runs with AEB on are reported separately.
* Every episode ends at arrival, at a collision or at the time limit (interaction suite 160 s, interstate 130 s; base suite 120 s, interstate 110 s).
* **Hazard collisions** count episodes that ended in contact with a scripted hazard agent, per kind, against the number of activations of that kind.

### Metrics

| Metric | Definition |
|---|---|
| Success | arrived at the destination with no collision and no traffic violation |
| Collisions | episodes that ended in contact with a vehicle, pedestrian or building |
| Violations | JevPilot red-light and missed-stop-sign events (car centre crosses the line) **plus** front-bumper crossings while a stop was required (red, amber that could still be stopped for, unserved stop sign) |
| Route completed | fraction of the route length driven |
| Mean speed | route-averaged speed |
| 95 % CI | bootstrap over episodes (2,000 resamples) |

```bash
python serve/serve.py --model checkpoints/DriveJev-4B --port 9031 &
node simulator/run.mjs --jobs eval/suites/interaction_test.json --out runs/itest --workers 4 --model-url http://127.0.0.1:9031/predict
node simulator/run.mjs --jobs eval/suites/interaction_test.json --out runs/itest-teacher --set policy=teacher   # upper bound, no GPU
node simulator/run.mjs --jobs eval/suites/test.json --out runs/test --workers 4
python eval/summarize.py runs/itest runs/itest-teacher runs/test
```

## Interaction test suite (35 episodes)

| Policy | Inputs | Success ↑ | Collisions ↓ (with hazard agent) | Violations ↓ | Route completed | Mean speed m/s |
|---|---|---|---|---|---|---|
| Observable teacher (privileged, upper bound) | simulator truth + rollouts of perceived agents | 97% <sub>[91, 100]</sub> (34/35) | 0 (0) | 0 | 99% | 7.6 |
| **DriveJev 1.0** | 3 camera frames + state | **80% <sub>[66, 91]</sub> (28/35)** | **1 (1)** | **4** | **97%** | 7.1 |
| DriveJev 1.0 + AEB | 3 camera frames + state | 80% <sub>[66, 91]</sub> (28/35) | 1 (1) | 3 | 96% | 7.0 |

Per family (successes / episodes):

| Policy | town + interaction | city + interaction | town + storm | city + storm | interstate |
|---|---|---|---|---|---|
| Observable teacher (privileged, upper bound) | 12/12 | 11/11 | 4/4 | 3/4 | 4/4 |
| **DriveJev 1.0** | **11/12** | **8/11** | **3/4** | **2/4** | **4/4** |
| DriveJev 1.0 + AEB | 11/12 | 9/11 | 2/4 | 2/4 | 4/4 |

Hazard activations and the episodes that ended in contact with that hazard agent (contacts / activations):

| Policy | oncoming | stop_contention | green_runner | occluded_ped | cut_in | turn_ped | jaywalker | cross_runner | lead_brake |
|---|---|---|---|---|---|---|---|---|---|
| Observable teacher (privileged, upper bound) | 0/23 | 0/9 | 0/8 | 0/26 | 0/24 | 0/22 | 0/91 | 0/12 | 0/27 |
| **DriveJev 1.0** | **0/16** | **0/9** | **0/6** | **0/25** | **0/24** | 1/25 | **0/92** | **0/13** | **0/32** |
| DriveJev 1.0 + AEB | 0/19 | 0/7 | 0/5 | 0/25 | 0/24 | 1/28 | 0/91 | 0/11 | 0/33 |

DriveJev 1.0 meets 241 of the 242 hazard activations of the suite without contact. None of the oncoming platoons and left-turners, pedestrians hidden behind parked cars, cut-ins, late red-light runners or four-way-stop contentions ends in contact; in every one of them the right answer is a decision (wait for a gap, yield, go) rather than a reflex.

| Real-time closed loop (8 interaction test episodes, 1 worker) | Applied decisions | Latency p50 / p95 |
|---|---|---|
| DriveJev 1.0 | 3.86 of 4 per second | 137 / 143 ms |

The model itself needs 111 ms per decision at batch size 1 (about 9 Hz, see [model.md](model.md#encoder)); the real-time latency adds camera capture, PNG encoding and the HTTP round trip.

## Base test suite (28 episodes)

| Policy | Success ↑ | Collisions ↓ | Violations ↓ | Route completed | Mean speed m/s | town | city | town + hazards | city + hazards | interstate |
|---|---|---|---|---|---|---|---|---|---|---|
| Reference teacher (privileged, upper bound) | 96% <sub>[89, 100]</sub> (27/28) | 0 | 0 | 99% | 9.0 | 8/8 | 8/8 | 4/5 | 5/5 | 2/2 |
| **DriveJev 1.0** | **89% <sub>[79, 100]</sub> (25/28)** | **0** | **2** | **100%** | 8.8 | **8/8** | **8/8** | 3/5 | 4/5 | **2/2** |

DriveJev 1.0 drives every plain town, city and interstate route of the base suite cleanly and, with the three base hazards, has no collision.

## Teachers

The observable teacher (the labels of DriveJev 1.0) and the reference teacher (all agents, 3.5 s horizon) are privileged policies, not part of the model. Every interaction scenario family was iterated until the observable teacher avoided all of its activations (see [scenarios.md](scenarios.md)), so each scenario can be solved from what the student is able to observe.

## Notes

* Batch closed-loop results are deterministic for a given seed and model; real-time results depend on latency.
* The reported runs used Google Chrome with GPU rendering (`CHROME_PATH=/usr/bin/google-chrome`, one RTX 5090). The model reads rendered camera frames, so a software renderer or another GPU can change individual pixels and, through them, individual decisions.
* `HazardDirector` tries the hazard kinds of an episode in a fixed rotation and waits until the current kind can be placed (a junction runner needs a suitable junction), so base hazard episodes contain 1 to 4 hazards depending on the route.
