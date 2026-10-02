# Scenarios

## Worlds

| World | JevPilot theme | Speed limit | Traffic | Junctions |
|---|---|---|---|---|
| `town` | Cedar Town, 5×5 grid | 14 m/s | 14 vehicles, 14 pedestrians | traffic lights and stop signs (every third node) |
| `city` | Skyline City, 5×5 grid, tall buildings | 18 m/s | 28 vehicles, 26 pedestrians | same, longer blocks, heavy occlusion |
| `highway` | Interstate 08 | 28 m/s | 18 vehicles | on-ramp, merge, exit, town arrival |

Every town/city episode uses a **seeded random route**: a random start node and a destination more than
260 m away (≤ 850 m route), reached by JevPilot's no-U-turn shortest path. Routes therefore cover every junction
type and turn direction instead of the single first junction used by earlier work. Highway episodes use the
JevPilot route.

## Hazard scenarios

Hazards are activated by `HazardDirector` relative to the ego route, using agents that are allocated before the
3-D scene is built (so they are rendered in the cameras). After one hazard, the next one of the episode's
list is tried after a cooldown (10–18 s by default).

| Hazard | What happens | Why it is hard |
|---|---|---|
| `jaywalker` | A pedestrian waits at the kerb 45–80 m ahead (mid-block, ≥ 24 m from junctions) and steps into the road at 1.4–3.2 m/s when the ego is 2.0–3.4 s from the crossing point; 60 % come from the right (near) side | the cue appears only ~2 s before contact; a prompt full brake always suffices, a late or comfortable one may not |
| `cross_runner` | When the ego is 2.6–4.5 s from a junction, a vehicle on a perpendicular arm drives through at 9–14 m/s ignoring its red light / stop sign, timed to arrive with the ego | the ego has right of way (green or served stop); buildings often hide the runner until late |
| `lead_brake` | A vehicle is inserted 14–28 m ahead in the ego lane at the ego speed and brakes to a stop at 7 m/s² after 1.2–3.2 s, waits 3–6 s and drives on | tests following distance and the choice between ACC, `yield_agent` and `emergency_brake` |

## Interaction scenarios (DriveJev 1.1)

`simulator/interactions.mjs` adds six multi-agent scenarios on top of the three hazards above
(`InteractionWorld` = the 1.0 world with a larger pool of pre-allocated agents; `InteractionDirector` tries every
requested kind in priority order whenever the cooldown has expired, so kinds that need a particular junction do not
block the others). Each one needs a *decision* — wait, yield, brake or go — rather than only a reflex, and most
involve several agents that react to the ego through JevPilot's own rules.

| Scenario | What happens | Why it is hard |
|---|---|---|
| `oncoming` (platoon) | The ego turns left on green (or is about to get green); 2–4 oncoming cars with right of way (they do **not** yield to the ego's junction reservation) pass the junction 1.7–4 s apart at 8–12 m/s | unprotected left turn: the ego must find a gap in a stream; turning into a short gap or stopping inside the oncoming lane ends in a T-bone |
| `oncoming` (left-turner) | The ego goes straight on green; an oncoming car turns left across its path, timed to the ego's arrival, without yielding | the conflict starts on the far side of the junction and closes fast |
| `stop_contention` | 2–3 cars reach the same all-way stop from the other arms within −2.5…+1.5 s of the ego | JevPilot's first-arrival order and junction reservations decide who goes; NPCs react to the ego's reservation, so going out of turn or waiting forever both fail |
| `green_runner` | The ego waits at a red light; 0.8–2.6 s after its light turns green a cross car runs its (late) red at 10–14 m/s | "go as soon as it is green" is wrong; the runner is often hidden by buildings until ~20 m from the junction |
| `occluded_ped` | A car is parked half on the kerb ahead; a pedestrian waits in front of its bumper (hidden from behind — in the images and in the perception summary), steps into the ego lane at 1.3–2.3 m/s, pauses there 2–3.5 s and then finishes crossing | the cue appears only when the pedestrian leaves cover. The step-out distance is recomputed every step as (hidden time + 0.6–0.9 s reaction) · u + u²/13 + slack, where u is the speed the ego could reach by full acceleration before it can react; the pedestrian is placed beyond that distance and never steps out in front of another car or later than a prompt reaction at the current speed can handle |
| `cut_in` | Town/city: a kerb-parked car 17–31 m ahead pulls out into the ego lane (13–18 m merge), half of them then brake to a stop for 2–4 s; it turns off at the next junction. Interstate: a car in the neighbour lane 9–18 m ahead at 72–87 % of the ego speed cuts in over 26–38 m and in 70 % of cases brakes to 45 % of its speed | a vehicle appears close ahead, partly beside the lane; ACC alone is not always enough |
| `turn_ped` | While the ego turns, 1–3 pedestrians (mostly from the near kerb) cross the road the ego turns into, 13.5–15.5 m past the junction centre, staggered by ~0.6 s; same step-out rules as `occluded_ped` | attention is on the junction; pedestrians arrive one after another |

Configurations used for training and evaluation:

| Family | Hazards | Cooldown |
|---|---|---|
| `interaction` | all nine kinds in a per-seed shuffled order | 9 s |
| `storm` | all nine kinds | 4 s (hazards back to back or overlapping) |
| `highway` | `cut_in` | 10 s |

Rules that keep the scenarios fair: a staged pedestrian whose crossing point the ego front has already reached
never steps out (it would walk into the side of the car); in 1.1 worlds NPC vehicles (except red-light runners) stop
for scripted pedestrians in their lane (JevPilot NPCs otherwise only yield on junction crosswalks and would drive
through a jaywalker, hiding it from the ego); `preflight.mjs` drops seeds whose initial state already overlaps an NPC
(a JevPilot spawn artefact). With these rules the observable teacher drives the interaction test suite with 0
collisions (34/35 clean, one time-out). With `legacyHazards: true` the 1.0 director and agent pool
are used, so the 1.0 suites replay exactly.

All hazard parameters are drawn from a per-world seeded RNG, so a `(world, seed, hazards)` triple replays
exactly. JevPilot's own interactions (stop-sign arrival order, junction reservations, queues at lights,
pedestrians crossing in the all-red walk phase, turning traffic, highway merges) are present in every episode.

## Suites

| Suite | File | Seeds | Episodes |
|---|---|---|---|
| 1.0 validation (model selection only) | `eval/suites/val.json` | 311000–311014 | 4 town, 4 city, 6 hazards (3 town / 3 city), 1 highway |
| 1.0 test (reported once per model) | `eval/suites/test.json` | 320000–320027 | 8 town, 8 city, 10 hazards (5 town / 5 city), 2 highway |
| interaction validation (1.1 model selection only) | `eval/suites/interaction_val.json` | 411000–411023 | 9 town + 9 city `interaction`, 2 + 2 `storm`, 2 interstate |
| interaction test (reported once per model) | `eval/suites/interaction_test.json` | 420000–420035 (420018 dropped by preflight) | 12 town + 11 city `interaction`, 4 + 4 `storm`, 4 interstate |

Interaction episodes have a 160 s limit (interstate 130 s): with hazards every 9–16 s even the reference teacher
needs up to ~150 s for an 850 m route. Training data used disjoint seeds (301000–309999 and 401000–409999;
310000–310017 and 410000–410019 for offline validation).
