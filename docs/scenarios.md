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

All hazard parameters are drawn from a per-world seeded RNG, so a `(world, seed, hazards)` triple replays
exactly. JevPilot's own interactions (stop-sign arrival order, junction reservations, queues at lights,
pedestrians crossing in the all-red walk phase, turning traffic, highway merges) are present in every episode.

## Suites

| Suite | Seeds | Episodes |
|---|---|---|
| validation (model selection only) | 311000–311014 | 4 town, 4 city, 6 hazards (3 town / 3 city), 1 highway |
| test (reported once per model) | 320000–320027 | 8 town, 8 city, 10 hazards (5 town / 5 city), 2 highway |

Training data used disjoint seeds (301000–303999 and 310000–310017 for offline validation).
