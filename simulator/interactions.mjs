/**
 * DriveJev 1.1 interaction world: the DriveJev 1.0 world (full JevPilot traffic, random routes, 1.0 hazards) plus six
 * multi-agent / high-risk interaction scenarios activated by InteractionDirector:
 *
 *  oncoming        ego turns left on green  -> a platoon of 2-4 oncoming cars with right of way (they do not yield);
 *                  ego goes straight        -> an oncoming car turns left across the ego path without yielding
 *  stop_contention 2-3 cars reach the same all-way stop within about +-2 s of the ego (first-arrival order)
 *  green_runner    ego waits at red; a cross car runs its red 0.8-2.6 s after the ego's light turns green
 *  occluded_ped    a car parked half on the kerb hides a pedestrian who steps into the ego lane and pauses there
 *  cut_in          a kerb-parked car pulls out ahead (interstate: neighbour lane cuts in) and may brake
 *  turn_ped        1-3 pedestrians cross the exit road while the ego turns
 *
 * DriveWorld and the 1.0 HazardDirector are imported unchanged. With `legacyHazards: true` the 1.0 director is
 * kept, so 1.0 suites replay exactly (extra hazard agents stay parked far away).
 */
import { DriveWorld, HazardDirector, HAZARD_KINDS as BASE_KINDS } from './world.mjs';
import { makeRoute, signalState } from '../third_party/jevpilot/src/world.js';
import { choose, dist, heading, move, pointAt, nearestOnPath, angle, samplePolyline, clamp } from '../third_party/jevpilot/src/math.js';

export const WORLD_VERSION = 'drivejev-world-1.1';
export const INTERACTION_KINDS = ['oncoming', 'stop_contention', 'green_runner', 'occluded_ped', 'cut_in', 'turn_ped'];
export const ALL_KINDS = [...BASE_KINDS, ...INTERACTION_KINDS];
const PARK = { x: 20000, z: 20000 };
const EXTRA_PEDS = 4, EXTRA_CARS = 6;
const COLORS = ['#3a6f9e', '#b8483a', '#5c5f66', '#e0d6c2', '#2f7a52', '#8a6d3b'];

function parkedRoute(k) {
  const points = [];
  for (let s = 0; s <= 2000; s += 50) points.push({ x: PARK.x + k * 60, z: PARK.z + s, s });
  return { ids: ['__parked_a', '__parked_b'], points, crossings: [], length: 2000, parked: true };
}
const STAGED_PED_KINDS = new Set(['occluded_ped', 'turn_ped']);
const smooth = (t) => { const u = clamp(t, 0, 1); return u * u * (3 - 2 * u); };

/** A copy of `base` from station fromS to toS, shifted sideways by lateral(d) metres (d = distance from fromS; + = right). */
export function offsetRoute(base, fromS, toS, lateral) {
  const raw = [];
  const hAt = (s) => heading(pointAt(base.points, Math.max(0, s - 0.5)), pointAt(base.points, Math.min(base.length, s + 0.5)));
  for (let s = fromS; s <= toS + 1e-6; s += 1) raw.push(move(pointAt(base.points, s), hAt(s) + Math.PI / 2, lateral(s - fromS)));
  const points = samplePolyline(raw);
  const crossings = base.crossings.filter((c) => c.stopS > fromS + 12 && c.stopS < toS - 5)
    .map((c) => ({ ...c, stopS: nearestOnPath(pointAt(base.points, c.stopS), points).s }));
  const sections = base.sections?.map((x) => ({ ...x, startS: x.startS - fromS, endS: x.endS - fromS }));
  return { ids: base.ids, points, crossings, length: points.at(-1).s, ...(sections ? { sections } : {}), hazardRoute: true };
}

export class InteractionWorld extends DriveWorld {
  constructor(seed, type = 'town', options = {}) {
    super(seed, type, options);
    for (let k = 2; k < 2 + EXTRA_PEDS; k++) {
      const p = { id: `pedestrian-${900 + k}`, type: 'pedestrian', nodeId: this.world.nodes[0]?.id, x: PARK.x - 50 * k, z: PARK.z, progress: 0,
        walkPath: { start: { x: PARK.x - 50 * k, z: PARK.z }, heading: 0, length: 1 }, direction: 0, crossing: false, walking: false,
        speed: 0, width: 0.6, depth: 0.6, height: 1.7, hazard: { kind: 'jaywalker', state: 'parked' } };
      // Legacy (1.0) replays keep the 1.0 pool of two pedestrians and two cars; extras stay parked.
      this.pedestrians.push(p); if (!options.legacyHazards) this.hazardPeds.push(p);
    }
    for (let k = 2; k < 2 + EXTRA_CARS; k++) {
      const route = parkedRoute(k);
      const v = { id: `vehicle-${900 + k}`, type: 'car', x: route.points[0].x, z: route.points[0].z, heading: 0, speed: 0, s: 0, route, stops: {},
        width: 1.9, depth: 4.2, color: COLORS[(k - 2) % COLORS.length], hazard: { kind: null, state: 'parked' } };
      this.traffic.push(v); if (!options.legacyHazards) this.hazardCars.push(v);
    }
    if (!options.legacyHazards) this.director = new InteractionDirector(this, options.hazards ?? []);
  }

  /** Priority hazard cars obey signals but do not yield to reservations / first arrivals. */
  rule(v, update = false) {
    const r = super.rule(v, update);
    if (v.hazard?.priority && r.mustStop && ['Yield to crossing traffic', 'Letting stopped traffic clear', 'Yield to first arrival'].includes(r.reason))
      return { ...r, mustStop: false, reason: 'Hazard agent takes priority' };
    return r;
  }

  speedEnvelope(v) {
    const h = v.hazard;
    if (h && h.state === 'active') {
      if (h.static || (h.startAt && this.time < h.startAt))
        return { max: 0, planningMax: 0, reason: 'parked', rule: { mustStop: false, distance: Infinity }, gap: Infinity, conflict: null, lead: null, released: false };
      const env = super.speedEnvelope(v);
      if (Number.isFinite(h.cruise)) env.max = Math.min(env.max, h.cruise);
      if (h.capActive) env.max = Math.min(env.max, h.capTo ?? 0);
      return env;
    }
    return super.speedEnvelope(v);
  }
}

/** 1.0 director + six interaction scenarios; tries every requested kind in priority order (no stalling on unavailable kinds). */
export class InteractionDirector extends HazardDirector {
  cloneFor(sim) {
    const d = super.cloneFor(sim);
    Object.setPrototypeOf(d, InteractionDirector.prototype);
    return d;
  }
  freeCars(n) { const l = this.sim.hazardCars.filter((v) => v.hazard.state === 'parked'); return l.length >= n ? l.slice(0, n) : null; }
  freePeds(n) { const l = this.sim.hazardPeds.filter((p) => p.hazard.state === 'parked'); return l.length >= n ? l.slice(0, n) : null; }
  busy(p, radius = 9, except = []) {
    return this.sim.traffic.some((o) => !except.includes(o) && o.hazard?.state !== 'parked' && dist(o, p) < radius);
  }
  place(car, route, s, speed, hazard) {
    const p = pointAt(route.points, s), nx = pointAt(route.points, s + 1);
    Object.assign(car, { route, s, x: p.x, z: p.z, heading: heading(p, nx), speed, stops: {}, amber: null, intersectionMemory: null, waitingSince: null });
    car.hazard = { state: 'active', started: this.sim.time, ...hazard };
  }
  stagePed(p, start, walkHeading, length, hazard) {
    Object.assign(p, { x: start.x, z: start.z, progress: 0, direction: 0, walking: false, speed: 0.9, heading: walkHeading, walkPath: { start, heading: walkHeading, length } });
    p.hazard = { state: 'waiting', startAt: Infinity, ...hazard };
  }
  arm(node, wanted, min = 0.95) {
    // neighbour n whose travel direction n -> node is `wanted`
    const w = this.sim.world;
    return node.neighbors.map((id) => w.byId[id]).find((n) => Math.cos(angle(heading(n, node) - wanted)) > min) ?? null;
  }
  exitTo(node, wanted, min = 0.95) {
    const w = this.sim.world;
    return node.neighbors.map((id) => w.byId[id]).find((n) => Math.cos(angle(heading(node, n) - wanted)) > min) ?? null;
  }
  front(c) { const v = this.sim.player; return c.stopS - v.s - v.depth / 2; }
  turnOf(c) { const t = Math.sin(angle(c.exit - c.approach)); return t > 0.5 ? 'right' : t < -0.5 ? 'left' : Math.cos(angle(c.exit - c.approach)) > 0.9 ? 'straight' : 'other'; }

  before(dt) {
    const sim = this.sim, kinds = this.kinds;
    this.kinds = [];
    // A staged staged pedestrian whose crossing point the ego front has already reached never steps out
    // (it would walk into the side of the car, which no driving decision can avoid).
    for (const p of sim.hazardPeds) {
      const h = p.hazard;
      if (h.state === 'waiting' && STAGED_PED_KINDS.has(h.kind) && h.at_s - sim.player.s - sim.player.depth / 2 < 1.0) { h.state = 'done'; h.doneAt = sim.time; h.expired = true; }
    }
    for (const p of sim.hazardPeds) {
      const h = p.hazard, v = sim.player;
      if (h.kind !== 'occluded_ped') continue;
      if (h.state === 'walking' && h.pauseAt && !h.paused && p.progress >= h.pauseAt) { h.paused = true; h.resumeAt = sim.time + h.pauseS; p.direction = 0; }
      if (h.state === 'walking' && h.paused && h.resumeAt && sim.time >= h.resumeAt && !this.frozen) { h.resumeAt = null; p.direction = h.speed / 0.9; }
      if (h.state !== 'staged' || this.frozen) continue;
      // Steps out when the ego front is (hidden time + reaction) * v + v^2 / (2 * 6.5) + slack away, re-evaluated every
      // step with the speed the ego can reach while the pedestrian is still hidden: avoidable for a prompt reaction.
      const d = h.at_s - v.s - v.depth / 2, hidden = 0.7 / h.speed, u = Math.min(sim.world.theme.limit, Math.max(v.speed, 0.5) + 3 * hidden);
      if (d < 1.0) { h.state = 'done'; h.doneAt = sim.time; h.expired = true; continue; }
      if (d <= (hidden + h.reaction) * u + (u * u) / 13 + h.slack) {
        h.state = 'walking'; h.startAt = sim.time; p.direction = h.speed / 0.9;
        this.record({ kind: 'occluded_ped', event: 'walk', ego_distance_m: d, ego_speed_mps: v.speed });
      }
    }
    super.before(dt); // 1.0 agent updates (pedestrian triggers, lead brake, parking); rotation handled below
    this.kinds = kinds;
    for (const car of sim.hazardCars) {
      const h = car.hazard;
      if (h.state !== 'active' || h.kind !== 'cut_in') continue;
      // After merging the car either drives on at the road speed or brakes (to a stop in town) and then drives on.
      if (!h.merged && car.s >= h.mergeEndS) { h.merged = true; h.mergedAt = sim.time; if (h.brake) h.brakeAt = sim.time + h.brakeDelay; else h.cruise = Infinity; }
      if (h.brakeAt && !h.capActive && !this.frozen && sim.time >= h.brakeAt) { h.capActive = true; this.record({ kind: 'cut_in', event: 'brake', id: car.id, cap_mps: h.capTo }); }
      if (h.capActive && sim.time >= h.brakeAt + h.holdS) { h.capActive = false; h.brakeAt = null; h.brake = false; h.cruise = Infinity; }
    }
    for (const car of sim.hazardCars) {
      // Kerb-parked occluders return to the parking lot once the ego has passed.
      const h = car.hazard;
      if (h.state === 'active' && h.static && sim.player.s > h.passS) this.parkCar(car);
    }
    if (!kinds.length || sim.isClone || sim.time < this.cooldownUntil || sim.complete || sim.crash) return;
    const gap = sim.options?.hazardCooldown ?? 10;
    for (let i = 0; i < kinds.length; i++) {
      const kind = kinds[i];
      if (this[`try_${kind}`]?.()) {
        kinds.splice(i, 1); kinds.push(kind);
        this.cooldownUntil = sim.time + gap + this.r() * gap * 0.8;
        break;
      }
    }
  }

  // ---------------------------------------------------------------- oncoming
  try_oncoming() {
    const sim = this.sim, v = sim.player, world = sim.world;
    if (world.type === 'highway') return false;
    const c = this.nextJunction(-1);
    if (!c || this.usedJunctions.has(c.nodeId)) return false;
    const node = world.byId[c.nodeId];
    if (node.control !== 'signal') return false;
    const front = this.front(c);
    if (front < -0.5 || front > 80) return false;
    const sig = signalState(node, sim.time, c.approach);
    // The ego reaches the junction centre after the light is (or turns) green.
    const travel = (c.stopS + 10.5 - v.s) / Math.max(v.speed, 5) + (v.speed < 2 ? 1.5 : 0);
    let egoT;
    if (sig.color === 'green') { if (v.speed < 2 && front > 3) return false; egoT = travel; if (sig.remaining < egoT + 1.0) return false; }
    else if (sig.color === 'red' && sig.remaining < 4.5 && front < 45) egoT = Math.max(travel, sig.remaining + 1.8);
    else return false;
    const opp = this.arm(node, c.approach + Math.PI);
    if (!opp) return false;
    const turn = this.turnOf(c);
    if (turn === 'left') {
      const exit = this.exitTo(node, c.approach + Math.PI);
      if (!exit) return false;
      const count = 2 + Math.floor(this.r() * 3), cars = this.freeCars(count);
      if (!cars) return false;
      const route = makeRoute(world, [opp.id, node.id, exit.id]), centreS = route.crossings[0].stopS + 10.5;
      const speed = Math.min(world.theme.limit, 8 + this.r() * 4);
      let t = egoT - 1.2 + this.r() * 1.6;
      const plan = [];
      for (let k = 0; k < count; k++) {
        const s = centreS - speed * t;
        if (s < 1 || this.busy(pointAt(route.points, s), 9, cars)) break;
        plan.push(s);
        t += 1.7 + this.r() * 2.3;
      }
      if (plan.length < 2) return false;
      plan.forEach((s, k) => this.place(cars[k], route, s, speed, { kind: 'oncoming', priority: true, cruise: speed, node: node.id }));
      this.usedJunctions.add(c.nodeId);
      this.record({ kind: 'oncoming', variant: 'platoon', node: node.id, cars: plan.length, speed, ego_t: egoT });
      return true;
    }
    if (turn === 'straight') {
      const exit = this.exitTo(node, c.approach + Math.PI / 2); // the oncoming car's left turn crosses the ego lane
      const car = this.freeCars(1)?.[0];
      if (!exit || !car) return false;
      const route = makeRoute(world, [opp.id, node.id, exit.id]), centreS = route.crossings[0].stopS + 10.5;
      const speed = Math.min(world.theme.limit, 7 + this.r() * 3);
      const t = egoT + (this.r() - 0.5) * 1.2;
      const s = centreS - 8 - speed * Math.max(0.5, t - 1.0); // slows to ~6.5 m/s for the turn
      if (s < 1 || this.busy(pointAt(route.points, s))) return false;
      this.place(car, route, s, speed, { kind: 'oncoming', variant: 'left_turner', priority: true, cruise: speed, node: node.id });
      this.usedJunctions.add(c.nodeId);
      this.record({ kind: 'oncoming', variant: 'left_turner', node: node.id, speed, ego_t: egoT });
      return true;
    }
    return false;
  }

  // --------------------------------------------------------- stop_contention
  try_stop_contention() {
    const sim = this.sim, v = sim.player, world = sim.world;
    if (world.type === 'highway' || v.speed < 2) return false;
    const c = this.nextJunction(0);
    if (!c || this.usedJunctions.has(c.nodeId)) return false;
    const node = world.byId[c.nodeId];
    if (node.control !== 'stop') return false;
    const front = this.front(c);
    if (front < 22 || front > 70) return false;
    const egoLineT = front / Math.max(v.speed, 4) + Math.max(v.speed, 4) / 7;
    const arms = node.neighbors.map((id) => world.byId[id]).filter((n) => Math.cos(angle(heading(n, node) - c.approach)) < 0.9);
    if (arms.length < 1) return false;
    const count = Math.min(arms.length, 2 + (this.r() < 0.5 ? 1 : 0)), cars = this.freeCars(count);
    if (!cars) return false;
    const order = [...arms].sort(() => this.r() - 0.5).slice(0, count);
    const placed = [];
    order.forEach((arm, k) => {
      const onward = node.neighbors.map((id) => world.byId[id]).filter((n) => n.id !== arm.id);
      const out = choose(this.r, onward);
      const route = makeRoute(world, [arm.id, node.id, out.id]), stopAt = route.crossings[0].stopS - 2.3;
      const u = Math.min(world.theme.limit, 7 + this.r() * 4), T = egoLineT + (-2.5 + this.r() * 4.0);
      const D = u * (T - u / 10);
      const s = stopAt - D;
      if (D < 4 || s < 1 || this.busy(pointAt(route.points, s), 9, cars)) return;
      this.place(cars[k], route, s, u, { kind: 'stop_contention', cruise: u, node: node.id, arrive_offset_s: T - egoLineT });
      placed.push({ id: cars[k].id, arm: arm.id, out: out.id, offset_s: Math.round((T - egoLineT) * 10) / 10 });
    });
    if (!placed.length) return false;
    this.usedJunctions.add(c.nodeId);
    this.record({ kind: 'stop_contention', node: node.id, cars: placed });
    return true;
  }

  // ------------------------------------------------------------ green_runner
  try_green_runner() {
    const sim = this.sim, v = sim.player, world = sim.world;
    if (world.type === 'highway' || v.speed > 3) return false;
    const c = this.nextJunction(-1);
    if (!c || this.usedJunctions.has(c.nodeId)) return false;
    const node = world.byId[c.nodeId];
    if (node.control !== 'signal') return false;
    const front = this.front(c);
    if (front < -0.5 || front > 25) return false;
    const sig = signalState(node, sim.time, c.approach);
    if (sig.color !== 'red' || sig.remaining < 0.6 || sig.remaining > 3.5) return false;
    const arms = node.neighbors.map((id) => world.byId[id]).filter((n) => Math.abs(Math.sin(angle(heading(n, node) - c.approach))) > 0.9);
    const car = this.freeCars(1)?.[0];
    if (!arms.length || !car) return false;
    const from = choose(this.r, arms), onward = this.exitTo(node, heading(from, node));
    if (!onward) return false;
    const route = makeRoute(world, [from.id, node.id, onward.id]), centreS = route.crossings[0].stopS + 10.5;
    const u = Math.min(world.theme.limit, 10 + this.r() * 4), delay = 0.8 + this.r() * 1.8;
    const s = centreS - u * (sig.remaining + delay);
    if (s < 1 || this.busy(pointAt(route.points, s))) return false;
    this.place(car, route, s, u, { kind: 'green_runner', ignoreRules: true, cruise: u, node: node.id });
    this.usedJunctions.add(c.nodeId);
    this.record({ kind: 'green_runner', id: car.id, node: node.id, green_in_s: sig.remaining, delay_s: delay, speed: u });
    return true;
  }

  // ------------------------------------------------------------ occluded_ped
  try_occluded_ped() {
    const sim = this.sim, v = sim.player, world = sim.world;
    if (world.type === 'highway' || v.speed < 6) return false;
    const car = this.freeCars(1)?.[0], p = this.freePeds(1)?.[0];
    if (!car || !p) return false;
    const s = v.s + v.depth / 2 + 42 + this.r() * 30; // crossing station on the ego route
    if (s > v.route.length - 25) return false;
    if (v.route.crossings.some((c) => Math.abs(c.stopS + 10.5 - s) < 26)) return false;
    const carS = s - 2.1 - 0.5;
    const route = offsetRoute(v.route, carS - 6, Math.min(v.route.length, carS + 80), () => 3.0); // long enough that JevPilot never re-routes it
    if (this.busy(pointAt(route.points, 6), 7)) return false;
    this.place(car, route, 6, 0, { kind: 'occluder', static: true, passS: s + 8 });
    const at = pointAt(v.route.points, s), h = heading(at, pointAt(v.route.points, s + 1));
    // Waits in front of the parked car's bumper (hidden from behind), steps into the ego lane, pauses there
    // (looking along the far lane) and then finishes crossing.
    const start = move(at, h + Math.PI / 2, 2.85), walkHeading = angle(h - Math.PI / 2);
    const speed = 1.3 + this.r() * 1.0, reaction = 0.6 + this.r() * 0.3, slack = 1.5 + this.r() * 2.5;
    this.stagePed(p, start, walkHeading, 8.35, { kind: 'occluded_ped', at_s: s, speed, reaction, slack, pauseAt: 3.05, pauseS: 2 + this.r() * 1.5, from: 'right', occluder: car.id });
    p.hazard.state = 'staged';
    this.record({ kind: 'occluded_ped', id: p.id, occluder: car.id, at_s: s, speed, reaction, slack });
    return true;
  }

  // ------------------------------------------------------------------ cut_in
  try_cut_in() {
    const sim = this.sim, v = sim.player, world = sim.world;
    if (v.speed < 7) return false;
    const car = this.freeCars(1)?.[0];
    if (!car) return false;
    const highway = world.type === 'highway';
    const gap = highway ? 9 + this.r() * 9 : 17 + this.r() * 14;
    const startS = v.s + v.depth / 2 + gap + 2.1;
    const merge = highway ? 26 + this.r() * 12 : 13 + this.r() * 5;
    if (startS + merge + 60 > v.route.length) return false;
    if (v.route.crossings.some((c) => c.stopS - startS > -15 && c.stopS - startS < merge + 30)) return false;
    if (highway) {
      const sec = v.route.sections?.find((x) => x.endS > v.s);
      if (!sec || sec.kind !== 'interstate' || sec.endS < startS + merge + 40) return false;
    }
    const L0 = highway ? -4.5 : 3.0;
    const endS = Math.min(v.route.length, startS + 220);
    let route = offsetRoute(v.route, startS, endS, (d) => L0 * (1 - smooth((d - 1) / merge)));
    if (!highway) route = this.turnAway(route, startS) ?? route;
    if (this.busy(pointAt(route.points, 0), 12) || this.busy(pointAt(v.route.points, startS + merge), 14)) return false;
    const speed = highway ? v.speed * (0.72 + this.r() * 0.15) : 0;
    const cruise = highway ? speed : 7 + this.r() * 3;
    const brake = this.r() < (highway ? 0.7 : 0.5);
    this.place(car, route, 0.5, speed, { kind: 'cut_in', cruise, mergeEndS: merge, brake, brakeDelay: 0.4 + this.r() * 1.0,
      capTo: highway ? cruise * 0.45 : 0, holdS: 2 + this.r() * 2, startAt: highway ? 0 : sim.time + 0.2 });
    this.record({ kind: 'cut_in', id: car.id, gap, merge, highway, brake });
    return true;
  }

  /** Re-join `route` (an offset copy of the ego route from station startS) to a road that leaves the ego route at the next junction. */
  turnAway(route, startS) {
    const sim = this.sim, v = sim.player, world = sim.world;
    const next = v.route.crossings.find((c) => c.stopS > startS + 30);
    if (!next) return null;
    const i = v.route.ids.indexOf(next.nodeId);
    if (i < 1 || i >= v.route.ids.length - 1) return null;
    const P = v.route.ids[i - 1], Q = v.route.ids[i + 1], node = world.byId[next.nodeId];
    const exits = node.neighbors.filter((id) => id !== P && id !== Q);
    if (!exits.length) return null;
    const alt = makeRoute(world, [P, next.nodeId, choose(this.r, exits)]);
    const joinAt = next.stopS - startS - 25; // well after the merge, before the junction
    const head = route.points.filter((q) => q.s <= joinAt);
    const j = nearestOnPath(head.at(-1), alt.points);
    if (j.distance > 0.6) return null;
    const points = samplePolyline([...head, ...alt.points.filter((q) => q.s > j.s + 1)]);
    const crossings = alt.crossings.map((c) => ({ ...c, stopS: nearestOnPath(pointAt(alt.points, c.stopS), points).s }));
    return { ids: alt.ids, points, crossings, length: points.at(-1).s, hazardRoute: true };
  }

  // ---------------------------------------------------------------- turn_ped
  try_turn_ped() {
    const sim = this.sim, v = sim.player, world = sim.world;
    if (world.type === 'highway' || v.speed < 2) return false;
    const c = this.nextJunction(0);
    if (!c) return false;
    const turn = this.turnOf(c);
    if (turn !== 'left' && turn !== 'right') return false;
    const front = this.front(c);
    if (front < 8 || front > 55) return false;
    const node = world.byId[c.nodeId];
    const n = 1 + Math.floor(this.r() * 3), peds = this.freePeds(n);
    if (!peds) return false;
    const X = move(node, c.exit, 13.5 + this.r() * 2.0);
    const laneAt = move(X, c.exit + Math.PI / 2, 3);
    const near = nearestOnPath(laneAt, v.route.points);
    if (near.distance > 1.5) return false;
    const atS = near.s;
    const placed = [];
    peds.forEach((p, k) => {
      const fromRight = k === 0 || this.r() < 0.7, side = fromRight ? 1 : -1;
      const start = move(move(X, c.exit, (k - (n - 1) / 2) * 0.9), c.exit + Math.PI / 2, side * 7.2);
      const walkHeading = angle(c.exit + (fromRight ? -Math.PI / 2 : Math.PI / 2));
      const speed = 1.2 + this.r() * 1.2, ttc = 2.4 + this.r() * 1.4 + k * 0.6;
      this.stagePed(p, start, walkHeading, 14.4, { kind: 'turn_ped', at_s: atS, ttc, speed, from: fromRight ? 'right' : 'left' });
      placed.push({ id: p.id, ttc: Math.round(ttc * 10) / 10, speed: Math.round(speed * 10) / 10, from: fromRight ? 'right' : 'left' });
    });
    this.record({ kind: 'turn_ped', node: node.id, turn, peds: placed });
    return true;
  }
}
