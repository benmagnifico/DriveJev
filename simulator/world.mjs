/**
 * DriveJev world: the unmodified JevPilot Simulation with
 *  - full scripted traffic and pedestrians (no "empty road" reset),
 *  - seeded random routes over the whole town/city grid (every junction type),
 *  - pre-allocated hazard agents that a HazardDirector activates mid-episode
 *    (jaywalker, red-light / stop-sign runner, hard-braking lead vehicle),
 *  - a cheap dynamic-state clone used by the privileged teacher's world rollouts.
 * JevPilot sources are imported read-only.
 */
import { Simulation } from '../third_party/jevpilot/src/simulation.js';
import { makeRoute, shortestPath } from '../third_party/jevpilot/src/world.js';
import { rng, choose, dist, heading, move, pointAt, nearestOnPath, angle } from '../third_party/jevpilot/src/math.js';
import { stopLineDistance } from '../third_party/jevpilot/src/planning.js';

export const WORLD_VERSION = 'drivejev-world-1.0';
export const HAZARD_KINDS = ['jaywalker', 'cross_runner', 'lead_brake'];
const PARK = { x: 20000, z: 20000 };

function parkedRoute(k) {
  const points = [];
  for (let s = 0; s <= 2000; s += 50) points.push({ x: PARK.x + k * 60, z: PARK.z + s, s });
  return { ids: ['__parked_a', '__parked_b'], points, crossings: [], length: 2000, parked: true };
}

export class DriveWorld extends Simulation {
  constructor(seed, type = 'town', options = {}) {
    super(seed, type);
    this.options = options;
    this.hazardRandom = rng(seed * 7 + 1234567);
    if (options.empty) { this.traffic = []; this.pedestrians = []; }
    if (Number.isFinite(options.trafficScale) && options.trafficScale !== 1 && !options.empty) {
      const keep = Math.round(this.traffic.length * Math.min(1, options.trafficScale));
      this.traffic = this.traffic.slice(0, keep);
      for (let i = this.traffic.length; i < Math.round(this.world.theme.traffic * options.trafficScale); i++) this.spawnTraffic(i);
    }
    if (options.randomRoute && type !== 'highway') this.installRandomRoute();
    // Hazard agents must exist before the scene is built so that they get meshes.
    this.hazardPeds = [];
    this.hazardCars = [];
    for (let k = 0; k < 2; k++) {
      const p = { id: `pedestrian-${900 + k}`, type: 'pedestrian', nodeId: this.world.nodes[0]?.id, x: PARK.x - 50 * k, z: PARK.z, progress: 0,
        walkPath: { start: { x: PARK.x - 50 * k, z: PARK.z }, heading: 0, length: 1 }, direction: 0, crossing: false, walking: false,
        speed: 0, width: 0.6, depth: 0.6, height: 1.7, hazard: { kind: 'jaywalker', state: 'parked' } };
      this.pedestrians.push(p); this.hazardPeds.push(p);
    }
    for (let k = 0; k < 2; k++) {
      const route = parkedRoute(k);
      const v = { id: `vehicle-${900 + k}`, type: 'car', x: route.points[0].x, z: route.points[0].z, heading: 0, speed: 0, s: 0, route, stops: {},
        width: 1.9, depth: 4.2, color: k ? '#c23b2a' : '#2a5bc2', hazard: { kind: null, state: 'parked' } };
      this.traffic.push(v); this.hazardCars.push(v);
    }
    this.director = new HazardDirector(this, options.hazards ?? []);
  }

  /** Seeded start/destination anywhere on the grid; keeps JevPilot's no-U-turn shortest path. */
  installRandomRoute() {
    const r = rng(this.world.seed * 13 + 99), world = this.world;
    for (let attempt = 0; attempt < 40; attempt++) {
      const a = choose(r, world.nodes), b = world.byId[choose(r, a.neighbors)];
      const far = world.nodes.filter((n) => dist(n, a) > 260 && n.id !== b.id);
      if (!far.length) continue;
      const dest = choose(r, far);
      let ids;
      try { ids = [a.id, ...shortestPath(world, b.id, dest.id, a.id)]; } catch { continue; }
      if (ids.length < 4) continue;
      const route = makeRoute(world, ids);
      if (route.length > (this.options.maxRouteM ?? 850) && attempt < 30) continue;
      world.route = route;
      world.destination = dest.id;
      const p = route.points[0], h = heading(p, route.points[1]);
      Object.assign(this.player, { x: p.x, z: p.z, heading: h, speed: 0, s: 0, route, stops: {}, intersectionMemory: null, steering: 0, wheelSteering: 0 });
      this.destinationApproach = route.ids.slice(-2);
      this.destinationPoint = { ...route.points.at(-1) };
      for (let i = 0; i < this.traffic.length; i++) if (dist(this.traffic[i], this.player) < 25) this.spawnTraffic(Number(this.traffic[i].id.split('-')[1]));
      return route;
    }
    return null;
  }

  rule(v, update = false) {
    const r = super.rule(v, update);
    if (v.hazard?.ignoreRules) return { ...r, mustStop: false, reason: 'Hazard agent ignores the rule' };
    return r;
  }

  speedEnvelope(v) {
    if (v.hazard) {
      if (v.hazard.state === 'parked') return { max: 0, planningMax: 0, reason: 'parked', rule: { mustStop: false, distance: Infinity }, gap: Infinity, conflict: null, lead: null, released: false };
      const env = super.speedEnvelope(v);
      if (v.hazard.ignoreRules) env.max = Math.min(Math.max(env.max, 0), v.hazard.cruise ?? env.max);
      if (v.hazard.brakeActive) env.max = 0;
      return env;
    }
    return super.speedEnvelope(v);
  }

  step(dt) {
    if (this.paused || this.crash) return;
    this.director.before(dt);
    super.step(dt);
    this.director.after();
  }

  /** Dynamic-state copy for privileged rollouts. Static world, routes and meshes are shared. */
  clone() {
    const c = Object.create(Object.getPrototypeOf(this));
    Object.assign(c, this);
    const copyCar = (v) => ({ ...v, stops: Object.fromEntries(Object.entries(v.stops ?? {}).map(([k, s]) => [k, { ...s }])),
      intersectionMemory: v.intersectionMemory ? { ...v.intersectionMemory, lastStop: v.intersectionMemory.lastStop ? { ...v.intersectionMemory.lastStop } : null } : null,
      amber: v.amber ? { ...v.amber } : v.amber, hazard: v.hazard ? { ...v.hazard } : undefined, maneuver: v.maneuver ? { ...v.maneuver } : v.maneuver });
    c.player = copyCar(this.player);
    c.traffic = this.traffic.map(copyCar);
    c.pedestrians = this.pedestrians.map((p) => ({ ...p, hazard: p.hazard ? { ...p.hazard } : undefined }));
    c.hazardPeds = c.pedestrians.filter((p) => p.hazard);
    c.hazardCars = c.traffic.filter((v) => v.hazard);
    c.locks = new Map([...this.locks].map(([k, l]) => [k, { ...l }]));
    c.courtesy = new Map([...this.courtesy].map(([k, l]) => [k, { ...l }]));
    c.events = [];
    c.contacts = new Set();
    c.r = rng(Math.floor(this.time * 1000) + 17);
    c.planRandom = rng(5);
    c.nextScan = Infinity; // perception scans are not needed inside a rollout
    c.rerouteIfNeeded = () => {};
    c.isClone = true;
    c.director = this.director.cloneFor(c);
    return c;
  }
}

/** Activates pre-allocated hazard agents relative to the ego route. */
export class HazardDirector {
  constructor(sim, kinds) {
    this.sim = sim;
    this.kinds = [...kinds];
    this.r = sim.hazardRandom;
    this.log = [];
    this.cooldownUntil = 4; // never at the very start
    this.usedJunctions = new Set();
    this.pending = [];
  }
  cloneFor(sim) {
    const d = Object.create(HazardDirector.prototype);
    Object.assign(d, this, { sim, log: [], pending: this.pending.map((p) => ({ ...p })), usedJunctions: new Set(this.usedJunctions), r: () => 0.5 });
    d.kinds = []; // rollouts never start new hazards; active ones continue
    d.frozen = true; // ...and never reveal a pending trigger the ego cannot observe yet
    return d;
  }
  freePed() { return this.sim.hazardPeds.find((p) => p.hazard.state === 'parked'); }
  freeCar() { return this.sim.hazardCars.find((v) => v.hazard.state === 'parked'); }
  record(event) { this.log.push({ time_s: Math.round(this.sim.time * 100) / 100, ...event }); }

  before(dt) {
    const sim = this.sim, v = sim.player;
    // Active pedestrians: start walking at the trigger time, freeze once across.
    for (const p of sim.hazardPeds) {
      const h = p.hazard;
      if (h.state === 'waiting' && !this.frozen) {
        const d = h.at_s - v.s - v.depth / 2;
        if (d / Math.max(v.speed, 0.5) <= h.ttc || d < 8 || sim.time - (h.armed ??= sim.time) > 25) { h.state = 'walking'; h.startAt = sim.time; p.direction = h.speed / 0.9; this.record({ kind: 'jaywalker', event: 'walk', ego_distance_m: d, ego_speed_mps: v.speed }); }
      }
      if (h.state === 'walking' && p.progress >= p.walkPath.length - 0.05) { h.state = 'done'; p.direction = 0; p.walking = false; h.doneAt = sim.time; }
      if (h.state === 'done' && sim.time - h.doneAt > 6) this.parkPed(p);
    }
    for (const car of sim.hazardCars) {
      const h = car.hazard;
      if (h.state === 'active' && h.kind === 'lead_brake' && !h.brakeActive && !this.frozen && sim.time >= h.brakeAt) { h.brakeActive = true; this.record({ kind: 'lead_brake', event: 'brake', id: car.id }); }
      if (h.state === 'active' && h.kind === 'lead_brake' && h.brakeActive && sim.time >= h.brakeAt + h.holdS) { h.brakeActive = false; h.kind = 'lead_brake_resumed'; }
      if (h.state === 'active' && (dist(car, v) > 160 || car.s >= car.route.length - 2) && sim.time - h.started > 4) this.parkCar(car);
    }
    if (!this.kinds.length || sim.isClone || sim.time < this.cooldownUntil || sim.complete || sim.crash) return;
    const kind = this.kinds[0];
    const gap = sim.options?.hazardCooldown ?? 10;
    if (this[`try_${kind}`]?.()) { this.kinds.push(this.kinds.shift()); this.cooldownUntil = sim.time + gap + this.r() * gap * 0.8; }
  }
  after() {}
  parkPed(p) {
    Object.assign(p, { x: PARK.x, z: PARK.z, progress: 0, direction: 0, walking: false, speed: 0, walkPath: { start: { ...PARK }, heading: 0, length: 1 } });
    p.hazard = { kind: 'jaywalker', state: 'parked' };
  }
  parkCar(car) {
    const route = parkedRoute(Number(car.id.split('-')[1]) - 900);
    Object.assign(car, { route, s: 0, x: route.points[0].x, z: route.points[0].z, speed: 0, stops: {}, amber: null });
    car.hazard = { kind: null, state: 'parked' };
  }
  /** Distance along the ego route to the next junction centre (route station). */
  nextJunction(minAhead = 0) {
    const v = this.sim.player;
    return v.route.crossings.find((c) => c.stopS - v.s > minAhead) ?? null;
  }

  try_jaywalker() {
    const sim = this.sim, v = sim.player, p = this.freePed();
    if (!p || v.speed < 5) return false;
    // A mid-block point 45–80 m ahead, away from junction boxes.
    const s = v.s + v.depth / 2 + 45 + this.r() * 35;
    if (s > v.route.length - 20) return false;
    if (v.route.crossings.some((c) => Math.abs(c.stopS + 10.5 - s) < 24)) return false;
    const at = pointAt(v.route.points, s), ahead = pointAt(v.route.points, s + 1), h = heading(at, ahead);
    // Ego lane centre is 3 m right of the road centre; walk from one sidewalk to the other.
    const fromRight = this.r() < 0.6;
    const side = fromRight ? 1 : -1;
    const start = move(move(at, h + Math.PI / 2, -3), h + Math.PI / 2, side * 7.2);
    const walkHeading = angle(h + (fromRight ? -Math.PI / 2 : Math.PI / 2));
    const speed = 1.4 + this.r() * 1.8;
    Object.assign(p, { x: start.x, z: start.z, progress: 0, direction: 0, walking: false, speed: 0.9, heading: walkHeading, walkPath: { start, heading: walkHeading, length: 14.4 } });
    // The pedestrian steps off the kerb when the ego is `ttc` seconds from the crossing point
    // (>= 2 s, so a prompt full brake can always avoid contact).
    const toLane = fromRight ? 4.2 : 10.2;
    const ttc = Math.max(2.0, Math.min(3.4, toLane / speed + (this.r() - 0.5) * 1.0));
    p.hazard = { kind: 'jaywalker', state: 'waiting', at_s: s, ttc, speed, from: fromRight ? 'right' : 'left', startAt: Infinity };
    this.record({ kind: 'jaywalker', id: p.id, ego_s: v.s, at_s: s, speed, ttc, from: p.hazard.from });
    return true;
  }

  try_cross_runner() {
    const sim = this.sim, v = sim.player, car = this.freeCar(), world = sim.world;
    if (!car || world.type === 'highway' || v.speed < 3) return false;
    const c = this.nextJunction(8);
    if (!c || this.usedJunctions.has(c.nodeId)) return false;
    const node = world.byId[c.nodeId];
    const egoToCentre = c.stopS + 10.5 - v.s;
    const egoT = egoToCentre / Math.max(v.speed, 1);
    if (egoT < 2.6 || egoT > 4.5) return false;
    // A perpendicular arm of this junction.
    const arms = node.neighbors.map((id) => world.byId[id]).filter((n) => Math.abs(Math.sin(heading(n, node) - c.approach)) > 0.9);
    if (!arms.length) return false;
    const from = choose(this.r, arms);
    const onward = node.neighbors.map((id) => world.byId[id]).filter((n) => n.id !== from.id && Math.abs(Math.sin(heading(node, n) - heading(from, node))) < 0.1);
    if (!onward.length) return false;
    const route = makeRoute(world, [from.id, node.id, onward[0].id]);
    const cross = route.crossings[0];
    const cruise = Math.min(world.theme.limit, 9 + this.r() * 5);
    const offset = (this.r() - 0.4) * 0.8;
    const s = Math.max(1, cross.stopS + 10.5 - cruise * (egoT + offset));
    const p = pointAt(route.points, s), nx = pointAt(route.points, s + 1);
    Object.assign(car, { route, s, x: p.x, z: p.z, heading: heading(p, nx), speed: cruise, stops: {}, amber: null });
    car.hazard = { kind: 'cross_runner', state: 'active', ignoreRules: true, cruise, started: sim.time, node: node.id, control: node.control };
    this.usedJunctions.add(c.nodeId);
    this.record({ kind: 'cross_runner', id: car.id, node: node.id, control: node.control, ego_t: egoT, cruise });
    return true;
  }

  try_lead_brake() {
    const sim = this.sim, v = sim.player, car = this.freeCar();
    if (!car || v.speed < 6) return false;
    const gap = 14 + this.r() * 14, s = v.s + v.depth / 2 + gap + 2.1;
    if (s > v.route.length - 60) return false;
    // Not inside or just before a junction (the lead would legally stop there anyway).
    if (v.route.crossings.some((c) => c.stopS - s > -25 && c.stopS - s < 45)) return false;
    if (sim.traffic.some((o) => !o.hazard && dist(o, pointAt(v.route.points, s)) < 18)) return false;
    const p = pointAt(v.route.points, s), nx = pointAt(v.route.points, s + 1);
    Object.assign(car, { route: v.route, s, x: p.x, z: p.z, heading: heading(p, nx), speed: v.speed * (0.85 + this.r() * 0.15), stops: { ...v.stops }, amber: null });
    car.hazard = { kind: 'lead_brake', state: 'active', started: sim.time, brakeAt: sim.time + 1.2 + this.r() * 2.0, holdS: 3 + this.r() * 3, brakeActive: false };
    this.record({ kind: 'lead_brake', id: car.id, gap });
    return true;
  }
}
