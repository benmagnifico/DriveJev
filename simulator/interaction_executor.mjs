/**
 * DriveJev interaction executor (observation schema 1.1: perception with vehicle occlusion) + observable
 * world-rollout teacher.
 *
 * The behaviour set, controller and candidate rules are the base ones (SemanticExecutor, unchanged). Schema 1.1 adds:
 *  - perception: vehicles occlude pedestrians and other vehicles (JevPilot's scan only models buildings);
 *    a parked car can now hide a pedestrian from the sensors as it does in the camera images;
 *  - two perception-derived junction fields: `oncoming_eta_s` / `cross_eta_s` = time until the nearest
 *    perceived oncoming / crossing vehicle reaches the junction box (constant-velocity estimate);
 *  - `obsSchema: '1.0'` reproduces the schema-1.0 student observation exactly (for evaluating schema-1.0 heads);
 *  - observableTeacher: (1) rollouts only contain road users the student's sensors have seen in the last 1 s
 *    (or that are within 8 m), so every label is explainable from the student inputs; (2) a collision is
 *    only excused when the ego is rear-ended (the base teacher excused every contact at ego speed <= 0.4 m/s, which let
 *    the teacher wait inside an oncoming lane); (3) 4.5 s horizon for gap acceptance from standstill;
 *    (4) a behaviour is only "safe" if the ego never enters the 0.6 s zone in front of a moving vehicle that is
 *    not following it (gap acceptance with a time margin instead of a 0.25 m footprint clearance);
 *    (5) when no stopping behaviour is collision-free, a motion that only lacks margin beats a colliding stop.
 */
import { angle, clamp, dist, heading } from '../third_party/jevpilot/src/math.js';
import { stopLineDistance } from '../third_party/jevpilot/src/planning.js';
import { leadVehicle, predictTrafficConflict, footprintClearance, relativeTrafficState } from '../third_party/jevpilot/src/traffic-safety.js';
import { SemanticExecutor, ACTIONS, DT, maneuverTarget, geometricLimit } from './executor.mjs';

export const VERSION = 'drivejev-exec-1.1';
export { ACTIONS, DT };
const r1 = (x) => (Number.isFinite(x) ? Math.round(x * 10) / 10 : null);

/** Does segment a->b pass through the oriented footprint of `box` (shrunk by `pad`)? */
export function segmentHitsBox(a, b, box, pad = -0.1) {
  const sin = Math.sin(box.heading || 0), cos = Math.cos(box.heading || 0);
  const loc = (p) => { const dx = p.x - box.x, dz = p.z - box.z; return [dx * sin - dz * cos, dx * cos + dz * sin]; };
  const A = loc(a), B = loc(b), half = [(box.depth || 4.2) / 2 + pad, (box.width || 1.9) / 2 + pad];
  let tmin = 0.02, tmax = 0.98;
  for (let k = 0; k < 2; k++) {
    const d = B[k] - A[k];
    if (Math.abs(d) < 1e-9) { if (A[k] < -half[k] || A[k] > half[k]) return false; continue; }
    let t1 = (-half[k] - A[k]) / d, t2 = (half[k] - A[k]) / d;
    if (t1 > t2) [t1, t2] = [t2, t1];
    tmin = Math.max(tmin, t1); tmax = Math.min(tmax, t2);
    if (tmax < tmin) return false;
  }
  return true;
}

export class InteractionExecutor extends SemanticExecutor {
  constructor(sim, { aeb = false, obsSchema = '1.1' } = {}) {
    super(sim, { aeb });
    this.obsSchema = obsSchema;
    this.vehicleOcclusion = obsSchema !== '1.0';
    this.seen = new Map();
    this.percCache = null;
  }

  eye() { const v = this.sim.player; return { x: v.x + Math.sin(v.heading) * 0.15, z: v.z - Math.cos(v.heading) * 0.15 }; }

  visible(o, occluders, eye) {
    const d = dist(o, eye);
    if (d < 6) return true;
    const blockers = occluders.filter((b) => b !== o && b.id !== o.id && dist(b, eye) < d);
    if (!blockers.length) return true;
    const hidden = (p) => blockers.some((b) => segmentHitsBox(eye, p, b));
    if (o.type === 'pedestrian') return !hidden(o);
    const f = 0.4 * (o.depth || 4.2), s = Math.sin(o.heading || 0), c = Math.cos(o.heading || 0);
    return !(hidden(o) && hidden({ x: o.x + s * f, z: o.z - c * f }) && hidden({ x: o.x - s * f, z: o.z + c * f }));
  }

  perceived() {
    const base = super.perceived();
    if (!this.vehicleOcclusion) { this.markSeen(base); return base; }
    if (this.percCache?.step === this.stepId && this.percCache.time === this.sim.time) return this.percCache.value;
    const eye = this.eye();
    const occluders = base.vehicles.filter((o) => o.type === 'car' && dist(o, eye) < 70);
    const value = { vehicles: base.vehicles.filter((o) => this.visible(o, occluders, eye)), pedestrians: base.pedestrians.filter((o) => this.visible(o, occluders, eye)) };
    this.percCache = { step: this.stepId, time: this.sim.time, value };
    this.markSeen(value);
    return value;
  }
  markSeen(p) { const t = this.sim.time; for (const o of p.vehicles) this.seen.set(o.id, t); for (const o of p.pedestrians) this.seen.set(o.id, t); }

  junctionSummary(c, p) {
    const base = super.junctionSummary(c, p);
    if (!base || this.obsSchema === '1.0') return base;
    const node = this.sim.world.byId[c.nodeId];
    let oncoming = Infinity, cross = Infinity;
    for (const o of p.vehicles) {
      const d = dist(o, node);
      if (d < 11 || d > 75) continue;
      const toNode = heading(o, node), closing = Math.cos(angle((o.heading ?? 0) - toNode)) * (o.speed ?? 0);
      if (closing < 0.5 || Math.cos(angle((o.heading ?? 0) - toNode)) < 0.8) continue;
      const eta = (d - 11) / closing, rel = angle((o.heading ?? 0) - c.approach);
      if (Math.cos(rel) < -0.8) oncoming = Math.min(oncoming, eta);
      else if (Math.abs(Math.sin(rel)) > 0.8) cross = Math.min(cross, eta);
    }
    return { ...base, oncoming_eta_s: oncoming < 12 ? r1(oncoming) : null, cross_eta_s: cross < 12 ? r1(cross) : null };
  }

  /** Evaluation only: stationary although the rule allows motion and nothing perceived blocks or conflicts. */
  unnecessaryStop() {
    const sim = this.sim, v = sim.player;
    if (Math.abs(v.speed) >= 0.2 || this.stepId <= 20 || sim.rule(v).mustStop) return false;
    const env = this.env();
    if (env.lead && env.lead.gap < 4) return false;
    const c = this.predictConflict(env.perceived);
    return !c;
  }
}

// ------------------------------------------------------------------------------------
// Observable privileged teacher (see header).
// ------------------------------------------------------------------------------------
const HORIZON_STEPS = 90; // 4.5 s
const SEEN_WINDOW_S = 1.0, NEAR_M = 8;

function observableClone(ex) {
  const sim = ex.sim.clone(), v = sim.player, t = ex.sim.time;
  const keep = (o) => (o.hazard && o.hazard.state === 'parked') || dist(o, v) < NEAR_M || (ex.seen.get(o.id) ?? -Infinity) >= t - SEEN_WINDOW_S;
  sim.traffic = sim.traffic.filter(keep);
  sim.pedestrians = sim.pedestrians.filter(keep);
  sim.hazardPeds = sim.pedestrians.filter((p) => p.hazard);
  sim.hazardCars = sim.traffic.filter((o) => o.hazard);
  return sim;
}

function rearEnded(v, other) {
  if (!other || !['car', 'motorcycle'].includes(other.type)) return false;
  const rel = relativeTrafficState(v, other);
  return rel.ahead_m < 0 && Math.abs(rel.heading_relative_deg) < 50;
}

/** The zone a moving vehicle will cover in the next 0.6 s (its footprint stretched forward). */
function frontZone(o) {
  const b = clamp((o.speed || 0) * 0.6, 0, 8);
  if (b < 0.3) return null;
  const s = Math.sin(o.heading || 0), c = Math.cos(o.heading || 0);
  return { ...o, x: o.x + (s * b) / 2, z: o.z - (c * b) / 2, depth: (o.depth || 4.2) + b };
}

function rollout(ex, candidate) {
  const sim = observableClone(ex), v = sim.player;
  const m = candidate.action_type === 'continue_current' && ex.maneuver ? { ...ex.maneuver, feedback: [] }
    : { action_type: candidate.action_type, line: candidate.action_type === 'stop_at_line' ? ex.line() : null, exit_s: null };
  const startS = v.s, startViol = sim.violations, startSpeed = v.speed;
  let collision = null, maxDecel = 0, minGap = Infinity, prev = v.speed, stoppedAt = null, intrusion = null;
  const dyn = () => [...sim.traffic.filter((o) => !o.hazard || o.hazard.state !== 'parked'), ...sim.pedestrians.filter((o) => !o.hazard || o.hazard.state !== 'parked')];
  for (let k = 0; k < HORIZON_STEPS; k++) {
    v.maneuver = { lane_offset_m: 0, lookahead_m: clamp(3 + Math.abs(v.speed) * 0.45, 3, 10) };
    const lead = leadVehicle(v, sim.traffic);
    let conflict = null;
    if (m.action_type === 'yield_agent') {
      const ghost = { ...v, target: geometricLimit(sim, v), maneuver: v.maneuver };
      const c = predictTrafficConflict(ghost, dyn().filter((o) => dist(o, v) < 70));
      conflict = c && c.braking_reduces_risk ? c : null;
    }
    v.target = maneuverTarget(sim, v, m, { lead: lead && lead.gap < 80 ? lead : null, conflict }).target;
    sim.autopilot = true; sim.safety = false;
    sim.step(DT);
    if (m.action_type === 'stop_at_line' && v.speed < 0.05 && m.line && stopLineDistance(v, m.line) <= 0.65) m.status = 'completed';
    maxDecel = Math.max(maxDecel, (prev - v.speed) / DT); prev = v.speed;
    if (stoppedAt === null && v.speed < 0.05) stoppedAt = k * DT;
    if (sim.crash) {
      const other = [...sim.traffic, ...sim.pedestrians].find((o) => o.id === sim.crash.object_id);
      collision = { ...sim.crash, at_s: (k + 1) * DT, rear_end: rearEnded(v, other) };
      break;
    }
    for (const o of dyn()) {
      const d = dist(o, v);
      if (d < 9) minGap = Math.min(minGap, footprintClearance(v, o) - (o.type === 'pedestrian' ? 0.05 : 0.02) * Math.abs(v.speed));
      if (intrusion === null && d < 20 && o.type !== 'pedestrian' && !rearEnded(v, o)) {
        const z = frontZone(o);
        if (z && footprintClearance(v, z) < 0) intrusion = { id: o.id, at_s: (k + 1) * DT };
      }
    }
    if (sim.complete) break;
  }
  const atFault = collision ? !collision.rear_end : false;
  return { collision: atFault, collision_any: !!collision, collision_at_s: collision?.at_s ?? null, collision_type: collision?.type ?? null, collision_speed_mps: collision?.player_speed_mps ?? null,
    violation: sim.violations > startViol, progress_m: v.s - startS, final_speed_mps: v.speed, max_decel_mps2: maxDecel, min_clearance_m: minGap, start_speed_mps: startSpeed,
    intrusion: !!intrusion, intrusion_at_s: intrusion?.at_s ?? null };
}

export function observableTeacher(ex, obs = ex.observe()) {
  const sim = ex.sim, v = sim.player, c = ex.crossing(), rule = sim.rule(v), ids = obs.candidates.map((x) => x.candidate_id);
  const has = (id) => ids.includes(id), nav = obs.student_obs.nav, front = c ? ex.frontDistance(c) : null;
  const moving = Math.abs(v.speed) >= 0.5;
  const scores = {};
  const score = (id) => (scores[id] ??= rollout(ex, obs.candidates.find((x) => x.candidate_id === id)));
  // Same observability corrections as referenceTeacher (future signal phase; reservation holders already outside the box).
  const node = c ? sim.world.byId[c.nodeId] : null;
  const futureSignalUnknown = node?.control === 'signal' && rule.color === 'green';
  const legal = (s) => !s.violation || futureSignalUnknown;
  const safe = (id) => { const s = score(id); return !s.collision && legal(s) && s.min_clearance_m > 0.25 && !s.intrusion; };
  const clean = (id) => { const s = score(id); return !s.collision && legal(s); };
  let ruleStop = rule.mustStop;
  if (ruleStop && rule.reason === 'Yield to crossing traffic' && rule.stopCompleted && node) {
    const lock = sim.locks.get(node.id), holder = lock ? sim.traffic.find((o) => o.id === lock.id) : null;
    if (holder && dist(holder, node) > 12) ruleStop = false;
  }
  const mustStop = c && ruleStop && front !== null && front > -0.3 && front < 85;
  let preferred = null, reason = null;
  if (mustStop) {
    const atLine = !moving && front <= 1.0;
    preferred = atLine ? 'hold_stop' : has('stop_at_line') ? 'stop_at_line' : !moving ? 'hold_stop' : 'emergency_brake';
    reason = `rule:${rule.reason}`;
    if (preferred === 'stop_at_line' && !safe('stop_at_line')) {
      preferred = has('yield_agent') && safe('yield_agent') ? 'yield_agent' : 'emergency_brake';
      reason += score('stop_at_line').collision ? '+collision' : '+clearance';
    }
  } else {
    const motion = !moving ? 'proceed_route' : has('route_turn') && ['left', 'right'].includes(nav.next_turn) && nav.turn_distance_m < 24 ? 'route_turn' : 'keep_route_cruise';
    if (safe(motion)) { preferred = motion; reason = 'clear'; }
    else {
      const sm = score(motion);
      reason = `conflict:${sm.collision ? 'collision' : !legal(sm) ? 'violation' : sm.intrusion ? 'gap' : 'clearance'}`;
      const stops = (moving ? ['yield_agent', 'stop_at_line', 'emergency_brake'] : ['hold_stop']).filter(has);
      preferred = stops.find(safe) ?? stops.find(clean) ?? (clean(motion) ? motion : null);
      if (preferred === 'emergency_brake' && has('yield_agent') && clean('yield_agent')) preferred = 'yield_agent';
      if (!preferred) {
        // Every behaviour collides: minimise the impact speed (harm), then prefer the latest contact.
        const harm = (id) => [score(id).collision_speed_mps ?? 0, -(score(id).collision_at_s ?? 99)];
        preferred = ids.filter((id) => id !== 'continue_current').sort((a, b) => harm(a)[0] - harm(b)[0] || harm(a)[1] - harm(b)[1])[0];
        reason += '+unavoidable';
      }
    }
  }
  if (!has(preferred)) preferred = has('hold_stop') ? 'hold_stop' : 'emergency_brake';
  const ref = score(preferred);
  const equivalent = ids.filter((id) => {
    if (id === preferred) return true;
    if (id === 'emergency_brake' && preferred !== 'emergency_brake') return false;
    const equivalentTypes = [['keep_route_cruise', 'route_turn', 'continue_current'], ['proceed_route', 'route_turn', 'continue_current'], ['stop_at_line', 'continue_current'], ['hold_stop', 'stop_at_line', 'yield_agent', 'emergency_brake']];
    if (!equivalentTypes.some((g) => g.includes(id) && g.includes(preferred))) return false;
    if (id === 'continue_current' && !(ex.maneuver && ex.maneuver.action_type !== 'yield_agent')) return false;
    const s = score(id);
    return !s.collision && legal(s) && Math.abs(s.progress_m - ref.progress_m) < 0.3 && Math.abs(s.final_speed_mps - ref.final_speed_mps) < 0.2;
  });
  const probabilities = Object.fromEntries(ids.map((id) => [id, equivalent.includes(id) ? 1 / equivalent.length : 0]));
  return { preferred, reason, probabilities, acceptable_set: equivalent, scores, rule: { must_stop: rule.mustStop, effective_must_stop: !!mustStop, reason: rule.reason, color: rule.color, stop_completed: rule.stopCompleted }, teacher_version: 'observable-rollout-teacher-1.1', horizon_s: HORIZON_STEPS * DT };
}
