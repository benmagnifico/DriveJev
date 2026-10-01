/**
 * DriveJev semantic executor and reference teacher.
 *
 *  - Offers the behaviours that are executable right now (see docs/simulator.md) and turns the chosen one into
 *    a speed target and lane keeping on JevPilot physics (fixed 0.05 s step).
 *  - Feedback (`stop_unreachable`, `turn_completed`) belongs to the current behaviour only.
 *  - `stop_at_line` is offered while the FRONT BUMPER has not passed the line and the car is not already resting at it.
 *  - Every moving behaviour keeps an ACC gap to the perceived lead vehicle; `yield_agent` stops short of a
 *    perceived path conflict; optional AEB caps speed when braking can prevent a predicted collision.
 *  - The student observation contains the ego clock (`stationary_s`), static junction control, the ego's own
 *    completed-stop memory and a perception summary — never signal colours, rules or other agents' plans.
 *  - `referenceTeacher` is a privileged world-rollout policy used as an upper bound.
 */
import { angle, clamp, dist, heading, nearestOnPath, pointAt, move } from '../third_party/jevpilot/src/math.js';
import { maneuverSteering, physics, routeSpeedLimit, stopLineDistance, BRAKING } from '../third_party/jevpilot/src/planning.js';
import { followingSpeed, leadVehicle, predictTrafficConflict, footprintClearance } from '../third_party/jevpilot/src/traffic-safety.js';
import { signalState } from '../third_party/jevpilot/src/world.js';

export const VERSION = 'drivejev-exec-1.0';
export const DT = 0.05;
export const ACTIONS = {
  keep_route_cruise: ['cruise', 'Follow the navigation route at the road speed limit, keeping a safe gap to the vehicle ahead.'],
  stop_at_line: ['line_stop', 'Approach and stop with the front bumper 0.5 m before the current stop line.'],
  hold_stop: ['zero', 'Remain stationary.'],
  proceed_route: ['smooth_start', 'Resume forward driving along the navigation route.'],
  route_turn: ['turn', 'Enter and follow the navigation turn through the current junction.'],
  yield_agent: ['yield_stop', 'Brake comfortably and stop short of the road user predicted to cross the path.'],
  continue_current: ['retain', 'Continue the current maneuver with its fixed target.'],
  emergency_brake: ['maximum_brake', 'Apply maximum braking immediately in the current lane.'],
};
const MOTION = ['keep_route_cruise', 'proceed_route', 'route_turn'];
const STOPPING = ['stop_at_line', 'yield_agent', 'emergency_brake'];
const r1 = (x) => (Number.isFinite(x) ? Math.round(x * 10) / 10 : null);
const r2 = (x) => (Number.isFinite(x) ? Math.round(x * 100) / 100 : null);

export function geometricLimit(sim, v) {
  let max = Math.min(sim.world.theme.limit, routeSpeedLimit(v));
  const dest = Math.max(0, v.route.length - v.s - 1.5);
  max = Math.min(max, Math.sqrt(2 * 4.5 * dest));
  for (let d = 2; d <= 42; d += 2) {
    const a = pointAt(v.route.points, v.s + d), b = pointAt(v.route.points, v.s + d + 2);
    const curvature = Math.abs(angle(heading(a, b) - (a.heading ?? v.heading))) / 2;
    const safe = Math.min(max, Math.sqrt(2.4 / Math.max(0.001, curvature)));
    max = Math.min(max, Math.sqrt(safe * safe + 2 * 4.5 * Math.max(0, d - 2)));
  }
  return max;
}

/** Speed target for one maneuver. `env` carries the lead/conflict used by this car. */
export function maneuverTarget(sim, v, m, env = {}) {
  const type = m?.action_type ?? 'keep_route_cruise';
  if (type === 'hold_stop' || type === 'emergency_brake') return { target: 0 };
  let target = geometricLimit(sim, v), unreachable = false;
  if (type === 'stop_at_line' && m.line) {
    const room = Math.max(0, stopLineDistance(v, m.line) - 0.5);
    unreachable = (v.speed * v.speed) / (2 * BRAKING) > room + 0.05;
    if (unreachable) return { target: 0, unreachable };
    // Constant 3.5 m/s^2 approach profile, settling within 0.15 m of the target point.
    target = Math.min(target, room < 0.15 ? 0 : Math.sqrt(2 * 3.5 * room));
  }
  if (type === 'yield_agent') {
    const c = env.conflict;
    if (c && Number.isFinite(c.distance_along_path_m)) {
      const room = Math.max(0, c.distance_along_path_m - (c.type === 'pedestrian' ? 3.5 : 3));
      const cap = Math.min(Math.sqrt(2 * 4.5 * room), 2.0 * room);
      target = Math.min(target, cap < 0.1 ? 0 : cap, Math.max(0, v.speed - 4 * DT));
    } else target = Math.min(target, Math.max(0, v.speed - 4 * DT));
  }
  // ACC: every moving behaviour keeps a following gap (JevPilot standstill gap + 0.8 s headway).
  if (env.lead) target = Math.min(target, accSpeed(v, env.lead));
  return { target, unreachable };
}

export function accSpeed(v, lead) {
  const other = lead.other, ego = Math.abs(v.speed);
  const leadSpeed = Math.max(0, (other.speed || 0) * Math.cos((other.heading || 0) - v.heading));
  const standstill = other.type === 'motorcycle' ? Math.min(3, 0.9 + ego * 0.15) : Math.min(2.5, 0.5 + ego * 0.12);
  const room = lead.gap - standstill - 0.8 * ego;
  if (room <= 0) return Math.max(0, Math.min(leadSpeed + 1.5 * room, followingSpeed(v, lead)));
  return Math.min(Math.sqrt(leadSpeed * leadSpeed + 2 * 4 * room), leadSpeed + 0.7 * room, followingSpeed(v, lead) + 0.0);
}

export class SemanticExecutor {
  constructor(sim, { aeb = false } = {}) {
    this.sim = sim; this.aeb = aeb; this.sequence = 0; this.stepId = 0; this.acceleration = 0; this.jerk = 0;
    this.maneuver = null; this.lastActions = []; this.frames = new Map(); this.policySource = 'local_executor';
    this.aebEvents = 0; this.aebSteps = 0; this.frontBumperEvents = []; this.frontLineLocks = new Map();
    this.frontRedViolations = 0; this.frontRequiredStopViolations = 0; this.stationarySince = sim.time; this.episodeId = 'episode';
    this.comfort = { steps: 0, accelAbs: 0, accelSq: 0, accelMax: 0, jerkAbs: 0, jerkSq: 0, jerkMax: 0, jerkSteps: 0, hardBrakeSteps: 0 };
    this.cachedEnv = null; this.cachedEnvStep = -1;
    sim.autopilot = true; sim.safety = false; sim.backgroundPlanning = true;
    sim.rerouteIfNeeded = () => {};
  }
  crossing() { return this.sim.crossingFor(this.sim.player); }
  line(c = this.crossing()) { return c ? { ...pointAt(this.sim.player.route.points, c.stopS), heading: c.approach, node_id: c.nodeId, stop_s: c.stopS } : null; }
  frontDistance(c = this.crossing()) { return c ? stopLineDistance(this.sim.player, this.line(c)) : null; }

  /** Live objects whose ids the last JevPilot sensor scan reported (range, 360°, building occlusion). */
  perceived() {
    const ids = new Set((this.sim.perception ?? []).map((o) => o.id));
    return { vehicles: this.sim.traffic.filter((o) => ids.has(o.id)), pedestrians: this.sim.pedestrians.filter((o) => ids.has(o.id)) };
  }
  /** Path conflict predicted if the ego continues its route at cruise speed (constant-velocity agents). */
  predictConflict(perceived = this.perceived(), cruiseTarget = null) {
    const v = this.sim.player;
    const ghost = { ...v, target: cruiseTarget ?? geometricLimit(this.sim, v), maneuver: { lane_offset_m: 0, lookahead_m: clamp(3 + Math.abs(v.speed) * 0.45, 3, 10) } };
    // Strip other vehicles' planned routes: the student-side predictor only extrapolates motion.
    const others = [...perceived.vehicles.map((o) => ({ ...o, route: undefined, s: undefined })), ...perceived.pedestrians];
    const c = predictTrafficConflict(ghost, others);
    return c && c.braking_reduces_risk ? c : null;
  }
  env() {
    const key = `${this.stepId}:${this.maneuver?.maneuver_id ?? ''}`;
    if (this.cachedEnvStep === key && this.cachedEnv) return this.cachedEnv;
    const p = this.perceived(), v = this.sim.player;
    const lead = leadVehicle(v, p.vehicles);
    const env = { perceived: p, lead: lead && lead.gap < 80 ? lead : null };
    if (this.maneuver?.action_type === 'yield_agent') env.conflict = this.predictConflict(p);
    this.cachedEnv = env; this.cachedEnvStep = key;
    return env;
  }
  stationaryS() { return Math.abs(this.sim.player.speed) < 0.2 ? this.sim.time - this.stationarySince : 0; }
  junctionSummary(c, p) {
    if (!c) return null;
    const sim = this.sim, node = sim.world.byId[c.nodeId], v = sim.player;
    if (!node || c.stopS - v.s > 70) return null;
    let inside = 0, approaching = 0, peds = 0, earlier = 0;
    const mine = v.stops[node.id]?.arrived ?? Infinity;
    for (const o of p.vehicles) {
      const d = dist(o, node);
      if (d < 11) { inside++; continue; }
      if (d > 45) continue;
      const toNode = heading(o, node), closing = Math.cos(angle((o.heading ?? 0) - toNode)) * (o.speed ?? 0);
      const crossing = Math.abs(Math.sin(angle((o.heading ?? 0) - c.approach))) > 0.7;
      if (crossing && closing > 0.5 && (d - 11) / closing < 4) approaching++;
      const stop = o.stops?.[node.id];
      if (node.control === 'stop' && stop && !stop.passed && sim.crossingFor(o)?.nodeId === node.id && stop.arrived < mine && d < 22) earlier++;
    }
    for (const o of p.pedestrians) if (o.crossing && o.walking && dist(o, node) < 14) peds++;
    return { vehicles_inside: inside, cross_approaching: approaching, pedestrians_crossing: peds, earlier_arrivals: node.control === 'stop' ? earlier : 0 };
  }
  candidates(env = this.env()) {
    const v = this.sim.player, c = this.crossing(), nav = this.sim.navigation(), moving = Math.abs(v.speed) >= 0.5;
    const front = c ? this.frontDistance(c) : null, rows = [];
    if (moving) rows.push(['keep_route_cruise', null]);
    const restingAtLine = !moving && front !== null && front <= 1.0;
    if (c && front > -0.3 && front < 85 && !restingAtLine) rows.push(['stop_at_line', c.nodeId]);
    if (!moving) { rows.push(['hold_stop', null]); rows.push(['proceed_route', null]); }
    if (c && ['left', 'right'].includes(nav.next_turn) && nav.turn_distance_m < 50) rows.push(['route_turn', c.nodeId]);
    if (moving && (env.hazard ?? this.lastHazard)) rows.push(['yield_agent', null]);
    if (this.maneuver && ['stop_at_line', 'route_turn', 'yield_agent'].includes(this.maneuver.action_type) && this.maneuver.status === 'active') rows.push(['continue_current', this.maneuver.target_id]);
    rows.push(['emergency_brake', null]);
    return rows.map(([action_type, target_id]) => ({ candidate_id: action_type, action_type, target_id, speed_profile: ACTIONS[action_type][0], description: ACTIONS[action_type][1] }));
  }
  observationId() { return `${this.episodeId}:${this.stepId}`; }
  candidateSignature(candidates) { return JSON.stringify(candidates.map((c) => [c.candidate_id, c.action_type, c.target_id ?? null, c.speed_profile])); }
  maneuverVersion() { return `${this.sequence}:${this.maneuver?.status ?? 'idle'}:${this.sim.routeVersion}`; }
  accept(result, { allowDelayed = false, maxAgeSeconds = 0.5 } = {}) {
    if (result.episode_id && result.episode_id !== this.episodeId) return { accepted: false, reason: 'cross_episode' };
    const original = this.frames.get(result.observation_id);
    if (!original) return { accepted: false, reason: 'unknown_observation' };
    const age = this.sim.time - original.sim_time;
    if (!allowDelayed && result.observation_id !== this.observationId()) return { accepted: false, reason: 'stale_observation' };
    if (age < 0 || age > maxAgeSeconds + 1e-8) return { accepted: false, reason: 'stale_observation', observation_age_s: age };
    const current = this.candidates(), chosen = current.find((c) => c.candidate_id === result.candidate_id);
    const offered = JSON.parse(original.candidate_signature).find((row) => row[0] === result.candidate_id);
    if (!offered) return { accepted: false, reason: 'chosen_not_offered_in_observation', observation_age_s: age };
    if (!chosen || JSON.stringify(offered) !== JSON.stringify([chosen.candidate_id, chosen.action_type, chosen.target_id ?? null, chosen.speed_profile])) return { accepted: false, reason: 'candidate_signature_changed', observation_age_s: age };
    if (original.maneuver_version !== this.maneuverVersion()) return { accepted: false, reason: 'maneuver_version_changed', observation_age_s: age };
    const accepted = this.select(chosen);
    if (accepted.accepted) this.policySource = result.model_version ?? 'unknown';
    return { ...accepted, observation_age_s: age };
  }
  select(c) {
    const type = typeof c === 'string' ? c : c.action_type, sim = this.sim, prior = this.maneuver;
    if (type === 'continue_current' && prior) return { accepted: true, continued: true, maneuver_id: prior.maneuver_id };
    const target = (typeof c === 'string' ? null : c.target_id) ?? (['stop_at_line', 'route_turn'].includes(type) ? this.crossing()?.nodeId ?? null : null);
    if (prior?.action_type === type && prior.target_id === target && prior.status === 'active') return { accepted: true, continued: true, maneuver_id: prior.maneuver_id };
    // Non-urgent hysteresis: a just-started braking behaviour is not cancelled by motion within 0.5 s.
    if (prior && MOTION.includes(type) && STOPPING.includes(prior.action_type) && prior.status === 'active' && sim.time - prior.started < 0.5 - 1e-8)
      return { accepted: false, reason: 'nonurgent_hysteresis', maneuver_id: prior.maneuver_id };
    const line = type === 'stop_at_line' ? this.line() : null;
    this.maneuver = { maneuver_id: `m${++this.sequence}`, action_type: type, target_id: target, started: sim.time, start_s: sim.player.s, status: 'active', line, feedback: [],
      exit_s: type === 'route_turn' ? (this.crossing()?.stopS ?? sim.player.s) + 28 : null };
    this.lastActions.push({ action_type: type, at: sim.time }); this.lastActions = this.lastActions.slice(-3);
    return { accepted: true, continued: false, maneuver_id: this.maneuver.maneuver_id };
  }
  step(count = 1) {
    for (let i = 0; i < count; i++) {
      const sim = this.sim, v = sim.player, oldSpeed = v.speed, previousAcceleration = this.acceleration;
      if (sim.crash || sim.complete) break;
      const originalCrossing = this.crossing(), originalLine = originalCrossing ? this.line(originalCrossing) : null;
      const oldFront = originalLine ? stopLineDistance(v, originalLine) : null, beforeRule = originalCrossing ? sim.rule(v) : null, oldTime = sim.time;
      v.maneuver = { lane_offset_m: 0, lookahead_m: clamp(3 + Math.abs(v.speed) * 0.45, 3, 10) };
      const env = this.env();
      const m = this.maneuver;
      const { target: raw, unreachable } = maneuverTarget(sim, v, m, env);
      if (unreachable && m && !m.feedback.includes('stop_unreachable')) m.feedback.push('stop_unreachable');
      let target = raw;
      // Optional AEB: only supervises moving behaviours; braking behaviours already slow down.
      const motionNow = !m || MOTION.includes(m.action_type) || (m.action_type === 'route_turn');
      const aebConflict = this.aeb && motionNow && raw > 0.1 && this.stepId % 2 === 0 ? this.predictConflict(env.perceived, raw) : this.aeb && this.aebActive ? this.aebConflict : null;
      this.aebConflict = aebConflict;
      if (aebConflict && Number.isFinite(aebConflict.max_speed_mps) && target > aebConflict.max_speed_mps + 0.01) {
        target = aebConflict.max_speed_mps; this.aebSteps++; if (!this.aebActive) this.aebEvents++; this.aebActive = true;
      } else this.aebActive = false;
      v.target = target; sim.autopilot = true; sim.safety = false; sim.step(DT); this.stepId++;
      if (Math.abs(v.speed) >= 0.2) this.stationarySince = sim.time;
      if (originalLine) {
        const newFront = stopLineDistance(v, originalLine), id = `${sim.routeVersion}:${originalCrossing.nodeId}:${originalCrossing.stopS}`;
        if (oldFront >= 0 && newFront <= 0 && newFront < oldFront && !this.frontLineLocks.has(id)) {
          const fraction = clamp(oldFront / (oldFront - newFront), 0, 1), t = oldTime + fraction * DT, node = sim.world.byId[originalCrossing.nodeId];
          const color = node.control === 'signal' ? signalState(node, t, originalCrossing.approach).color : node.control;
          const required = node.control === 'signal' ? color === 'red' || (color === 'amber' && beforeRule.mustStop) : node.control === 'stop' ? !beforeRule.stopCompleted : false;
          this.frontBumperEvents.push({ step_id: this.stepId, sim_time: t, node_id: originalCrossing.nodeId, control: node.control, color, required_stop: required, speed_mps: v.speed });
          this.frontLineLocks.set(id, true);
          if (color === 'red') this.frontRedViolations++;
          if (required) this.frontRequiredStopViolations++;
        }
      }
      this.acceleration = (v.speed - oldSpeed) / DT; this.jerk = (this.acceleration - previousAcceleration) / DT;
      const cf = this.comfort; cf.steps++; cf.accelAbs += Math.abs(this.acceleration); cf.accelSq += this.acceleration ** 2; cf.accelMax = Math.max(cf.accelMax, Math.abs(this.acceleration));
      if (this.acceleration < -5.5) cf.hardBrakeSteps++;
      if (cf.steps > 1) { cf.jerkSteps++; cf.jerkAbs += Math.abs(this.jerk); cf.jerkSq += this.jerk ** 2; cf.jerkMax = Math.max(cf.jerkMax, Math.abs(this.jerk)); }
      if (m?.status === 'active') {
        if (m.action_type === 'stop_at_line' && v.speed < 0.05 && stopLineDistance(v, m.line) <= 0.65) m.status = 'completed';
        if (m.action_type === 'route_turn' && v.s >= m.exit_s) { m.status = 'completed'; m.feedback.push('turn_completed'); }
        if (['yield_agent', 'emergency_brake'].includes(m.action_type) && v.speed < 0.05) m.status = 'completed';
      }
    }
    return this.sim.crash || this.sim.complete;
  }
  /** Student observation (whitelisted in drivejev/compiler.py) plus evaluation-only meta. */
  observe() {
    const sim = this.sim, v = sim.player, nav = sim.navigation(), c = this.crossing(), near = nearestOnPath(v, v.route.points), m = this.maneuver;
    const p = this.perceived();
    const hazard = this.predictConflict(p);
    this.lastHazard = hazard;
    const env = { ...this.env(), hazard };
    const lead = env.lead;
    const candidates = this.candidates(env);
    const context = { episode_id: this.episodeId, observation_id: this.observationId(), sim_time: sim.time, maneuver_version: this.maneuverVersion(), candidate_signature: this.candidateSignature(candidates) };
    this.frames.set(context.observation_id, context); if (this.frames.size > 256) this.frames.delete(this.frames.keys().next().value);
    const node = c ? sim.world.byId[c.nodeId] : null, rule = sim.rule(v);
    const side = hazard ? (hazard.right_m > 1.5 ? 'right' : hazard.right_m < -1.5 ? 'left' : 'ahead') : null;
    const student_obs = {
      ego: { speed_mps: r2(v.speed), acceleration_mps2: r2(this.acceleration), steering: r2(v.steering), route_offset_m: r2(near.distance), heading_error_deg: r1(nav.heading_error_deg), stationary_s: r1(this.stationaryS()) },
      nav: { remaining_m: r1(Math.max(0, v.route.length - v.s)), next_turn: nav.next_turn, turn_distance_m: r1(nav.turn_distance_m), speed_limit_mps: r1(Math.min(sim.world.theme.limit, routeSpeedLimit(v))),
        junction_control: node && node.control !== 'none' ? node.control : null, stop_line_ahead_m: c ? r1(this.frontDistance(c)) : null, stop_completed: c ? !!rule.stopCompleted : false },
      maneuver: { action_type: m?.action_type ?? 'none', target_id: m?.target_id ?? null, elapsed_s: r1(m ? sim.time - m.started : 0), progress_m: r1(m ? Math.max(0, v.s - m.start_s) : 0), status: m?.status ?? 'idle', feedback: m ? [...m.feedback] : [] },
      recent_actions: this.lastActions.map((a) => ({ action_type: a.action_type, age_s: r1(sim.time - a.at) })),
      traffic: {
        lead: lead ? { gap_m: r1(lead.gap), speed_mps: r1(lead.other.speed) } : null,
        hazard: hazard ? { type: hazard.type, in_s: r1(hazard.time_s), distance_m: r1(hazard.distance_along_path_m), side } : null,
        junction: this.junctionSummary(c, p),
      },
    };
    const cf = this.comfort;
    return {
      student_obs, candidates, observation_id: this.observationId(), observation_context: context,
      meta: {
        episode_id: this.episodeId, step_id: this.stepId, sim_time: sim.time, controller_version: VERSION, dt: DT, policy_source: this.policySource,
        teacher_state: { signal: c ? rule.color : null, required_stop: c ? rule.mustStop : false, reason: c ? rule.reason : null, control: node?.control ?? null },
        evaluation_truth: { collisions: sim.collisions, crash: sim.crash, violations: sim.violations, front_bumper_red_violations: this.frontRedViolations,
          front_bumper_required_stop_violations: this.frontRequiredStopViolations, front_bumper_crossing_events: this.frontBumperEvents, arrived: sim.complete,
          progress_m: sim.distance, route_length_m: v.route.length, route_s: v.s, aeb_events: this.aebEvents, aeb_steps: this.aebSteps, physics_steps: cf.steps,
          acceleration_rms_mps2: Math.sqrt(cf.accelSq / Math.max(1, cf.steps)), acceleration_abs_max_mps2: cf.accelMax, jerk_rms_mps3: Math.sqrt(cf.jerkSq / Math.max(1, cf.jerkSteps)), hard_brake_steps: cf.hardBrakeSteps,
          hazards: sim.director?.log ?? [] },
      },
    };
  }
}

export function placeAt(sim, s, speed = 0) {
  const p = pointAt(sim.player.route.points, s), ahead = pointAt(sim.player.route.points, s + 0.5);
  Object.assign(sim.player, { x: p.x, z: p.z, heading: heading(p, ahead), s, speed, steering: 0, wheelSteering: 0 });
}

// ------------------------------------------------------------------------------------
// Privileged teacher: reads the JevPilot rule (signal colour, stop-sign service, junction
// reservations) and rolls the whole world forward for each candidate on a dynamic-state
// clone (NPCs react to the ego as in the real world; unobservable future triggers frozen).
// ------------------------------------------------------------------------------------
const HORIZON_STEPS = 70; // 3.5 s

function rollout(ex, candidate) {
  const sim = ex.sim.clone(), v = sim.player;
  const m = candidate.action_type === 'continue_current' && ex.maneuver ? { ...ex.maneuver, feedback: [] }
    : { action_type: candidate.action_type, line: candidate.action_type === 'stop_at_line' ? ex.line() : null, exit_s: null };
  const startS = v.s, startViol = sim.violations, startSpeed = v.speed;
  let collision = null, maxDecel = 0, minGap = Infinity, prev = v.speed, stoppedAt = null;
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
    if (sim.crash) { collision = { ...sim.crash, at_s: (k + 1) * DT }; break; }
    // Required clearance grows with speed for pedestrians (0.05 m per m/s).
    if (k % 5 === 4 || k < 5) for (const o of dyn()) if (dist(o, v) < 9) minGap = Math.min(minGap, footprintClearance(v, o) - (o.type === 'pedestrian' ? 0.05 * Math.abs(v.speed) : 0));
    if (sim.complete) break;
  }
  // The ego is at fault if it was still moving into the contact.
  const atFault = collision ? collision.player_speed_mps > 0.4 : false;
  return { collision: atFault, collision_any: !!collision, collision_at_s: collision?.at_s ?? null, collision_type: collision?.type ?? null,
    violation: sim.violations > startViol, progress_m: v.s - startS, final_speed_mps: v.speed, max_decel_mps2: maxDecel, min_clearance_m: minGap, start_speed_mps: startSpeed };
}

export function referenceTeacher(ex, obs = ex.observe()) {
  const sim = ex.sim, v = sim.player, c = ex.crossing(), rule = sim.rule(v), ids = obs.candidates.map((x) => x.candidate_id);
  const has = (id) => ids.includes(id), nav = obs.student_obs.nav, front = c ? ex.frontDistance(c) : null;
  const moving = Math.abs(v.speed) >= 0.5;
  const scores = {};
  const score = (id) => (scores[id] ??= rollout(ex, obs.candidates.find((x) => x.candidate_id === id)));
  // Labels must be explainable from what the student can observe:
  //  - a green light's future phase is unknown to the student, so a rollout that only becomes a red-light
  //    violation later is not held against a moving behaviour while the light is still green;
  //  - JevPilot keeps a junction reservation until the holder is 17 m away; after a served stop the student
  //    only sees vehicles inside the junction box, so a holder that has already left the box (> 12 m) is ignored
  //    (the world rollout still has to be collision-free).
  const node = c ? sim.world.byId[c.nodeId] : null;
  const futureSignalUnknown = node?.control === 'signal' && rule.color === 'green';
  const legal = (s) => !s.violation || futureSignalUnknown;
  const safe = (id) => { const s = score(id); return !s.collision && legal(s) && s.min_clearance_m > 0.25; };
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
    // The stop-line profile ignores road users before the line; yield or brake if it is not safe.
    if (preferred === 'stop_at_line' && !safe('stop_at_line')) {
      preferred = has('yield_agent') && safe('yield_agent') ? 'yield_agent' : 'emergency_brake';
      reason += score('stop_at_line').collision ? '+collision' : '+clearance';
    }
  } else {
    const motion = !moving ? 'proceed_route' : has('route_turn') && ['left', 'right'].includes(nav.next_turn) && nav.turn_distance_m < 24 ? 'route_turn' : 'keep_route_cruise';
    if (safe(motion)) { preferred = motion; reason = 'clear'; }
    else {
      reason = `conflict:${score(motion).collision ? 'collision' : !legal(score(motion)) ? 'violation' : 'clearance'}`;
      const order = moving ? ['yield_agent', 'stop_at_line', 'emergency_brake'] : ['hold_stop'];
      preferred = order.filter(has).find((id) => !score(id).collision && legal(score(id))) ?? (moving ? 'emergency_brake' : 'hold_stop');
      // Prefer a comfortable stop when it is as safe as emergency braking.
      if (preferred === 'emergency_brake' && has('yield_agent') && !score('yield_agent').collision) preferred = 'yield_agent';
    }
  }
  if (!has(preferred)) preferred = has('hold_stop') ? 'hold_stop' : 'emergency_brake';
  // Soft target over behaviours whose rollouts are indistinguishable from the preferred one.
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
  return { preferred, reason, probabilities, acceptable_set: equivalent, scores, rule: { must_stop: rule.mustStop, effective_must_stop: !!mustStop, reason: rule.reason, color: rule.color, stop_completed: rule.stopCompleted }, teacher_version: 'world-rollout-teacher-1.0', horizon_s: HORIZON_STEPS * DT };
}
