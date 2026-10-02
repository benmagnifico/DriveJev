// Semantic pilots (DriveJev and the privileged reference teacher) inside the JevPilot world.
//
// The model sees exactly what the closed-loop harness (simulator/harness.mjs) gives it: two wide
// 640x384 front frames (t-0.5 s, t) and one 384x224 tele frame (t), rendered from a second,
// training-profile scene, plus the whitelisted student observation and the offered behaviours.
// The semantic executor (simulator/interaction_executor.mjs: the 1.0 executor with vehicle-occlusion
// perception) turns the accepted behaviour into steering and speed. Loop: fixed 0.05 s physics, capture every 5 steps (4 Hz), history frame exactly 10 steps
// back, at most one request in flight, an answer is applied only if its observation is <= 0.5 s old.
import * as THREE from "three";
import { DriveScene } from "/third_party/jevpilot/src/scene.js";
import { renderProfile } from "/third_party/jevpilot/src/render-profile.js";
import { clamp, nearestOnPath } from "/third_party/jevpilot/src/math.js";
import { maneuverSteering, physics } from "/third_party/jevpilot/src/planning.js";
import { maneuverTarget, DT } from "/simulator/executor.mjs";
import { InteractionExecutor, observableTeacher } from "/simulator/interaction_executor.mjs";

// simulator/harness.mjs CAMERA and render profile.
export const CAMERA = Object.freeze({
  wide: Object.freeze({ version: "ego-front-v4-vfov90-pitch10", width: 640, height: 384, vfov: 90, pitch: 10 }),
  tele: Object.freeze({ version: "ego-tele-v7-vfov15-pitch2", width: 384, height: 224, vfov: 15, pitch: 2 }),
});
export const TRAINING_PROFILE = Object.freeze({ pixelRatio: 1, antialias: false, shadowSize: 512, detailedFoliage: false, leafCards: 8 });
export const STEPS_PER_CAPTURE = 5; // 20 Hz physics, 4 Hz camera + decisions
export const HISTORY_STEPS = 10; // t-0.5 s
export const MAX_OBSERVATION_AGE_S = 0.5;
export const STALL_MS = 2500;
export const SCHEMA_VERSION = "drivejev-online-1.1";

export const PILOTS = [
  {
    id: "drivejev",
    kind: "drivejev",
    arm: "default",
    name: "DriveJev (ours)",
    short: "DriveJev",
    tag: "Ours",
    icon: "scan-eye",
    detail: "Frozen Qwen-Drive-1.0 4B + decision head. Wide front t, t−0.5 s + 15° tele view + student state → one of up to 8 behaviours.",
  },
  {
    id: "teacher",
    kind: "teacher",
    name: "Reference teacher (privileged)",
    short: "Teacher",
    tag: "Privileged",
    icon: "route",
    detail: "Reads the true signal / stop rule and rolls the world forward 4.5 s for each offered behaviour, with only the road users its sensors have seen. No camera.",
  },
  {
    id: "kev",
    kind: "jev",
    backend: "kev",
    name: "Kev (local)",
    short: "Kev",
    tag: "Open Jev",
    icon: "server",
    detail: "A local server speaking the Jev /v1/systemone API (e.g. the open Kev reconstruction), with JevPilot's structured state and sampled paths.",
  },
  {
    id: "jev",
    kind: "jev",
    backend: "jev",
    name: "Jev (cloud)",
    short: "Jev",
    tag: "OpenRouter",
    icon: "cloud",
    detail: "TypeSafe Jev via the OpenRouter Decisions API, with JevPilot's structured state and sampled path candidates (no camera).",
  },
];
export const isModelPilot = (pilot) => pilot?.kind === "drivejev";
export const isTeacherPilot = (pilot) => pilot?.kind === "teacher";
export const isSemantic = (pilot) => isModelPilot(pilot) || isTeacherPilot(pilot);

export function withProfile(profile, build) {
  const saved = { ...renderProfile };
  Object.assign(renderProfile, profile);
  try {
    return build();
  } finally {
    Object.assign(renderProfile, saved);
  }
}
/** harness.mjs aim(): 1.6 m high, 0.15 m ahead of the reference point, pitched up `pitchDeg`. */
function aim(camera, v, pitchDeg) {
  camera.position.set(v.x + Math.sin(v.heading) * 0.15, 1.6, v.z - Math.cos(v.heading) * 0.15);
  camera.lookAt(v.x + Math.sin(v.heading) * 30, 1.6 + 30 * Math.tan((pitchDeg * Math.PI) / 180), v.z - Math.cos(v.heading) * 30);
}
const b64 = (dataUrl) => dataUrl.slice(dataUrl.indexOf(",") + 1);

/** Second DriveScene bound to the same world, built under the training render profile, rendered only by the model cameras. */
export class SensorRig {
  constructor(host) {
    this.canvas = host.querySelector("canvas");
    this.layer = host.querySelector("div");
    const { wide, tele } = CAMERA;
    this.wideCamera = new THREE.PerspectiveCamera(wide.vfov, wide.width / wide.height, 0.1, 600);
    this.teleCamera = new THREE.PerspectiveCamera(tele.vfov, tele.width / tele.height, 0.5, 900);
    this.tele2d = Object.assign(document.createElement("canvas"), { width: tele.width, height: tele.height });
    this.teleCtx = this.tele2d.getContext("2d");
    this.scene = null;
    this.world = null;
    this.ready = Promise.resolve();
  }
  bind(sim) {
    if (this.scene && this.world === sim.world && this.scene.sim === sim) return this.ready;
    this.world = sim.world;
    withProfile(TRAINING_PROFILE, () => {
      if (this.scene) {
        this.scene.sim = sim;
        this.scene.build();
      } else this.scene = new DriveScene(this.canvas, sim, this.layer);
    });
    const scene = this.scene;
    // As in the harness: every world gets a fresh leaf material with its own wind uniform.
    const template = scene.vegetation.leafMaterial;
    const leaves = template.clone();
    scene.vegetation.addWind(leaves, 0.045);
    scene.vegetation.leafMaterial = leaves;
    scene.scene.traverse((object) => {
      if (object.material === template) object.material = leaves;
      else if (Array.isArray(object.material)) object.material = object.material.map((m) => (m === template ? leaves : m));
    });
    scene.mode = "hood";
    scene.showSensors = false;
    scene.renderer.setPixelRatio(1);
    scene.renderer.shadowMap.enabled = false;
    this.ready = scene.ready.then(async () => {
      scene.render(0, false);
      aim(this.wideCamera, sim.player, CAMERA.wide.pitch);
      aim(this.teleCamera, sim.player, CAMERA.tele.pitch);
      await scene.renderer.compileAsync(scene.scene, this.wideCamera);
      await scene.renderer.compileAsync(scene.scene, this.teleCamera);
    });
    return this.ready;
  }
  /**
   * Same sequence as simulator/harness.mjs capture(): update, hide the ego car and annotations,
   * render the wide view to PNG, render the tele view into the bottom-left 384x224 sub-viewport,
   * crop it through a 2D canvas, restore the viewport.
   */
  capture(sim) {
    const scene = this.scene,
      r = scene.renderer,
      { wide: W, tele: T } = CAMERA,
      started = performance.now();
    scene.render(0, false);
    scene.vectors.group.visible = false;
    scene.vectorLayer.hidden = true;
    scene.sensorCone.visible = false;
    scene.destination.visible = false;
    scene.player.visible = false;
    const v = sim.player;
    aim(this.wideCamera, v, W.pitch);
    r.setScissorTest(false);
    r.setViewport(0, 0, W.width, W.height);
    r.render(scene.scene, this.wideCamera);
    const wide = this.canvas.toDataURL("image/png");
    aim(this.teleCamera, v, T.pitch);
    r.setViewport(0, 0, T.width, T.height);
    r.setScissor(0, 0, T.width, T.height);
    r.setScissorTest(true);
    r.render(scene.scene, this.teleCamera);
    this.teleCtx.drawImage(this.canvas, 0, W.height - T.height, T.width, T.height, 0, 0, T.width, T.height);
    const tele = this.tele2d.toDataURL("image/png");
    r.setScissorTest(false);
    r.setViewport(0, 0, W.width, W.height);
    scene.player.visible = true;
    return { dataUrl: wide, png_base64: b64(wide), tele: { dataUrl: tele, png_base64: b64(tele) }, capture_ms: performance.now() - started };
  }
}

/**
 * InteractionExecutor driven by the display loop. advance() is always exactly one fixed 0.05 s
 * step: student_obs numbers are prompt text, so variable steps would feed the model values it
 * never saw. The stall watchdog only zeroes the ego target while no answer has arrived.
 */
export class DemoExecutor extends InteractionExecutor {
  constructor(sim, { aeb, obsSchema = "1.1" }) {
    super(sim, { aeb, obsSchema });
    this.stalled = false;
    // Harness episodes start at t = 0; the demo attaches to a running world, so the ego
    // stationary clock starts at engagement.
    this.stationarySince = sim.time;
  }
  advance() {
    if (!this.stalled) return this.step(1);
    const sim = this.sim,
      base = Object.getPrototypeOf(sim).step;
    sim.step = function (dt) {
      this.player.target = 0;
      return base.call(this, dt);
    };
    try {
      return this.step(1);
    } finally {
      delete sim.step;
    }
  }
}

/**
 * Display-only 3 s previews of every offered behaviour with the executor's speed law on a cloned
 * ego car (lead and conflict distances extrapolated at constant speed). Never fed back.
 */
export function controllerPreviews(executor, candidates) {
  const sim = executor.sim,
    ego = sim.player,
    env = executor.env(),
    hazard = executor.lastHazard ?? null;
  const leadSpeed = env.lead ? Math.max(0, env.lead.other.speed || 0) : 0;
  return candidates.map((candidate) => {
    const car = { ...ego };
    const maneuver =
      candidate.action_type === "continue_current" && executor.maneuver
        ? { ...executor.maneuver }
        : { action_type: candidate.action_type, line: candidate.action_type === "stop_at_line" ? executor.line() : null };
    const points = [{ x: car.x, z: car.z }];
    for (let i = 0; i < 60; i++) {
      const travelled = car.s - ego.s;
      const lead = env.lead ? { ...env.lead, gap: env.lead.gap - travelled + leadSpeed * i * DT } : null;
      const conflict =
        maneuver.action_type === "yield_agent" && hazard ? { ...hazard, distance_along_path_m: hazard.distance_along_path_m - travelled } : null;
      const steering = { lane_offset_m: 0, lookahead_m: clamp(3 + Math.abs(car.speed) * 0.45, 3, 10) };
      const { target } = maneuverTarget(sim, car, maneuver, { lead, conflict });
      physics(car, maneuverSteering(car, steering), target, DT);
      car.s = nearestOnPath(car, car.route.points).s;
      points.push({ x: car.x, z: car.z });
    }
    return { candidate_id: candidate.candidate_id, action_type: candidate.action_type, description: candidate.description, points, progress_m: car.s - ego.s, final_speed_mps: car.speed };
  });
}

const round = (x, k = 2) => (Number.isFinite(x) ? Math.round(x * 10 ** k) / 10 ** k : x);
const strip = (images) => images.map(({ png_base64, ...rest }) => ({ ...rest, png_bytes: Math.round((png_base64.length * 3) / 4) }));
function summarizeScores(scores) {
  return Object.fromEntries(
    Object.entries(scores ?? {}).map(([id, s]) => [
      id,
      { collision: s.collision, violation: s.violation, progress_m: round(s.progress_m), final_speed_mps: round(s.final_speed_mps), min_clearance_m: Number.isFinite(s.min_clearance_m) ? round(s.min_clearance_m) : null },
    ]),
  );
}

export class SemanticRuntime {
  constructor({ sensor, onDecision, onFrame, onPreview, onError, onStatus }) {
    Object.assign(this, { sensor, onDecision, onFrame, onPreview, onError, onStatus });
    this.executor = null;
    this.generation = 0;
    this.stats = this.freshStats();
  }
  freshStats() {
    return { requests: 0, completed: 0, failed: 0, accepted: 0, rejected: 0, skipped_busy: 0, latencies: [], backbone: [], teacher_ms: [], accept_times: [], rejections: {} };
  }
  get active() {
    return !!this.executor;
  }
  // obsSchema: the student observation the served head was trained on ("1.0" heads see the 1.0 fields only).
  async engage(sim, pilot, { aeb, obsSchema }) {
    this.disengage();
    this.sim = sim;
    this.pilot = pilot;
    const generation = ++this.generation;
    if (isModelPilot(pilot)) {
      this.onStatus?.("Preparing the model cameras…");
      await this.sensor.bind(sim);
      if (generation !== this.generation) return false;
    }
    this.executor = new DemoExecutor(sim, { aeb, obsSchema });
    this.executor.episodeId = `demo-${pilot.id}-${sim.world.type}-${sim.world.seed}-${Date.now().toString(36)}`;
    // Bootstrap until the first decision, as in the harness: cruise when moving, hold when stopped.
    this.executor.select(sim.player.speed >= 0.5 ? "keep_route_cruise" : "hold_stop");
    this.executor.policySource = "local_bootstrap";
    this.frames = [];
    this.inflight = null;
    this.lastDecision = null;
    this.lastRequest = null;
    this.lastResponseWall = performance.now();
    this.errors = 0;
    this.stats = this.freshStats();
    this.onGrid(); // step 0: the history frame for the first request at step 10
    return true;
  }
  disengage() {
    this.generation++;
    if (!this.executor) return;
    const sim = this.sim;
    delete sim.rerouteIfNeeded; // the executor pins the route; restore JevPilot rerouting
    sim.safety = true;
    sim.player.maneuver = null;
    sim.player.target = 0;
    this.executor = null;
    this.inflight = null;
  }
  setAeb(on) {
    if (this.executor) this.executor.aeb = on;
  }
  /** One fixed 0.05 s physics step; observation/capture/decision slot every 5 steps. */
  stepFixed() {
    const executor = this.executor;
    if (!executor || this.sim.paused || this.sim.crash) return;
    executor.advance();
    if (executor.stepId % STEPS_PER_CAPTURE === 0) this.onGrid();
  }
  /** Stall watchdog (called every ~25 ms of wall time). */
  tick() {
    const executor = this.executor;
    if (executor && isModelPilot(this.pilot) && this.lastRequest) executor.stalled = performance.now() - this.lastResponseWall > STALL_MS;
  }
  // Observe only on the 4 Hz grid: observe() also refreshes the hazard memory that decides
  // whether yield_agent is offered, so it must not run between slots.
  onGrid() {
    const executor = this.executor,
      sim = this.sim;
    if (!executor || sim.crash || sim.complete) return;
    const row = executor.observe();
    if (isTeacherPilot(this.pilot)) return this.teach(row);
    const shot = this.sensor.capture(sim);
    const frame = { step_id: executor.stepId, sim_time: sim.time, observation_id: row.observation_id, ...shot };
    this.frames.push(frame);
    if (this.frames.length > 12) this.frames.shift();
    const history = this.frames.find((f) => f.step_id === frame.step_id - HISTORY_STEPS) ?? null;
    this.onFrame?.(frame, history);
    this.preview(row, this.lastDecision);
    if (!history) return; // still inside the 0.5 s bootstrap
    if (this.inflight) {
      this.stats.skipped_busy++; // the world keeps moving; only the newest frame is ever sent
      return;
    }
    this.request(row, frame, history);
  }
  preview(row, decision) {
    const executor = this.executor;
    this.onPreview?.({
      origin: { ...this.sim.player },
      previews: controllerPreviews(executor, row.candidates),
      probabilities: decision?.probabilities ?? {},
      executing: executor.maneuver?.action_type ?? null,
      chosen: decision?.candidate_id ?? null,
    });
  }
  teach(row) {
    const executor = this.executor,
      started = performance.now();
    const t = observableTeacher(executor, row);
    const elapsed = performance.now() - started;
    this.stats.teacher_ms.push(elapsed);
    if (this.stats.teacher_ms.length > 40) this.stats.teacher_ms.shift();
    const result = {
      candidate_id: t.preferred,
      probabilities: t.probabilities,
      model_version: "observable-teacher",
      episode_id: executor.episodeId,
      observation_id: row.observation_id,
      confidence: { value: Math.max(...Object.values(t.probabilities)) },
      teacher: { preferred: t.preferred, reason: t.reason, acceptable_set: t.acceptable_set, rule: t.rule, version: t.teacher_version, horizon_s: t.horizon_s, scores: summarizeScores(t.scores) },
      timings_ms: { policy_total: elapsed },
    };
    this.finish(row, result, null, null, started);
  }
  async request(row, frame, history) {
    const executor = this.executor,
      generation = this.generation,
      started = performance.now();
    // simulator/harness.mjs request body.
    const body = {
      arm: this.pilot.arm,
      schema_version: SCHEMA_VERSION,
      episode_id: executor.episodeId,
      observation_id: row.observation_id,
      student_obs: row.student_obs,
      candidates: row.candidates,
      images: [
        { camera: "front", relative_time: -0.5, png_base64: history.png_base64 },
        { camera: "front", relative_time: 0, png_base64: frame.png_base64 },
        { camera: "front_tele", relative_time: 0, png_base64: frame.tele.png_base64 },
      ],
    };
    this.lastRequest = { ...body, images: strip(body.images), sent_sim_time: this.sim.time };
    this.inflight = row.observation_id;
    this.stats.requests++;
    try {
      const response = await fetch("/api/drivejev/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(4000),
      });
      const result = await response.json();
      if (generation !== this.generation) return;
      if (!response.ok) throw Error(result.error || `DriveJev HTTP ${response.status}`);
      this.errors = 0;
      this.lastResponseWall = performance.now();
      this.stats.completed++;
      this.finish(row, result, frame, history, started);
    } catch (error) {
      if (generation !== this.generation) return;
      this.stats.failed++;
      this.errors++;
      this.onError?.(error, this.errors);
    } finally {
      if (generation === this.generation) this.inflight = null;
    }
  }
  finish(row, result, frame, history, started) {
    const executor = this.executor;
    if (!executor) return;
    const wall = performance.now() - started;
    const acceptance = executor.accept({ ...result, episode_id: executor.episodeId }, { allowDelayed: true, maxAgeSeconds: MAX_OBSERVATION_AGE_S });
    const keep = (list, value) => {
      list.push(value);
      if (list.length > 40) list.shift();
    };
    if (acceptance.accepted) {
      this.stats.accepted++;
      executor.stalled = false;
      keep(this.stats.accept_times, this.sim.time);
    } else {
      this.stats.rejected++;
      this.stats.rejections[acceptance.reason] = (this.stats.rejections[acceptance.reason] ?? 0) + 1;
    }
    keep(this.stats.latencies, wall);
    if (result.timings_ms?.backbone !== undefined) keep(this.stats.backbone, result.timings_ms.backbone);
    const decision = {
      ...result,
      pilot: this.pilot,
      candidates: row.candidates,
      student_obs: row.student_obs,
      teacher_state: isTeacherPilot(this.pilot) ? row.meta.teacher_state : undefined,
      captured_sim_time: row.meta.sim_time,
      applied_sim_time: this.sim.time,
      latency_ms: wall,
      acceptance,
      executing: executor.maneuver?.action_type ?? null,
      frame,
      history,
    };
    this.lastDecision = decision;
    this.preview(row, decision); // the observation's own candidates (no extra observe())
    this.onDecision?.(decision);
  }
  /** Accepted decisions per second over the last few seconds of simulation time. */
  appliedHz() {
    const times = this.stats.accept_times;
    if (times.length < 2) return 0;
    const span = this.sim.time - times[0];
    return span > 0 ? (times.length - 1) / span : 0;
  }
  violations() {
    const ex = this.executor;
    return ex ? { red: ex.frontRedViolations, required: ex.frontRequiredStopViolations, events: ex.frontBumperEvents } : null;
  }
  aeb() {
    const ex = this.executor;
    return ex ? { on: ex.aeb, events: ex.aebEvents, steps: ex.aebSteps, active: !!ex.aebActive, conflict: ex.aebConflict?.type ?? null } : null;
  }
}
