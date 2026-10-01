/**
 * DriveJev headless harness (one page = one worker). Episodes run inside the page with fixed
 * 0.05 s physics, 4 Hz capture/decision slots and a history frame exactly 10 steps back.
 *   policy 'model'   : DriveJev behind /predict
 *   policy 'teacher' : privileged reference teacher (upper bound, no cameras)
 *   runEpisode       : batch closed loop (the world waits for each answer)
 *   runRealtime      : real-time closed loop (the world never waits)
 */
import * as THREE from 'three';
import { DriveScene } from '/third_party/jevpilot/src/scene.js';
import { renderProfile } from '/third_party/jevpilot/src/render-profile.js';
import { DriveWorld, WORLD_VERSION } from './world.mjs';
import { SemanticExecutor, referenceTeacher, DT, VERSION } from './executor.mjs';

// JevPilot render profile used for every model frame.
Object.assign(renderProfile, { pixelRatio: 1, antialias: false, shadowSize: 512, detailedFoliage: false, leafCards: 8 });
export const CAMERA = {
  wide: { version: 'ego-front-v4-vfov90-pitch10', width: 640, height: 384, vfov: 90, pitch: 10 },
  tele: { version: 'ego-tele-v7-vfov15-pitch2', width: 384, height: 224, vfov: 15, pitch: 2 },
};
const canvas = document.querySelector('#sensor');
const tele2d = Object.assign(document.createElement('canvas'), { width: CAMERA.tele.width, height: CAMERA.tele.height });
const teleCtx = tele2d.getContext('2d');
const wideCam = new THREE.PerspectiveCamera(CAMERA.wide.vfov, CAMERA.wide.width / CAMERA.wide.height, 0.1, 600);
const teleCam = new THREE.PerspectiveCamera(CAMERA.tele.vfov, CAMERA.tele.width / CAMERA.tele.height, 0.5, 900);
let sim, ex, scene, leaf, config;

function aim(cam, v, pitchDeg) {
  cam.position.set(v.x + Math.sin(v.heading) * 0.15, 1.6, v.z - Math.cos(v.heading) * 0.15);
  cam.lookAt(v.x + Math.sin(v.heading) * 30, 1.6 + 30 * Math.tan((pitchDeg * Math.PI) / 180), v.z - Math.cos(v.heading) * 30);
}

/** Render the wide and tele model views (ego car and annotations hidden). Returns base64 PNGs. */
export function capture() {
  scene.render(0, false);
  scene.vectors.group.visible = false; scene.vectorLayer.hidden = true; scene.sensorCone.visible = false; scene.destination.visible = false; scene.player.visible = false;
  const v = sim.player, r = scene.renderer;
  aim(wideCam, v, CAMERA.wide.pitch);
  r.setScissorTest(false); r.setViewport(0, 0, CAMERA.wide.width, CAMERA.wide.height);
  r.render(scene.scene, wideCam);
  const wide = canvas.toDataURL('image/png');
  aim(teleCam, v, CAMERA.tele.pitch);
  r.setViewport(0, 0, CAMERA.tele.width, CAMERA.tele.height); r.setScissor(0, 0, CAMERA.tele.width, CAMERA.tele.height); r.setScissorTest(true);
  r.render(scene.scene, teleCam);
  teleCtx.drawImage(canvas, 0, CAMERA.wide.height - CAMERA.tele.height, CAMERA.tele.width, CAMERA.tele.height, 0, 0, CAMERA.tele.width, CAMERA.tele.height);
  const tele = tele2d.toDataURL('image/png');
  r.setScissorTest(false); r.setViewport(0, 0, CAMERA.wide.width, CAMERA.wide.height);
  scene.player.visible = true;
  const b64 = (d) => d.slice(d.indexOf(',') + 1);
  return { wide: b64(wide), tele: b64(tele) };
}

export async function reset(cfg) {
  config = { seed: 320000, type: 'town', hazards: [], randomRoute: true, trafficScale: 1, empty: false, aeb: false, ...cfg };
  sim = new DriveWorld(config.seed, config.type, { randomRoute: config.randomRoute && config.type !== 'highway', hazards: config.hazards, trafficScale: config.trafficScale, empty: config.empty, hazardCooldown: config.hazardCooldown });
  ex = new SemanticExecutor(sim, { aeb: config.aeb });
  ex.episodeId = config.episode_id ?? `drivejev-${config.type}-${config.seed}`;
  if (leaf) { leaf.dispose(); leaf = null; }
  if (scene) { scene.sim = sim; scene.build(); } else scene = new DriveScene(canvas, sim, document.querySelector('#vector-layer'));
  // A fresh leaf material per world (three.js caches uniforms per material).
  const template = scene.vegetation.leafMaterial;
  leaf = template.clone(); scene.vegetation.addWind(leaf, 0.045); scene.vegetation.leafMaterial = leaf;
  scene.scene.traverse((o) => { if (o.material === template) o.material = leaf; else if (Array.isArray(o.material)) o.material = o.material.map((m) => (m === template ? leaf : m)); });
  scene.mode = 'hood'; scene.showSensors = false; scene.renderer.setPixelRatio(1); scene.renderer.shadowMap.enabled = false;
  await scene.ready;
  // Bootstrap until the first two-frame decision: cruise when moving, hold when stopped.
  ex.select(sim.player.speed >= 0.5 ? 'keep_route_cruise' : 'hold_stop'); ex.policySource = 'local_bootstrap';
  return { world: WORLD_VERSION, executor: VERSION, route_length_m: sim.player.route.length, crossings: sim.player.route.crossings.map((c) => ({ node: c.nodeId, control: sim.world.byId[c.nodeId].control })) };
}

const post = (url, body) => fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) });
const requestBody = (cfg, obs, history, shot) => ({
  arm: cfg.arm, schema_version: 'drivejev-online-1.0', episode_id: ex.episodeId, observation_id: obs.observation_id,
  student_obs: obs.student_obs, candidates: obs.candidates,
  images: [{ camera: 'front', relative_time: -0.5, png_base64: history.wide }, { camera: 'front', relative_time: 0, png_base64: shot.wide }, { camera: 'front_tele', relative_time: 0, png_base64: shot.tele }],
});
const stuckStep = () => Math.abs(sim.player.speed) < 0.2 && ex.stepId > 20 && !sim.rule(sim.player).mustStop && !(ex.env().lead && ex.env().lead.gap < 4);

function summary(info, extra) {
  const truth = ex.observe().meta.evaluation_truth;
  return { ...info, config, sim_time_s: sim.time, arrived: sim.complete, crash: sim.crash, collisions: sim.collisions, violations: sim.violations,
    front_red_violations: ex.frontRedViolations, front_required_stop_violations: ex.frontRequiredStopViolations,
    route_completion: Math.min(1, sim.player.s / Math.max(1, sim.player.route.length)), aeb_events: ex.aebEvents, aeb_steps: ex.aebSteps,
    hard_brake_steps: truth.hard_brake_steps, jerk_rms_mps3: truth.jerk_rms_mps3, hazards: sim.director?.log ?? [], front_events: ex.frontBumperEvents, ...extra };
}

export async function runEpisode(cfg) {
  const info = await reset(cfg);
  const policy = cfg.policy ?? 'model', maxTime = cfg.max_time_s ?? 120, frames = [], latencies = [], actions = {}, rejections = {}, trace = [];
  let decisions = 0, accepted = 0, stuck = 0, speedSum = 0;
  const t0 = performance.now();
  while (sim.time < maxTime && !sim.crash && !sim.complete) {
    if (ex.stepId % 5 === 0) {
      const obs = ex.observe();
      const shot = policy === 'model' ? capture() : null;
      if (shot) { frames.push({ step: ex.stepId, ...shot }); if (frames.length > 4) frames.shift(); }
      const history = frames.find((f) => f.step === ex.stepId - 10);
      if (ex.stepId >= 10 && (history || policy !== 'model')) {
        let result;
        if (policy === 'teacher') {
          const t = referenceTeacher(ex, obs);
          result = { candidate_id: t.preferred, probabilities: t.probabilities, observation_id: obs.observation_id, model_version: 'reference-teacher' };
        } else {
          const started = performance.now(), response = await post(cfg.model_url ?? '/predict', requestBody(cfg, obs, history, shot));
          result = await response.json();
          if (!response.ok) throw Error(`model ${response.status}: ${JSON.stringify(result).slice(0, 300)}`);
          latencies.push(performance.now() - started);
        }
        const res = ex.accept({ ...result, episode_id: ex.episodeId }, { allowDelayed: true, maxAgeSeconds: 0.5 });
        if (res.accepted) accepted++; else rejections[res.reason] = (rejections[res.reason] ?? 0) + 1;
        decisions++; actions[result.candidate_id] = (actions[result.candidate_id] ?? 0) + 1;
        if (cfg.trace) trace.push({ t: Math.round(sim.time * 100) / 100, v: Math.round(sim.player.speed * 100) / 100, chosen: result.candidate_id, executing: ex.maneuver?.action_type, p: result.probabilities });
      }
    }
    if (stuckStep()) stuck += DT;
    speedSum += Math.abs(sim.player.speed) * DT;
    if (ex.step(1)) break;
    if (ex.stepId % 200 === 0) await new Promise((r) => setTimeout(r, 0));
  }
  latencies.sort((a, b) => a - b);
  return summary(info, { policy, mode: 'batch', wall_s: (performance.now() - t0) / 1000, mean_speed_mps: speedSum / Math.max(sim.time, 1e-6), stuck_s: stuck,
    decisions, accepted, rejections, action_counts: actions, latency_ms_p50: latencies[Math.floor(latencies.length / 2)] ?? null, trace: cfg.trace ? trace : undefined });
}

/** Real time: physics follows the wall clock at 20 Hz; at most one request in flight; answers older than 0.5 s are dropped. */
export async function runRealtime(cfg) {
  const info = await reset(cfg);
  const maxTime = cfg.max_time_s ?? 120, frames = [], latencies = [], rejections = {};
  let inflight = false, requests = 0, accepted = 0, stuck = 0, speedSum = 0;
  frames.push({ step: ex.stepId, ...capture() }); // history frame for the first request at 0.5 s
  ex.observe();
  const start = performance.now();
  const dispatch = async (obs, history, shot) => {
    inflight = true; requests++;
    const t = performance.now();
    try {
      const response = await post(cfg.model_url ?? '/predict', requestBody(cfg, obs, history, shot));
      const result = await response.json();
      latencies.push(performance.now() - t);
      if (response.ok) { const res = ex.accept({ ...result, episode_id: ex.episodeId }, { allowDelayed: true, maxAgeSeconds: 0.5 }); if (res.accepted) accepted++; else rejections[res.reason] = (rejections[res.reason] ?? 0) + 1; }
    } finally { inflight = false; }
  };
  while (sim.time < maxTime && !sim.crash && !sim.complete) {
    const due = Math.floor((performance.now() - start) / (DT * 1000));
    let stepped = false;
    while (ex.stepId < due && !sim.crash && !sim.complete && sim.time < maxTime) {
      if (stuckStep()) stuck += DT;
      speedSum += Math.abs(sim.player.speed) * DT;
      ex.step(1); stepped = true;
      if (ex.stepId % 5 === 0) {
        const obs = ex.observe(), shot = capture();
        frames.push({ step: ex.stepId, ...shot }); if (frames.length > 4) frames.shift();
        const history = frames.find((f) => f.step === ex.stepId - 10);
        if (history && !inflight) void dispatch(obs, history, shot);
      }
    }
    await new Promise((r) => setTimeout(r, stepped ? 0 : 4));
  }
  while (inflight) await new Promise((r) => setTimeout(r, 10));
  const wall = (performance.now() - start) / 1000;
  latencies.sort((a, b) => a - b);
  return summary(info, { policy: 'model', mode: 'realtime', wall_s: wall, realtime_factor: sim.time / wall, mean_speed_mps: speedSum / Math.max(sim.time, 1e-6), stuck_s: stuck,
    requests, accepted, applied_hz: accepted / Math.max(sim.time, 1e-6), rejections, latency_ms_p50: latencies[Math.floor(latencies.length / 2)] ?? null,
    latency_ms_p95: latencies[Math.floor(latencies.length * 0.95)] ?? null });
}

window.drivejev = { reset, runEpisode, runRealtime, capture, get sim() { return sim; }, get executor() { return ex; } };
window.drivejevReady = true;
