// DriveJev interactive demo: JevPilot's playground (third_party/jevpilot/src/main.js) with
// switchable pilots on the DriveJev world (simulator/world.mjs: seeded random routes, optional
// scripted hazards). Semantic pilots (DriveJev, the reference teacher) drive through the
// semantic executor (simulator/executor.mjs); Jev-style pilots (cloud Jev, local Kev) use
// JevPilot's own structured state and sampled path candidates.
import {
  createIcons,
  Braces,
  RotateCw,
  Video,
  Pause,
  Play,
  Map,
  Maximize,
  Minimize,
  Sparkles,
  ArrowUp,
  CornerUpLeft,
  CornerUpRight,
  Flag,
  ArrowUpRight,
  X,
  Copy,
  Download,
  RotateCcw,
  Plus,
  Minus,
  Grip,
  ChevronDown,
  ChevronsUpDown,
  Cloud,
  Server,
  PanelRight,
  ScanEye,
  Route,
} from "/demo/vendor/lucide.js";
import { HAZARD_KINDS } from "/simulator/world.mjs";
import { InteractionWorld, ALL_KINDS } from "/simulator/interactions.mjs";
import { BackgroundPlanner } from "/third_party/jevpilot/src/background-planner.js";
import { DriveScene } from "/third_party/jevpilot/src/scene.js";
import { MinimapControls } from "/third_party/jevpilot/src/minimap-controls.js";
import { Tooltips } from "/third_party/jevpilot/src/tooltips.js";
import { TouchControls } from "/third_party/jevpilot/src/touch-controls.js";
import { showLoading, hideLoading, loadingFailed, nextPaint } from "/third_party/jevpilot/src/loading-screen.js";
import { prepareJevRequest, expandJevAnswers, decisionInterval } from "/third_party/jevpilot/src/jev-request.js";
import { THEMES, signalState } from "/third_party/jevpilot/src/world.js";
import { candidateName, decisionControls, decisionSelection, stopLineDistance } from "/third_party/jevpilot/src/planning.js";
import { dist, nearestOnPath, pointAt } from "/third_party/jevpilot/src/math.js";
import { DT } from "/simulator/executor.mjs";
import { PILOTS, SensorRig, SemanticRuntime, isSemantic, isModelPilot, isTeacherPilot } from "./runtime.js";
import { SemanticVectors, actionStyle } from "./semantic-vectors.js";
import { DecisionPanel } from "./decision-panel.js";

const icons = {
  Braces, RotateCw, Video, Pause, Play, Map, Maximize, Minimize, Sparkles, ArrowUp, CornerUpLeft, CornerUpRight, Flag, ArrowUpRight, X, Copy, Download, RotateCcw, Plus, Minus, Grip, ChevronDown, ChevronsUpDown, Cloud, Server, PanelRight, ScanEye, Route,
};
const icon = (name) => `<i data-lucide="${name}"></i>`,
  $ = (id) => document.getElementById(id);

// Random demo worlds never use the training / validation / test seed range (300000–329999).
function randomSeed() {
  for (;;) {
    const seed = Math.floor(Math.random() * 999999);
    if (seed < 300000 || seed > 329999) return seed;
  }
}
const params = new URLSearchParams(location.search),
  aliases = { suburb: "town", country: "highway" },
  requested = aliases[params.get("world")] || params.get("world") || "town";
let worldKind = THEMES[requested] ? requested : "town";
// World options: `route=default` keeps JevPilot's default route (otherwise a seeded random route
// over the grid; the interstate always uses its default route), `hazards=1` enables the scripted
// hazard and interaction scenarios (`hazards=classic`: only the three base hazards; `hazards=storm`:
// all of them back to back).
const routeMode = params.get("route") === "default" ? "default" : "random";
// `hazards=oncoming,green_runner,...` restricts the scenarios to the listed kinds.
const HAZARD_MODES = { 1: "all", on: "all", true: "all", all: "all", classic: "classic", storm: "storm" };
const listedKinds = (params.get("hazards") ?? "").split(",").filter((k) => ALL_KINDS.includes(k));
let hazardMode = listedKinds.length ? "list" : HAZARD_MODES[params.get("hazards")] ?? "all";
let hazardsOn = params.get("hazards") in HAZARD_MODES || listedKinds.length > 0;
// The interstate only has neighbour-lane cut-ins (junction scenarios cannot be placed there); the base
// junction runner is left out on the interstate as before.
const hazardKinds = (kind) =>
  hazardMode === "classic" ? HAZARD_KINDS.filter((k) => kind !== "highway" || k !== "cross_runner")
  : hazardMode === "list" ? listedKinds
  : kind === "highway" ? ["cut_in"] : ALL_KINDS;
/** Every world is a fresh InteractionWorld: its options and hazard agents exist only from construction. */
function makeWorld(seed, kind) {
  return new InteractionWorld(seed, kind, {
    randomRoute: routeMode === "random" && kind !== "highway",
    hazards: hazardsOn ? hazardKinds(kind) : [],
    hazardCooldown: hazardMode === "storm" ? 4 : hazardMode === "classic" ? 10 : 9,
  });
}
let sim = makeWorld(Number(params.get("seed")) || randomSeed(), worldKind);
// `arm=<name>` selects a decision head when the model service serves several (default "default").
if (params.get("arm")) PILOTS.find((p) => p.kind === "drivejev").arm = params.get("arm");
let pilot = PILOTS.find((p) => p.id === params.get("pilot")) ?? PILOTS[0];
let backends = { drivejev: { status: "unknown" }, jev: { configured: false }, kev: { available: false } };
// One switch: the executor's AEB for semantic pilots, JevPilot's traffic safety envelope for Jev/Kev.
let safetyBrake = params.get("aeb") !== "off" && params.get("safety") !== "off";
let loading = true,
  lastMapDraw = 0;
showLoading("Loading car and scenery…");
let busy = false,
  generation = 0,
  lastDecision = null,
  lastInput = null,
  lastContext = null,
  lastApplied = 0,
  nextDecision = 0,
  nextContextCheck = 0,
  errors = 0,
  inspectorTab = "request",
  inspectFrozen = false,
  uiTime = 0,
  lastNow = performance.now(),
  toastTimer,
  crashHandled = false,
  engaging = false,
  semanticDecision = null;
const keys = new Set(),
  tally = { cost: 0, calls: 0, constrained_steps: 0, request_bytes: 0, input: 0, output: 0, latencies: [], intervals: [] };

$("app").innerHTML = `
<main class="drive-area" aria-label="3D driving simulator"><canvas id="world-canvas" aria-label="Interactive three-dimensional driving world"></canvas><div id="vector-labels" aria-label="Jev motion vector probabilities"></div><div id="semantic-labels" aria-label="DriveJev behaviour probabilities"></div></main>
<header class="topbar glass"><a href="/" class="brand" aria-label="DriveJev demo"><img class="brand-mark" src="/demo/brand.svg" alt=""/><b>DriveJev</b><span class="brand-sub">Demo</span></a><div class="world-picker"><select id="world-select" aria-label="Driving scene"><option value="town">Small town</option><option value="city">Skyline City</option><option value="highway">Interstate 08</option></select><button id="new-world" title="New world" aria-label="New world">${icon("rotate-cw")}</button></div></header>
<aside id="decision-panel" class="decision-panel glass" aria-label="Live decision process"></aside>
<div class="navigation-hud"><div class="navigation-card glass"><span id="turn-icon">${icon("arrow-up")}</span><div><strong id="next-maneuver">Continue straight</strong><span id="turn-distance"></span></div><span class="nav-divider"></span><span id="remaining"></span><button id="map-toggle" aria-label="Toggle route map" aria-pressed="true" title="Hide route map">${icon("map")}</button></div>
<div id="minimap" class="minimap glass"><div class="minimap-toolbar" role="toolbar" aria-label="Minimap controls"><button id="map-drag" aria-label="Move minimap" title="Move minimap · drag or use arrow keys">${icon("grip")}</button><div><button id="map-zoom-out" aria-label="Zoom out" title="Zoom out">${icon("minus")}</button><button id="map-zoom-in" aria-label="Zoom in" title="Zoom in">${icon("plus")}</button><button id="map-reset" aria-label="Reset minimap" title="Reset map position, zoom and following">${icon("rotate-ccw")}</button></div></div><canvas id="map-canvas" width="380" height="310" aria-label="Route map. Drag to pan, scroll to zoom, double-click to follow the car."></canvas></div></div>
<div id="paused-overlay" hidden><div class="glass"><span>${icon("pause")} Paused</span><button id="resume" class="primary">Resume driving</button></div></div>
<div id="arrival" class="arrival glass" hidden><span class="arrival-mark">${icon("flag")}</span><span class="eyebrow" id="arrival-eyebrow">DESTINATION REACHED</span><h1 id="arrival-title">You made it.</h1><p id="arrival-summary"></p><button id="next-trip" class="primary">Next drive ${icon("arrow-up-right")}</button><button id="keep-driving" class="subtle">Keep exploring</button></div>
<div id="pilot-menu" class="pilot-menu glass" role="menu" aria-label="Choose pilot" hidden></div>
<div class="bottom-hud"><div class="driver-dock glass"><div class="speed-cluster"><div title="Current speed"><strong id="speed">0</strong><span>km/h</span></div><span class="speed-limit" title="Speed limit"><small>LIMIT</small><b id="speed-limit">50</b></span></div><span class="dock-divider"></span><div class="pilot-actions"><div class="pilot-combo"><button id="autopilot" class="pilot-button" role="switch" aria-checked="false" aria-label="Engage pilot" title="Engage · J">${icon("sparkles")}<span id="pilot-label">Engage</span><kbd>J</kbd></button><button id="pilot-picker" class="pilot-picker" aria-haspopup="menu" aria-expanded="false" aria-label="Choose pilot" title="Choose pilot · M">${icon("chevrons-up-down")}</button></div><button id="candidates-toggle" class="candidate-button" aria-label="Show candidates" aria-pressed="false" title="Show candidates"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false"><path d="M12 20V3m-3 3 3-3 3 3M12 20C12 14 7 12 3 8m0 3V8h3M12 20c0-6 5-8 9-12m-3 0h3v3"/><circle cx="12" cy="21" r="1" fill="currentColor" stroke="none"/></svg></button></div><div id="decision-status"><span id="pilot-state">Free play</span><span id="context-message">WASD to drive · Space to brake</span><span class="cost-total"><span id="cost-label">Pilot</span> <strong id="cost"></strong></span></div><span class="dock-divider"></span><div class="dock-tools" role="group" aria-label="View and driving controls"><button id="camera" title="Change camera · C" aria-label="Change camera">${icon("video")}<span id="camera-name">Chase</span></button><button id="panel-toggle" aria-label="Toggle decision panel" aria-pressed="true" title="Decision panel · I">${icon("panel-right")}</button><button id="scene-json" aria-label="Inspect live JSON" title="Inspect live JSON">${icon("braces")}</button><button id="fullscreen" aria-label="Enter fullscreen" title="Fullscreen">${icon("maximize")}</button><span class="divider"></span><button id="pause" aria-label="Pause simulation" title="Pause · P">${icon("pause")}</button></div></div></div>
<dialog id="crash-dialog" aria-labelledby="crash-title" aria-describedby="crash-description"><span class="crash-symbol">${icon("x")}</span><span class="eyebrow">DRIVE ENDED</span><h1 id="crash-title">Game over.</h1><p id="crash-description"></p><div class="crash-stats"><div><strong id="crash-speed"></strong><span>km/h at impact</span></div><div><strong id="crash-distance"></strong><span>meters driven</span></div></div><button id="retry-drive" class="primary">${icon("rotate-ccw")} Restart drive</button><button id="crash-new-world" class="secondary">Try a new world ${icon("arrow-up-right")}</button></dialog>
<div id="toast" role="status" hidden></div>
<div id="sensor-rig" aria-hidden="true"><canvas width="640" height="384"></canvas><div></div></div>
<dialog id="json-dialog"><div class="json-header"><div>${icon("braces")}<strong>Under the hood</strong><span id="json-live">LIVE · 4 Hz</span></div><button id="close-json" aria-label="Close JSON inspector">${icon("x")}</button></div><div class="json-toolbar"><div class="json-tabs"><button data-tab="request" class="active">Model input</button><button data-tab="sensor">Perception</button><button data-tab="world">Full world</button><button data-tab="decision">Response</button></div><div class="json-actions"><button id="freeze-json">Freeze</button><button id="copy-json" aria-label="Copy displayed JSON">${icon("copy")} <span id="copy-json-label" aria-live="polite">Copy</span></button><button id="download-json">${icon("download")} Download</button></div></div><p id="json-description"></p><pre id="json-content"></pre></dialog>
<dialog id="help-dialog"><button id="close-help" class="dialog-close" aria-label="Close help">${icon("x")}</button><span class="eyebrow">DRIVEJEV DEMO</span><h2>Take the wheel, or hand it over.</h2><div class="help-keys"><span><kbd>W / ↑</kbd> Accelerate</span><span><kbd>S / ↓</kbd> Brake / reverse</span><span><kbd>A / D</kbd> Steer</span><span><kbd>SPACE</kbd> Brake</span><span><kbd>J</kbd> Engage pilot</span><span><kbd>M</kbd> Choose pilot</span><span><kbd>1–${PILOTS.length}</kbd> Pilot shortcut</span><span><kbd>C</kbd> Camera</span><span><kbd>I</kbd> Decision panel</span><span><kbd>P</kbd> Pause</span></div><p>Drag the scene to orbit in Chase or Bird's eye; drag to look around in Driver view; scroll to zoom; double-click to recenter.</p><p><b>DriveJev</b> sees three frames rendered exactly like its training data — the wide 640×384 front camera now and 0.5 s ago plus a 384×224 tele view (15°) for distant lights and signs — and a small state record (ego, navigation, junction control, lead vehicle, predicted path conflict), then chooses one of up to eight behaviours; the semantic executor steers and sets the speed (with a following gap). The panel shows the actual model input, the probabilities, whether the choice was applied, and a 20 s timeline. <b>Jev</b> and <b>Kev</b> receive JevPilot's structured state and sampled path candidates instead (no camera).</p><p>The pilot menu switches the AEB (collision-mitigation braking from perceived objects; for Jev/Kev, JevPilot's traffic safety envelope) — it never stops for red lights — and the scripted hazard and interaction scenarios (jaywalker, red-light/stop runner, hard-braking lead; oncoming platoons during a left turn, 4-way-stop contention, a red-light runner just after your green, a pedestrian hidden behind a parked car, cut-ins, pedestrians at the turn). Probabilities are model scores, not calibrated success rates. Interactive drives are demonstrations, not the evaluation protocol.</p></dialog>`;
$("app").insertAdjacentHTML(
  "beforeend",
  `<div id="touch-controls" class="touch-controls" role="group" aria-label="Touch driving controls" hidden>
  <div class="touch-steering">
    <div class="touch-stick" role="group" aria-label="Driving joystick: drag up to accelerate, down to brake or reverse, left or right to steer">
      <span class="stick-up" aria-hidden="true">↑</span><span class="stick-down" aria-hidden="true">↓</span>
      <span class="stick-left" aria-hidden="true">‹</span><span class="stick-right" aria-hidden="true">›</span>
      <span class="touch-knob" aria-hidden="true"></span>
    </div>
    <span class="touch-hint">Drag to drive</span>
  </div>
  <button class="touch-brake" aria-label="Hold to brake"><span aria-hidden="true">Ⅱ</span>Brake</button>
</div>`,
);
createIcons({ icons });

const scene = new DriveScene($("world-canvas"), sim, $("vector-labels")),
  map = $("map-canvas").getContext("2d");
const semanticVectors = new SemanticVectors(scene.scene, $("semantic-labels"));
const minimap = new MinimapControls($("minimap"), sim, drawMap);
const tooltips = new Tooltips();
const panel = new DecisionPanel($("decision-panel"));
const sensor = new SensorRig($("sensor-rig"));
const touch = new TouchControls(
  $("touch-controls"),
  () => !loading && !sim.autopilot && !sim.paused && !sim.crash && !document.hidden && !document.querySelector("dialog[open]") && (sim.freeExplore || !sim.complete),
);
const dockObserver = new ResizeObserver(([entry]) => {
  const height = entry.borderBoxSize?.[0]?.blockSize ?? entry.target.offsetHeight;
  document.documentElement.style.setProperty("--dock-height", `${height}px`);
});
dockObserver.observe(document.querySelector(".driver-dock"));
for (const element of document.querySelectorAll(".bottom-hud button, .bottom-hud [title], .minimap button, #map-toggle"))
  tooltips.set(element, element.title || element.getAttribute("aria-label"));

const runtime = new SemanticRuntime({
  sensor,
  onStatus: (text) => ($("context-message").textContent = text),
  onFrame: (frame, history) => panel.showFrame(frame, history),
  onPreview: (preview) => semanticVectors.set(preview),
  onDecision: (decision) => {
    semanticDecision = decision;
    panel.semanticDecision(decision, sim.time);
    panel.showObs(decision.student_obs);
    if (isModelPilot(decision.pilot)) panel.setLive("live", `${runtime.appliedHz().toFixed(1)} Hz`);
    else {
      panel.setLive("live", "4 Hz");
      const truth = decision.teacher_state ?? {},
        t = decision.teacher ?? {},
        label = (id) => actionStyle(decision.candidates.find((c) => c.candidate_id === id)?.action_type ?? id).label;
      panel.showState([
        ["control (truth)", truth.control === "signal" ? `signal ${truth.signal ?? ""}` : truth.control === "stop" ? "stop sign" : truth.control ?? "no junction ahead"],
        ["rule (truth)", `${t.rule?.must_stop ? "must stop" : "may go"}${t.rule?.reason ? ` · ${t.rule.reason}` : ""}`],
        ["preferred", `${label(t.preferred)} · ${t.reason ?? ""} · ${Math.round(decision.timings_ms?.policy_total ?? 0)} ms`],
        ["acceptable", (t.acceptable_set ?? []).map(label).join(", ")],
      ]);
    }
  },
  onError: (error, count) => {
    panel.error(error.message);
    panel.setLive("error", "Error");
    toast(error.message, "error");
    if (count >= 3) {
      setPilot(false);
      toast(`${pilot.name} paused after three failed requests. Check the model service, then engage again.`, "error");
    }
  },
});

const planner = new BackgroundPlanner();
sim.backgroundPlanning = true;
/** Swap in a freshly constructed InteractionWorld (resetWorld); every holder of the old one is rebound. */
function installWorld(next) {
  sim = next;
  sim.backgroundPlanning = true;
  sim.requestReroute = requestReroute;
  scene.sim = sim;
  minimap.simulation = sim;
}
let planningJob = null,
  rerouting = false,
  previewError = false;
async function refreshPlan() {
  if (planningJob) return planningJob;
  const token = generation,
    version = sim.routeVersion;
  const job = planner
    .run("plan", sim)
    .then((result) => {
      if (token !== generation || version !== sim.routeVersion || sim.crash) return null;
      sim.lastPlan = result.plan;
      sim.lastDecisionState = result.state;
      sim.routeChoices = result.routeChoices;
      sim.routeChoicesOrigin = result.routeChoicesOrigin;
      sim.nextRouteChoices = result.nextRouteChoices;
      return result;
    })
    .finally(() => {
      if (planningJob === job) planningJob = null;
    });
  planningJob = job;
  return job;
}
function requestPreview() {
  if (sim.autopilot || sim.crash || isSemantic(pilot)) return;
  refreshPlan()
    .then((result) => {
      if (result && !sim.autopilot) scene.vectors.setCandidates(result.plan);
    })
    .catch((error) => {
      if (!previewError) {
        previewError = true;
        toast(error.message);
      }
    });
}
sim.requestReroute = requestReroute;
async function requestReroute() {
  if (rerouting || !sim.routeChoiceNeeded()) return;
  rerouting = true;
  const token = generation,
    version = sim.routeVersion;
  try {
    const next = await planner.run("reroute", sim);
    if (!next || token !== generation || version !== sim.routeVersion || sim.crash || !sim.routeChoiceNeeded()) return;
    if (next.route.ids.join(",") === sim.player.route.ids.join(",")) return;
    sim.installRoute({ ...next, progress: nearestOnPath(sim.player, next.route.points).s });
  } catch (error) {
    toast(error.message);
  } finally {
    rerouting = false;
  }
}

function toast(text, type = "info") {
  $("toast").textContent = text;
  $("toast").classList.toggle("error", type === "error");
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("toast").hidden = true), 4200);
}

/* ---------- pilots ---------- */
function pilotAvailability(p) {
  if (p.kind === "teacher") return { ok: true, note: "local · world rollouts" };
  if (p.kind === "drivejev") {
    const info = backends.drivejev,
      arms = Object.keys(info?.arms ?? {});
    if (info?.status === "ready" && info.arms?.[p.arm]) {
      const version = String(info.arms[p.arm].sha256 ?? "");
      return { ok: true, note: `model service · head “${p.arm}”${/^[0-9a-f]{64}$/.test(version) ? ` ${version.slice(0, 8)}` : ""}` };
    }
    if (info?.status === "ready") return { ok: false, note: `head “${p.arm}” not served (available: ${arms.join(", ") || "none"}; use ?arm=)` };
    return { ok: false, note: info?.status === "unknown" ? "checking model service…" : info?.status === "error" ? "model service error" : "model service offline" };
  }
  if (p.backend === "jev") return backends.jev?.configured ? { ok: true, note: backends.jev.model } : { ok: false, note: "not configured (set OPENROUTER_API_KEY on the server)" };
  return backends.kev?.available ? { ok: true, note: backends.kev.url } : { ok: false, note: `not running (${backends.kev?.url ?? "--kev-url"})` };
}
function renderPilotMenu() {
  $("pilot-menu").innerHTML = `<div class="pilot-menu-head"><strong>Pilot</strong><span>Switch any time · J to engage</span></div>${PILOTS.map((p, i) => {
    const a = pilotAvailability(p);
    return `<button role="menuitemradio" aria-checked="${p.id === pilot.id}" data-pilot="${p.id}" class="pilot-option${a.ok ? "" : " unavailable"}"><span class="pilot-option-icon">${icon(p.icon)}</span><span class="pilot-option-text"><span class="pilot-option-name">${p.name}<em data-tag="${p.tag.toLowerCase().replace(/\s+/g, "-")}">${p.tag}</em></span><span class="pilot-option-detail">${p.detail}</span><span class="pilot-option-status"><i class="${a.ok ? "ok" : "off"}"></i>${a.note}</span></span><kbd>${i + 1}</kbd></button>`;
  }).join("")}<label class="pilot-menu-safety"><input type="checkbox" id="safety-brake" ${safetyBrake ? "checked" : ""}/> ${
    isSemantic(pilot)
      ? "AEB (collision-mitigation braking) <small>(perceived objects · never brakes for red lights)</small>"
      : "Safety brake for traffic collisions <small>(never brakes for red lights)</small>"
  }</label><label class="pilot-menu-safety"><input type="checkbox" id="hazard-toggle" ${hazardsOn ? "checked" : ""}/> Hazard &amp; interaction scenarios <small>(${hazardMode === "classic" ? "jaywalker · junction runner · braking lead" : "oncoming platoons · 4-way stops · red-light runners · hidden pedestrians · cut-ins · …"} · resets the world)</small></label>`;
  createIcons({ icons, root: $("pilot-menu") });
  for (const button of $("pilot-menu").querySelectorAll("[data-pilot]"))
    button.onclick = () => {
      selectPilot(PILOTS.find((p) => p.id === button.dataset.pilot));
      closePilotMenu();
    };
  $("safety-brake").onchange = (e) => {
    safetyBrake = e.target.checked;
    runtime.setAeb(safetyBrake);
    if (sim.autopilot && !isSemantic(pilot)) sim.safety = safetyBrake;
  };
  $("hazard-toggle").onchange = (e) => setHazards(e.target.checked);
}
function openPilotMenu() {
  renderPilotMenu();
  $("pilot-menu").hidden = false;
  $("pilot-picker").setAttribute("aria-expanded", "true");
  $("pilot-menu").querySelector('[aria-checked="true"]')?.focus();
}
function closePilotMenu() {
  $("pilot-menu").hidden = true;
  $("pilot-picker").setAttribute("aria-expanded", "false");
}
$("pilot-picker").onclick = () => ($("pilot-menu").hidden ? openPilotMenu() : closePilotMenu());
document.addEventListener("pointerdown", (e) => {
  if (!$("pilot-menu").hidden && !e.target.closest("#pilot-menu, #pilot-picker")) closePilotMenu();
});
function selectPilot(next) {
  if (!next || next.id === pilot.id) return;
  const wasOn = sim.autopilot;
  if (wasOn) setPilot(false);
  pilot = next;
  const url = new URL(location.href);
  url.searchParams.set("pilot", pilot.id);
  history.replaceState(null, "", url);
  panel.setPilot(pilot, icon(pilot.icon));
  createIcons({ icons, root: $("decision-panel") });
  scene.vectors.clear();
  semanticVectors.clear();
  syncPilot();
  if (isModelPilot(pilot) && !loading) sensor.bind(sim).catch((error) => console.warn("Sensor camera unavailable", error));
  toast(`${pilot.name} selected${pilotAvailability(pilot).ok ? "" : ` · ${pilotAvailability(pilot).note}`}`);
  if (wasOn) setPilot(true);
}

function setHazards(on) {
  if (on === hazardsOn) return;
  hazardsOn = on;
  const url = new URL(location.href);
  if (hazardsOn) url.searchParams.set("hazards", hazardMode === "list" ? listedKinds.join(",") : hazardMode === "all" ? "1" : hazardMode);
  else url.searchParams.delete("hazards");
  history.replaceState(null, "", url);
  closePilotMenu();
  Promise.resolve(resetWorld(sim.world.seed, worldKind)).then(() =>
    toast(hazardsOn ? (hazardMode === "classic" ? "Hazard scenarios on · a jaywalker, junction runner or braking lead at most every 10–18 s" : hazardMode === "list" ? `Scenarios on · ${listedKinds.join(", ")}` : "Hazard & interaction scenarios on · nine kinds, one at most every 9–16 s") : "Hazard scenarios off"),
  );
}
function refreshWorld() {
  $("world-select").value = worldKind;
  $("speed-limit").textContent = Math.round(sim.world.theme.limit * 3.6);
  $("arrival").hidden = true;
}
function syncPilot() {
  const on = sim.autopilot;
  $("autopilot").setAttribute("aria-checked", String(on));
  $("pilot-label").textContent = engaging ? `Starting ${pilot.short}…` : on ? `${pilot.short} engaged` : `Engage ${pilot.short}`;
  tooltips.set($("autopilot"), `${on ? "Disengage" : "Engage"} ${pilot.name} · J`);
  $("autopilot").disabled = !!sim.crash;
  $("autopilot").dataset.kind = pilot.kind;
  document.body.classList.toggle("piloting", on);
  touch.sync();
}

async function setPilot(on) {
  if (loading || engaging) return;
  touch.reset();
  closePilotMenu();
  if (on) {
    const availability = pilotAvailability(pilot);
    if (!availability.ok) {
      toast(`${pilot.name} is not available · ${availability.note}`, "error");
      openPilotMenu();
      return;
    }
  }
  if (sim.crash || (on && sim.complete)) return;
  runtime.disengage();
  generation++;
  lastApplied = 0;
  nextDecision = 0;
  errors = 0;
  sim.player.target = 0;
  sim.player.steering = 0;
  sim.player.steeringProgress = 0;
  sim.player.maneuver = null;
  scene.vectors.clear();
  semanticVectors.clear();
  if (!on) {
    sim.autopilot = false;
    panel.setLive("idle", "Idle");
    syncPilot();
    return;
  }
  sim.freeExplore = false;
  if (isSemantic(pilot)) {
    engaging = true;
    syncPilot();
    const token = generation;
    let ready = false;
    try {
      ready = await runtime.engage(sim, pilot, { aeb: safetyBrake, obsSchema: backends.drivejev?.arms?.[pilot.arm]?.observation_schema ?? "1.1" });
    } catch (error) {
      toast(error.message, "error");
    }
    engaging = false;
    if (!ready || token !== generation) {
      syncPilot();
      return;
    }
    semanticDecision = null;
    panel.reset();
    panel.setLive("live", "Bootstrap");
  } else {
    sim.safety = safetyBrake;
    panel.reset();
    panel.setLive("live", "Live");
  }
  sim.autopilot = true;
  syncPilot();
}

async function resetWorld(seed = sim.world.seed, kind = worldKind) {
  if (loading) return;
  loading = true;
  runtime.disengage();
  sim.autopilot = false;
  touch.reset();
  showLoading("Building your next drive…");
  generation++;
  crashHandled = false;
  $("crash-dialog").close();
  document.body.classList.remove("crashed");
  keys.clear();
  await nextPaint();
  try {
    worldKind = kind;
    installWorld(makeWorld(seed, kind));
    const url = new URL(location.href);
    url.searchParams.set("world", kind);
    url.searchParams.set("seed", seed);
    history.replaceState(null, "", url);
    minimap.resetView();
    planner.reset();
    previewError = false;
    lastApplied = 0;
    lastDecision = null;
    lastInput = null;
    lastContext = null;
    semanticDecision = null;
    scene.build();
    semanticVectors.attach(scene.scene);
    semanticVectors.clear();
    scene.vectors.showCandidates = showCandidates && !isSemantic(pilot);
    $("next-trip").innerHTML = `Next drive ${icon("arrow-up-right")}`;
    $("keep-driving").textContent = "Keep exploring";
    $("arrival-eyebrow").textContent = "DESTINATION REACHED";
    $("arrival-title").textContent = "You made it.";
    delete $("arrival").dataset.result;
    refreshWorld();
    syncPilot();
    panel.reset();
    $("paused-overlay").hidden = true;
    $("pause").innerHTML = icon("pause");
    $("pause").setAttribute("aria-label", "Pause simulation");
    tooltips.set($("pause"), "Pause simulation · P");
    createIcons({ icons });
    await finishLoading();
  } catch (error) {
    loadingFailed(error);
  }
}
async function finishLoading() {
  showLoading("Loading car and scenery…");
  await scene.ready;
  showLoading("Preparing the road…");
  await nextPaint();
  await scene.prepare();
  if (isModelPilot(pilot)) {
    showLoading("Preparing the DriveJev cameras…");
    await sensor.bind(sim);
  }
  await document.fonts.ready;
  await nextPaint();
  lastNow = performance.now();
  loading = false;
  hideLoading();
  touch.sync();
  updateUI();
  drawMap();
}
function changeCamera() {
  const modes = ["chase", "hood", "map"];
  scene.mode = modes[(modes.indexOf(scene.mode) + 1) % 3];
  scene.snap = true;
  $("camera-name").textContent = { chase: "Chase", hood: "Driver", map: "Bird’s eye" }[scene.mode];
  tooltips.set($("camera"), `Change camera · ${$("camera-name").textContent} · C`);
}
function togglePause() {
  if (sim.crash || loading) return;
  touch.reset();
  keys.clear();
  sim.paused = !sim.paused;
  touch.sync();
  generation++;
  lastApplied = 0;
  nextDecision = 0;
  $("paused-overlay").hidden = !sim.paused;
  $("pause").innerHTML = icon(sim.paused ? "play" : "pause");
  tooltips.set($("pause"), `${sim.paused ? "Resume" : "Pause"} simulation · P`);
  $("pause").setAttribute("aria-label", sim.paused ? "Resume simulation" : "Pause simulation");
  createIcons({ icons });
}
function togglePanel(force) {
  const hidden = force ?? !$("decision-panel").hidden;
  $("decision-panel").hidden = hidden;
  $("panel-toggle").setAttribute("aria-pressed", String(!hidden));
}
$("autopilot").onclick = () => setPilot(!sim.autopilot);
let showCandidates = false,
  candidatePreviewAt = 0;
$("candidates-toggle").onclick = () => {
  showCandidates = !showCandidates;
  scene.vectors.showCandidates = showCandidates && !isSemantic(pilot);
  semanticVectors.showCandidates = showCandidates;
  $("candidates-toggle").setAttribute("aria-pressed", String(showCandidates));
  const label = `${showCandidates ? "Hide" : "Show"} candidates`;
  $("candidates-toggle").setAttribute("aria-label", label);
  tooltips.set($("candidates-toggle"), label);
  if (showCandidates && !scene.vectors.plan) requestPreview();
};
$("new-world").onclick = () => resetWorld(randomSeed());
$("world-select").onchange = (e) => resetWorld(randomSeed(), e.target.value);
$("retry-drive").onclick = () => {
  resetWorld();
  $("autopilot").focus();
};
$("crash-new-world").onclick = () => resetWorld(randomSeed());
$("crash-dialog").addEventListener("cancel", (e) => e.preventDefault());
$("camera").onclick = changeCamera;
$("pause").onclick = togglePause;
$("resume").onclick = togglePause;
$("panel-toggle").onclick = () => togglePanel();
$("next-trip").onclick = () => resetWorld(randomSeed());
$("keep-driving").onclick = () => {
  sim.complete = false;
  sim.freeExplore = true;
  $("arrival").hidden = true;
};
$("map-toggle").onclick = () => {
  $("minimap").hidden = !$("minimap").hidden;
  $("map-toggle").setAttribute("aria-pressed", String(!$("minimap").hidden));
  tooltips.set($("map-toggle"), `${$("minimap").hidden ? "Show" : "Hide"} route map`);
  minimap.constrainPosition();
  drawMap();
};
$("fullscreen").onclick = async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.documentElement.requestFullscreen();
  } catch {
    toast("Use your browser’s fullscreen shortcut.");
  }
};
document.addEventListener("fullscreenchange", () => {
  $("fullscreen").innerHTML = icon(document.fullscreenElement ? "minimize" : "maximize");
  $("fullscreen").setAttribute("aria-label", document.fullscreenElement ? "Exit fullscreen" : "Enter fullscreen");
  tooltips.set($("fullscreen"), $("fullscreen").getAttribute("aria-label"));
  createIcons({ icons });
});
window.addEventListener("keydown", (e) => {
  if (sim.crash || loading) return;
  if ($("json-dialog").open || $("help-dialog").open) return;
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  if (e.target.closest("button") && ["Space", "Enter"].includes(e.code)) return;
  const driving = ["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"];
  if (driving.includes(e.code)) {
    e.preventDefault();
    keys.add(e.code);
    if (sim.autopilot) setPilot(false);
  }
  if (e.repeat) return;
  if (e.code === "Escape") closePilotMenu();
  if (e.code === "KeyJ") setPilot(!sim.autopilot);
  if (e.code === "KeyM") $("pilot-menu").hidden ? openPilotMenu() : closePilotMenu();
  if (e.code === "KeyI") togglePanel();
  if (/^Digit[1-9]$/.test(e.code) && PILOTS[Number(e.code.slice(5)) - 1]) selectPilot(PILOTS[Number(e.code.slice(5)) - 1]);
  if (e.code === "KeyC") changeCamera();
  if (e.code === "KeyP") togglePause();
  if (e.key === "?") {
    e.preventDefault();
    touch.reset();
    $("help-dialog").showModal();
  }
});
window.addEventListener("keyup", (e) => keys.delete(e.code));
window.addEventListener("blur", () => keys.clear());
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    generation++;
    keys.clear();
    if (!isSemantic(pilot)) sim.player.target = 0;
    lastApplied = 0;
    nextDecision = 0;
  }
});
$("close-help").onclick = () => $("help-dialog").close();

/* ---------- JSON inspector ---------- */
const TAB_TEXT = {
  request: "Exact model input. DriveJev: student state, offered behaviours and the three PNG frames (wide t−0.5 s, wide t, tele t; shown as sizes). Jev/Kev: the exact Decisions API payload.",
  sensor: "Complete forward perception, route guidance, geometry predictions and vehicle telemetry (local simulator view).",
  world: "All roads, buildings, vehicles, pedestrians, controls, and the current route.",
  decision: "Actual pilot response: probabilities, acceptance and timing. Jev costs accumulate across every completed call.",
};
$("json-description").textContent = TAB_TEXT.request;
$("scene-json").onclick = () => {
  touch.reset();
  $("json-dialog").showModal();
  renderJSON();
};
$("close-json").onclick = () => $("json-dialog").close();
document.querySelectorAll("[data-tab]").forEach(
  (button) =>
    (button.onclick = () => {
      inspectorTab = button.dataset.tab;
      inspectFrozen = false;
      syncFreeze();
      document.querySelectorAll("[data-tab]").forEach((b) => b.classList.toggle("active", b === button));
      $("json-description").textContent = TAB_TEXT[inspectorTab];
      renderJSON();
    }),
);
function syncFreeze() {
  $("freeze-json").textContent = inspectFrozen ? "Resume" : "Freeze";
  $("json-live").textContent = inspectFrozen ? "FROZEN" : "LIVE · 4 Hz";
}
$("freeze-json").onclick = () => {
  inspectFrozen = !inspectFrozen;
  syncFreeze();
};
function inspectRequest(state) {
  const { request, fixed } = prepareJevRequest(state);
  return Object.keys(request.questions).length ? request : { status: "No Jev call needed: only one eligible action.", resolved_locally: fixed };
}
function semanticResponse() {
  if (!semanticDecision) return { status: "No decision yet" };
  const { frame, history, pilot: who, ...rest } = semanticDecision;
  return { pilot: who.id, arm: who.arm ?? null, ...rest };
}
function inspectData() {
  if (inspectorTab === "request") {
    if (isSemantic(pilot)) return runtime.lastRequest ?? { status: isTeacherPilot(pilot) ? "The rule teacher reads simulator truth directly; no model request." : `Engage ${pilot.name} to send the first request.` };
    if (!sim.lastDecisionState || (!sim.autopilot && !showCandidates && !sim.paused)) requestPreview();
    return lastInput || (sim.lastDecisionState ? inspectRequest(sim.lastDecisionState) : { status: "Preparing driving state…" });
  }
  if (inspectorTab === "decision") {
    if (isSemantic(pilot)) return { response: semanticResponse(), session: runtime.stats, violations: runtime.violations(), aeb: runtime.aeb(), hazards: sim.director?.log ?? [] };
    return {
      response: lastDecision,
      last_submitted_input: lastInput,
      session: { ...tally, average_input_tokens: tally.calls ? Math.round(tally.input / tally.calls) : 0, average_request_bytes: tally.calls ? Math.round(tally.request_bytes / tally.calls) : 0 },
    };
  }
  return sim.observation(inspectorTab === "world");
}
let copyFeedbackTimer;
$("copy-json").onclick = async () => {
  const text = $("json-content").textContent;
  const button = $("copy-json");
  clearTimeout(copyFeedbackTimer);
  button.disabled = true;
  try {
    await navigator.clipboard.writeText(text);
    $("copy-json-label").textContent = "Copied!";
  } catch {
    inspectFrozen = true;
    syncFreeze();
    const range = document.createRange();
    range.selectNodeContents($("json-content"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    $("copy-json-label").textContent = "Press ⌘C / Ctrl+C";
  } finally {
    button.disabled = false;
    copyFeedbackTimer = setTimeout(() => ($("copy-json-label").textContent = "Copy"), 3000);
  }
};
$("download-json").onclick = () => {
  const text = inspectFrozen ? $("json-content").textContent : JSON.stringify(inspectData(), null, 2),
    a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  a.download = `drivejev-demo-${pilot.id}-${inspectorTab}-${sim.world.seed}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
};
function renderJSON() {
  if (inspectFrozen) return;
  const text = JSON.stringify(inspectData(), null, 2);
  $("json-content").innerHTML = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(
      /("(?:\\.|[^"\\])*"\s*:?)|\b(true|false|null)\b|(-?\d+(?:\.\d+)?)/g,
      (m) => `<span class="${m.startsWith('"') ? (m.endsWith(":") ? "json-key" : "json-string") : /true|false|null/.test(m) ? "json-bool" : "json-number"}">${m}</span>`,
    );
}

/* ---------- Jev / Kev decisions (JevPilot's structured-state loop) ---------- */
async function jevDecision(state) {
  const started = performance.now();
  const prepared = prepareJevRequest(state);
  const questions = prepared.request.questions;
  const apiCall = Object.keys(questions).length > 0;
  let data = { answers: {}, usage: { input_tokens: 0, output_tokens: 0 } };
  if (apiCall) {
    const res = await fetch(`/api/${pilot.backend}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request: prepared.request }),
      signal: AbortSignal.timeout(12000),
    });
    data = await res.json();
    if (!res.ok) throw Error(data.error || `${pilot.name} request failed`);
  }
  const answers = expandJevAnswers(prepared, data.answers);
  const selection = decisionSelection(state, answers);
  if (!selection) throw Error(`${pilot.name} returned an incomplete decision.`);
  if (questions.route && !Object.hasOwn(questions.route.criteria, answers.route?.choice)) throw Error(`${pilot.name} returned an invalid route choice.`);
  const selected = state.vectors[selection.choice];
  if ((selected.collision_imminent ?? selected.collision_predicted) && selected.velocity_mps !== 0)
    throw Error(`${pilot.name} selected a path with an imminent collision. Braking before retry.`);
  const usage = { input_tokens: data.usage?.input_tokens ?? 0, output_tokens: data.usage?.output_tokens ?? 0, cost: data.usage?.cost };
  const price = backends.jev?.pricing?.input_per_million ?? 0.042;
  return {
    model: data.model ?? null,
    decision_source: apiCall ? pilot.backend : "only_eligible_action",
    request_bytes: apiCall ? (data.request_bytes ?? JSON.stringify(prepared.request).length) : 0,
    candidate_ids: prepared.aliases,
    resolved_single_choices: Object.keys(prepared.fixed),
    answers,
    selection,
    batch_id: state.batch_id,
    controls: { steering: selected.steering, velocity: selected.velocity_mps },
    usage,
    latency_ms: Math.round(performance.now() - started),
    cost_usd: pilot.backend === "jev" ? (Number.isFinite(usage.cost) ? usage.cost : (usage.input_tokens * price) / 1e6) : 0,
  };
}
async function decide() {
  if (loading || busy || !sim.autopilot || sim.paused || document.hidden || sim.complete || sim.crash) return;
  if (isSemantic(pilot)) {
    runtime.tick();
    return;
  }
  const now = performance.now();
  if (now < nextDecision) {
    if (errors || now < nextContextCheck || now - lastApplied < 250) return;
    nextContextCheck = now + 100;
    if (!lastContext || !sim.decisionContextChanged(lastContext)) return;
  }
  busy = true;
  const token = generation,
    started = performance.now();
  try {
    const planned = await refreshPlan();
    if (!planned || token !== generation || !sim.autopilot || sim.paused || sim.crash) return;
    const { state, plan } = planned;
    scene.vectors.setCandidates(plan);
    lastInput = inspectRequest(state);
    const data = await jevDecision(state);
    if (data.decision_source === "only_eligible_action") tally.constrained_steps++;
    else tally.calls++;
    tally.request_bytes += data.request_bytes ?? 0;
    tally.cost += data.cost_usd;
    tally.input += data.usage.input_tokens;
    tally.output += data.usage.output_tokens;
    tally.latencies.push(data.latency_ms);
    if (tally.latencies.length > 25) tally.latencies.shift();
    if (token !== generation || state.route_version !== sim.routeVersion || !sim.autopilot || sim.paused || sim.crash) return;
    const controls = decisionControls(state, data);
    if (!controls) throw Error(`${pilot.name} returned a mismatched candidate batch.`);
    const now = performance.now();
    if (now - started > 1800) throw Error(`${pilot.name} decision expired before it arrived. Replanning.`);
    if (sim.decisionContextChanged(state)) {
      nextDecision = 0;
      return;
    }
    if (data.answers.route?.choice && data.answers.route.choice !== "keep" && sim.chooseRoute(data.answers.route.choice)) {
      nextDecision = 0;
      return;
    }
    if (lastApplied) tally.intervals.push(now - lastApplied);
    if (tally.intervals.length > 20) tally.intervals.shift();
    lastDecision = { ...data, received_at_simulation_s: sim.time };
    lastContext = state;
    lastApplied = now;
    errors = 0;
    sim.player.maneuver = state.vectors[data.selection.choice];
    sim.player.steering = controls.steering;
    sim.player.target = controls.velocity;
    scene.vectors.setAnswer(data.selection, plan);
    panel.jevDecision(data, plan, sim.time);
    panel.setLive("live", tally.intervals.length ? `${(1000 / (tally.intervals.reduce((a, b) => a + b, 0) / tally.intervals.length)).toFixed(1)} Hz` : "Live");
    nextDecision = started + decisionInterval(state);
  } catch (error) {
    if (token === generation) {
      sim.player.target = 0;
      scene.vectors.clear();
      errors++;
      nextDecision = performance.now() + Math.min(15000, 1000 * 2 ** errors);
      toast(error.message, "error");
      panel.error(error.message);
      panel.setLive("error", "Error");
      sim.event(error.message, "error");
      if (errors >= 3) {
        setPilot(false);
        toast(`${pilot.name} paused after three failed requests. Toggle the pilot to reconnect.`, "error");
      }
    }
  } finally {
    busy = false;
  }
}

/* ---------- minimap (unchanged from JevPilot) ---------- */
function drawMap() {
  const w = sim.world,
    v = sim.player,
    W = 380,
    H = 310;
  const view = minimap.view(),
    scale = view.scale;
  const pt = (p) => [(p.x - view.center.x) * scale, (p.z - view.center.z) * scale];
  map.clearRect(0, 0, W, H);
  map.fillStyle = "#f3f4f6";
  map.fillRect(0, 0, W, H);
  map.save();
  map.translate(W / 2, H * 0.65);
  map.rotate(-view.heading);
  map.lineCap = "round";
  map.strokeStyle = "#d0d3d8";
  if (w.roadSamples) {
    map.lineWidth = 25 * scale;
    map.beginPath();
    w.roadSamples.forEach((p, i) => (i ? map.lineTo(...pt(p)) : map.moveTo(...pt(p))));
    map.stroke();
  }
  if (w.connectorRoads) {
    for (const road of w.connectorRoads) {
      map.lineWidth = road.width * scale;
      map.beginPath();
      road.points.forEach((p, i) => (i ? map.lineTo(...pt(p)) : map.moveTo(...pt(p))));
      map.stroke();
    }
  } else if (!w.roadSamples)
    for (const e of w.edges) {
      map.lineWidth = e.width * scale;
      map.beginPath();
      map.moveTo(...pt(w.byId[e.a]));
      map.lineTo(...pt(w.byId[e.b]));
      map.stroke();
    }
  map.strokeStyle = "#3e6ae1";
  map.lineWidth = 4;
  map.beginPath();
  sim.player.route.points.forEach((p, i) => (i ? map.lineTo(...pt(p)) : map.moveTo(...pt(p))));
  map.stroke();
  for (const car of sim.traffic) {
    map.fillStyle = car.type === "motorcycle" ? "#e82127" : "#81858d";
    map.beginPath();
    map.arc(...pt(car), 4, 0, Math.PI * 2);
    map.fill();
  }
  const end = pt(sim.player.route.points.at(-1));
  map.fillStyle = "#171a20";
  map.fillRect(end[0] - 3, end[1] - 6, 8, 7);
  map.fillRect(end[0] - 3, end[1] - 6, 1, 14);
  map.save();
  map.translate(...pt(v));
  map.rotate(v.heading);
  map.fillStyle = "#ffffff";
  map.beginPath();
  map.arc(0, 0, 13, 0, Math.PI * 2);
  map.fill();
  map.fillStyle = "#171a20";
  map.beginPath();
  map.moveTo(0, -10);
  map.lineTo(7, 7);
  map.lineTo(0, 4);
  map.lineTo(-7, 7);
  map.closePath();
  map.fill();
  map.restore();
  map.restore();
}

/* ---------- HUD ---------- */
function signalAhead() {
  const crossing = sim.crossingFor(sim.player);
  if (!crossing) return null;
  const node = sim.world.byId[crossing.nodeId];
  const line = { ...pointAt(sim.player.route.points, crossing.stopS), heading: crossing.approach };
  const distance = stopLineDistance(sim.player, line);
  if (distance < -1 || distance > 120) return null;
  return { control: node.control, color: node.control === "signal" ? signalState(node, sim.time, crossing.approach).color : "stop", distance: Math.max(0, distance) };
}
function updateUI() {
  const v = sim.player,
    nav = sim.navigation();
  $("speed").textContent = Math.round(Math.abs(v.speed) * 3.6);
  $("speed-limit").textContent = Math.round((nav.speed_limit_mps ?? sim.world.theme.limit) * 3.6);
  $("remaining").textContent = nav.remaining_m >= 1000 ? `${(nav.remaining_m / 1000).toFixed(1)} km` : `${Math.round(nav.remaining_m)} m`;
  $("next-maneuver").textContent =
    nav.instruction ||
    (nav.next_turn === "arrive"
      ? sim.world.type === "highway"
        ? "Follow Interstate 08"
        : "Destination ahead"
      : nav.next_turn === "straight"
        ? "Continue straight"
        : nav.next_turn === "uturn"
          ? "Make a U-turn"
          : `Turn ${nav.next_turn}`);
  $("turn-distance").textContent = nav.next_turn === "arrive" ? "to your destination" : `in ${Math.round(nav.turn_distance_m)} m`;
  const turnIcon = { uturn: "rotate-ccw", left: "corner-up-left", right: "corner-up-right", straight: "arrow-up", arrive: "flag", merge: "corner-up-left", exit: "corner-up-right" }[nav.next_turn];
  if ($("turn-icon").dataset.icon !== turnIcon) {
    $("turn-icon").innerHTML = icon(turnIcon);
    $("turn-icon").dataset.icon = turnIcon;
    createIcons({ icons });
  }
  const signal = signalAhead();
  panel.truth(signal, activeHazard());
  announceHazards();
  if (sim.autopilot) panel.sample(sim.time, Math.abs(v.speed) / Math.max(1, sim.world.theme.limit), signal?.color ?? null);
  panel.draw(sim.time);
  if (isSemantic(pilot)) updateSemanticStatus();
  else updateJevStatus();
  if (nav.rerouted) $("context-message").textContent = "Route recalculated · continuing to your destination";
  else if (rerouting && sim.routeChoiceNeeded()) $("context-message").textContent = "Recalculating route…";
  if (sim.complete && !sim.freeExplore) {
    $("arrival").hidden = false;
    $("arrival-summary").textContent = `${Math.round(sim.distance)} m driven · ${sim.collisions} contacts · ${sim.violations} violations${sim.autopilot ? ` · ${pilot.name}` : ""}`;
    if ($("autopilot").getAttribute("aria-checked") === "true") {
      generation++;
      runtime.disengage();
      sim.autopilot = false;
      syncPilot();
    }
  }
  if ($("json-dialog").open) renderJSON();
  previewCamera();
}
// Scripted hazards (truth, for the viewer only — never a model input).
let hazardSim = null,
  hazardSeen = 0;
const HAZARD_TEXT = {
  jaywalker: (e) => `a pedestrian will step out from the ${e.from} ${Math.round(e.at_s - e.ego_s)} m ahead`,
  cross_runner: (e) => `a car will run the ${e.control === "signal" ? "red light" : e.control === "stop" ? "stop sign" : "junction"} ahead`,
  lead_brake: (e) => `a car pulls in ${Math.round(e.gap)} m ahead and will brake hard`,
  oncoming: (e) => (e.variant === "platoon" ? `${e.cars} oncoming cars with right of way while you turn left` : "an oncoming car will turn left across your path"),
  stop_contention: (e) => `${e.cars.length} more car${e.cars.length > 1 ? "s" : ""} arriving at the 4-way stop`,
  green_runner: () => "a cross car will run its red just after your light turns green",
  occluded_ped: () => "a pedestrian is hidden behind a parked car ahead",
  cut_in: (e) => (e.highway ? "a car will cut in from the next lane" : `a parked car will pull out ${Math.round(e.gap)} m ahead`) + (e.brake ? " and brake" : ""),
  turn_ped: (e) => `${e.peds.length} pedestrian${e.peds.length > 1 ? "s" : ""} crossing the road you turn into`,
};
function hazardActivations() {
  return (sim.director?.log ?? []).filter((e) => !e.event);
}
function announceHazards() {
  if (hazardSim !== sim) [hazardSim, hazardSeen] = [sim, 0];
  const activations = hazardActivations();
  if (activations.length <= hazardSeen) return;
  const e = activations.at(-1);
  hazardSeen = activations.length;
  toast(`Hazard · ${HAZARD_TEXT[e.kind]?.(e) ?? e.kind}`);
}
const PED_TEXT = { jaywalker: "jaywalker", occluded_ped: "hidden pedestrian", turn_ped: "pedestrian at the turn" };
function activeHazard() {
  const v = sim.player,
    ped = sim.hazardPeds?.find((p) => ["staged", "waiting", "walking"].includes(p.hazard?.state));
  if (ped) return `${PED_TEXT[ped.hazard.kind] ?? "pedestrian"} ${ped.hazard.state === "walking" ? "crossing" : "waiting"} · ${Math.round(dist(ped, v))} m`;
  const car = sim.hazardCars?.filter((c) => c.hazard?.state === "active" && !c.hazard.static).sort((a, b) => dist(a, v) - dist(b, v))[0];
  if (!car) return null;
  const h = car.hazard;
  const what =
    h.kind === "cross_runner" ? `${h.control === "signal" ? "red-light" : h.control === "stop" ? "stop-sign" : "junction"} runner`
    : h.kind === "lead_brake" ? (h.brakeActive ? "lead car braking hard" : "lead car (will brake)")
    : h.kind === "oncoming" ? (h.variant === "left_turner" ? "oncoming left-turner" : "oncoming car (right of way)")
    : h.kind === "stop_contention" ? "car at the 4-way stop"
    : h.kind === "green_runner" ? "late red-light runner"
    : h.kind === "cut_in" ? (h.capActive ? "cut-in car braking" : "cut-in car")
    : "lead car resumed";
  return `${what} · ${Math.round(dist(car, v))} m`;
}
// Idle preview: show what the DriveJev cameras would see before the pilot is engaged.
let lastCameraPreview = 0;
function previewCamera() {
  if (!isModelPilot(pilot) || sim.autopilot || engaging || !sensor.scene || sensor.world !== sim.world || sensor.scene.sim !== sim) return;
  if (performance.now() - lastCameraPreview < 400 || $("decision-panel").hidden) return;
  lastCameraPreview = performance.now();
  sensor
    .capture(sim)
    .then((shot) => panel.showPreview(shot.dataUrl, shot.tele.dataUrl))
    .catch((error) => console.warn("Sensor camera unavailable", error));
}
function updateSemanticStatus() {
  const v = sim.player,
    stats = runtime.stats;
  if (!sim.autopilot) {
    $("pilot-state").textContent = sim.crash ? "Drive ended" : "Free play";
    $("context-message").textContent = engaging ? "Preparing the model camera…" : touch.available ? "Drag to drive · Hold Brake to stop" : "WASD to drive · Space to brake";
    $("cost-label").textContent = isModelPilot(pilot) ? "Local GPU" : "Local rule";
    $("cost").textContent = pilotAvailability(pilot).ok ? pilot.short : "offline";
    return;
  }
  const maneuver = runtime.executor?.maneuver;
  const decision = semanticDecision;
  const executing = actionStyle(maneuver?.action_type ?? "keep_route_cruise").label + (maneuver?.status === "completed" ? " ✓" : "");
  const chosenType = decision?.candidates.find((c) => c.candidate_id === decision.candidate_id)?.action_type;
  const p = decision?.probabilities?.[decision.candidate_id];
  const share = Number.isFinite(p) ? ` ${Math.round(p * 100)}%` : "";
  $("pilot-state").textContent = !decision
    ? "Reading the road…"
    : chosenType === "continue_current"
      ? `${executing} · continue${share}`
      : chosenType === maneuver?.action_type
        ? `${executing} ·${share}`
        : `${executing} · wants ${actionStyle(chosenType).label.toLowerCase()}${share}`;
  const aeb = runtime.aeb();
  $("context-message").textContent = runtime.executor?.stalled
    ? "Waiting for the model · braking"
    : aeb?.active
      ? `AEB · braking for a predicted ${aeb.conflict ?? "conflict"}`
      : sim.brakeReason
      ? `Safety brake · ${sim.brakeReason}`
      : !decision
        ? "0.5 s bootstrap · cruise/hold until the first decision"
        : `${Math.round(v.target * 3.6)} km/h target · ${Math.round(decision.latency_ms)} ms`;
  const model = isModelPilot(pilot);
  $("cost-label").textContent = model ? "Applied" : "Local rule";
  $("cost").textContent = model ? `${runtime.appliedHz().toFixed(1)} Hz` : `${stats.accepted} decisions`;
  const violations = runtime.violations();
  const median = (values) => (values?.length ? [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)] : null);
  const safety = aeb
    ? `<span>${aeb.on ? `<b>${aeb.events}</b> AEB intervention${aeb.events === 1 ? "" : "s"}${aeb.active ? " · braking now" : ""}` : "AEB off"}</span>`
    : "";
  const hazards = hazardsOn && worldKind !== "signal" ? `<span><b>${hazardActivations().length}</b> hazards triggered</span>` : "";
  const required = violations && violations.required !== violations.red ? ` · <b>${violations.required}</b> missed required stops` : "";
  panel.metrics(
    model
      ? `<span><b>${DecisionPanel.latencyLine(stats)}</b> round trip</span><span><b>${stats.completed}</b> requests · <b>${stats.rejected}</b> not applied · <b>${stats.skipped_busy}</b> frames skipped</span><span><b>${violations?.red ?? 0}</b> red-light crossings${required}</span>${safety}${hazards}`
      : `<span><b>${stats.accepted}</b> decisions · local rule${stats.teacher_ms?.length ? ` · ${Math.round(median(stats.teacher_ms))} ms/decision` : ""}</span><span><b>${violations?.red ?? 0}</b> red-light crossings${required}</span>${safety}${hazards}`,
  );
  panel.rate(model ? `${runtime.appliedHz().toFixed(1)} Hz applied · 4 Hz camera` : `${runtime.appliedHz().toFixed(1)} Hz applied`);
}
function updateJevStatus() {
  const v = sim.player;
  const answer = lastDecision?.selection,
    stale = !lastApplied || performance.now() - lastApplied > 1800;
  $("pilot-state").textContent = sim.autopilot
    ? stale
      ? "Reading the road…"
      : `${candidateName(scene.vectors.answeredPlan?.vectors[answer.choice])} · ${Math.round(answer.confidence * 100)}%`
    : sim.crash
      ? "Drive ended"
      : "Free play";
  $("context-message").textContent = sim.autopilot
    ? sim.brakeReason
      ? `Safety brake · ${sim.brakeReason}`
      : stale
        ? "Waiting for a fresh decision"
        : `${Math.round(v.target * 3.6)} km/h target · ${lastDecision.latency_ms} ms`
    : touch.available
      ? "Drag to drive · Hold Brake to stop"
      : "WASD to drive · Space to brake";
  if (sim.autopilot && !stale && !sim.brakeReason && v.speed < 0.5 && v.target < 0.5)
    $("context-message").textContent = lastDecision.decision_source === "only_eligible_action" ? "Only stop is available · rechecking scene" : `${pilot.short} chose to wait · evaluating traffic`;
  if (sim.lastPlan?.recovery.active && sim.autopilot) $("context-message").textContent = sim.lastPlan.road.on_road ? "Returning to the route" : "Finding a way back onto the road";
  $("cost-label").textContent = pilot.backend === "jev" ? "Session" : "Local";
  $("cost").textContent = pilot.backend === "jev" ? `$${tally.cost.toFixed(6)}` : `${tally.calls} calls`;
  const latency = tally.latencies.length ? [...tally.latencies].sort((a, b) => a - b)[Math.floor(tally.latencies.length / 2)] : null;
  panel.metrics(
    `<span><b>${latency ?? "—"}${latency !== null ? " ms" : ""}</b> round trip</span><span><b>${tally.calls}</b> calls · <b>${tally.calls ? Math.round(tally.input / tally.calls).toLocaleString() : 0}</b> tokens/call</span>` +
      (pilot.backend === "jev" ? `<span><b>$${tally.cost.toFixed(5)}</b> session</span>` : ""),
  );
  panel.rate(lastDecision ? `~${(1000 / decisionInterval(sim.lastDecisionState ?? { recovery: {}, scene: {} })).toFixed(1)} Hz target` : "");
  if (!sim.autopilot) return;
  const state = sim.lastDecisionState;
  if (state)
    panel.showState([
      ["speed", `${(state.speed_mps * 3.6).toFixed(0)} km/h · limit ${(state.limit_mps * 3.6).toFixed(0)}`],
      ["next turn", `${state.turn.direction} in ${Math.round(state.turn.in_m)} m`],
      ["junction", state.scene?.intersection ? `${state.scene.intersection.control === "signal" ? `signal ${state.scene.intersection.signal}` : "stop sign"} · line ${state.scene.intersection.stop_line_ahead_m?.toFixed(1)} m` : "none ahead"],
      ["traffic", `${state.scene?.nearby?.length ?? 0} nearby${state.scene?.following ? ` · lead ${state.scene.following.gap_m?.toFixed(1)} m` : ""}`],
      ["candidates", `${Object.keys(state.vectors).length} sampled 3 s paths`],
    ]);
}

/* ---------- main loop ---------- */
let physicsClock = 0,
  previousPoses = null;
const movers = () => [sim.player, ...sim.traffic, ...sim.pedestrians];
const poses = () => movers().map((o) => [o, o.x, o.z, o.heading]);
// Draw objects between the last two fixed physics steps; the true state is restored
// right after rendering so physics, capture and decisions never see display poses.
function interpolate(previous, alpha) {
  const saved = [];
  for (const [o, x, z, heading] of previous) {
    saved.push([o, o.x, o.z, o.heading]);
    if (Math.hypot(o.x - x, o.z - z) > 4) continue; // respawn/teleport
    const turn = Math.atan2(Math.sin(o.heading - heading), Math.cos(o.heading - heading));
    o.x = x + (o.x - x) * alpha;
    o.z = z + (o.z - z) * alpha;
    o.heading = heading + turn * alpha;
  }
  return saved;
}
function restore(saved) {
  for (const [o, x, z, heading] of saved) Object.assign(o, { x, z, heading });
}
function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min((now - lastNow) / 1000, 0.2);
  lastNow = now;
  if (document.hidden || loading) return;
  touch.sync();
  if (!sim.paused && !sim.crash) {
    if (!sim.autopilot) {
      let steer = 0;
      const left = keys.has("KeyA") || keys.has("ArrowLeft");
      const right = keys.has("KeyD") || keys.has("ArrowRight");
      if (left || right) steer = Number(right) - Number(left);
      let throttle = 0;
      if (keys.has("KeyW") || keys.has("ArrowUp")) throttle = 1;
      if (keys.has("KeyS") || keys.has("ArrowDown")) throttle = -1;
      sim.pedals.throttle = throttle || touch.throttle;
      sim.pedals.brake = keys.has("Space") ? 1 : touch.brake;
      sim.steeringInput = steer || touch.steering;
      sim.player.target = 0;
    } else if (!runtime.active && (!lastApplied || now - lastApplied > 1800)) sim.player.target = 0;
    if (sim.autopilot && runtime.active) {
      // Semantic pilots run on the fixed 20 Hz training clock; the display interpolates.
      physicsClock += dt;
      let steps = 0;
      while (physicsClock >= DT - 1e-9 && steps < 8) {
        previousPoses = poses();
        runtime.stepFixed();
        physicsClock -= DT;
        steps++;
      }
      if (steps === 8) physicsClock = 0;
    } else {
      physicsClock = 0;
      previousPoses = null;
      const steps = Math.max(1, Math.ceil(dt / 0.025));
      for (let i = 0; i < steps; i++) sim.step(dt / steps);
    }
  }
  if (scene.routeVersion !== sim.routeVersion) {
    scene.routeVersion = sim.routeVersion;
    generation++;
    lastApplied = 0;
    lastDecision = null;
    lastInput = null;
    lastContext = null;
    nextDecision = 0;
    scene.vectors.clear();
    const destination = sim.player.route.points.at(-1);
    scene.destination.position.set(destination.x, 0.2, destination.z);
  }
  if (sim.crash && !crashHandled) {
    crashHandled = true;
    generation++;
    keys.clear();
    runtime.disengage();
    scene.vectors.clear();
    semanticVectors.clear();
    syncPilot();
    $("arrival").hidden = true;
    $("paused-overlay").hidden = true;
    $("json-dialog").close();
    $("help-dialog").close();
    document.body.classList.add("crashed");
    $("crash-description").textContent = {
      building: "You collided with a building.",
      pedestrian: "You struck a pedestrian.",
      car: "You collided with another car.",
      motorcycle: "You collided with a motorcycle.",
    }[sim.crash.type] + (sim.autopilot ? ` (${pilot.name} was driving.)` : "");
    $("crash-speed").textContent = Math.round(sim.crash.impact_speed_mps * 3.6);
    $("crash-distance").textContent = Math.round(sim.distance);
    $("crash-dialog").showModal();
  }
  if (showCandidates && !sim.autopilot && !sim.paused && !sim.crash && now - candidatePreviewAt > 500) {
    candidatePreviewAt = now;
    requestPreview();
  }
  const actual = runtime.active && previousPoses ? interpolate(previousPoses, Math.min(1, physicsClock / DT)) : null;
  scene.render(dt, false);
  const { width, height } = scene.viewport;
  semanticVectors.render(sim.player, scene.camera, width, height, dt, sim.autopilot && runtime.active);
  scene.renderer.render(scene.scene, scene.camera);
  if (actual) restore(actual);
  if (!$("minimap").hidden && now - lastMapDraw >= 100) {
    drawMap();
    lastMapDraw = now;
  }
  uiTime += dt;
  if (uiTime > 0.2) {
    uiTime = 0;
    updateUI();
  }
}

async function pollStatus() {
  try {
    const data = await (await fetch("/api/status", { cache: "no-store" })).json();
    const wasReady = pilotAvailability(pilot).ok;
    backends = data;
    if (!$("pilot-menu").hidden) renderPilotMenu();
    if (!wasReady && pilotAvailability(pilot).ok) toast(`${pilot.name} is ready · press J to engage`);
    syncPilot();
    if (!sim.autopilot) updateSemanticStatusIfIdle();
  } catch {
    backends = { drivejev: { status: "unavailable" }, jev: { configured: false }, kev: { available: false } };
  }
  setTimeout(pollStatus, backends.drivejev?.status === "ready" ? 15000 : 3000);
}
function updateSemanticStatusIfIdle() {
  if (isSemantic(pilot)) updateSemanticStatus();
}

panel.setPilot(pilot, icon(pilot.icon));
if (matchMedia("(max-width: 650px)").matches) panel.toggleCollapsed(true);
createIcons({ icons });
refreshWorld();
syncPilot();
updateUI();
requestAnimationFrame(animate);
finishLoading().catch(loadingFailed);
setInterval(decide, 25);
pollStatus();

// Handle for headless checks and the browser console.
window.__demo = {
  get sim() {
    return sim;
  },
  get pilot() {
    return pilot;
  },
  get loading() {
    return loading;
  },
  get engaging() {
    return engaging;
  },
  get backends() {
    return backends;
  },
  set backends(value) {
    backends = value;
  },
  get hazardsOn() {
    return hazardsOn;
  },
  get semanticDecision() {
    return semanticDecision;
  },
  runtime,
  sensor,
  scene,
  panel,
  setPilot,
  selectPilot: (id) => selectPilot(PILOTS.find((p) => p.id === id)),
  resetWorld,
  setHazards,
  PILOTS,
};
export { sim, scene, runtime };
