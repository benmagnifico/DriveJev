// Live "decision process" panel: what the pilot saw (wide t, wide t-0.5 s, tele t and the
// student state), what it was asked, how it answered, whether the answer was applied, and a
// rolling 20 s timeline.
import { actionStyle } from "./semantic-vectors.js";
import { candidateName } from "/third_party/jevpilot/src/planning.js";

const WINDOW_S = 20;
const pct = (p) => `${Math.round(p * 100)}%`;
const esc = (text) =>
  String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const median = (values) => {
  if (!values?.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
};
const REASONS = {
  nonurgent_hysteresis: "hysteresis: previous action < 0.5 s old",
  stale_observation: "observation older than 0.5 s",
  candidate_signature_changed: "offered actions changed while thinking",
  maneuver_version_changed: "maneuver changed while thinking",
  chosen_not_offered_in_observation: "choice not offered",
  unknown_observation: "unknown observation",
  cross_episode: "answer belongs to an earlier episode",
  missing_model_version: "response without model version",
};
const STATE_KINDS = ["drivejev", "teacher"];
const CAMERA_KINDS = ["drivejev"];
const TEACHER_KINDS = ["teacher"];
const num = (x, unit = "", digits = 1) => (Number.isFinite(x) ? `${x.toFixed(digits)}${unit}` : "—");
const SIGNAL_COLORS = { red: "#e5484d", amber: "#f0a83c", green: "#3fae6a" };

export class DecisionPanel {
  constructor(root) {
    this.root = root;
    root.innerHTML = `
<header class="dp-head">
  <span class="dp-icon" id="dp-icon"></span>
  <div class="dp-title"><strong id="dp-name"></strong><span id="dp-tag" class="dp-tag"></span></div>
  <span id="dp-live" class="dp-live" data-state="idle">Idle</span>
  <button id="dp-collapse" class="dp-collapse" aria-label="Collapse decision panel" aria-expanded="true" title="Collapse panel · I"><i data-lucide="chevron-down"></i></button>
</header>
<div class="dp-body" id="dp-body">
  <section class="dp-input">
    <div class="dp-section-title"><span id="dp-input-title">Model input</span><span id="dp-input-meta"></span></div>
    <div id="dp-camera" class="dp-camera" hidden>
      <img id="dp-current" alt="Current front-camera frame sent to the model" />
      <figure class="dp-history"><img id="dp-history" alt="Front-camera frame from 0.5 s earlier" /><figcaption>t−0.5 s</figcaption></figure>
      <span class="dp-camera-label">t</span>
    </div>
    <div id="dp-strip" class="dp-strip" hidden>
      <figure><img id="dp-history7" alt="Wide front frame from 0.5 s earlier sent to the model" /><figcaption>wide · t−0.5 s</figcaption></figure>
      <figure class="dp-tele"><img id="dp-tele" alt="Tele (15°) front frame sent to the model" /><figcaption>tele 15° · t</figcaption></figure>
    </div>
    <dl id="dp-state" class="dp-state" hidden></dl>
    <div id="dp-obs-wrap" hidden>
      <div class="dp-section-title dp-obs-title"><span>Student state</span><span>student_obs · model input</span></div>
      <dl id="dp-obs" class="dp-state dp-obs"></dl>
    </div>
  </section>
  <section class="dp-decision">
    <div class="dp-section-title"><span id="dp-question">Waiting for the first decision</span><span id="dp-confidence"></span></div>
    <ol id="dp-bars" class="dp-bars"></ol>
    <p id="dp-acceptance" class="dp-acceptance"></p>
  </section>
  <section class="dp-timeline">
    <div class="dp-section-title"><span>Timeline · ${WINDOW_S} s</span><span id="dp-rate"></span></div>
    <canvas id="dp-canvas" width="584" height="104" aria-label="Chosen action, confidence, speed and true signal state over the last 20 seconds"></canvas>
    <div class="dp-legend"><span><i class="dp-key line"></i>confidence</span><span><i class="dp-key speed"></i>speed</span><span><i class="dp-key signal"></i>signal (truth)</span></div>
  </section>
  <footer class="dp-foot">
    <div id="dp-metrics" class="dp-metrics"></div>
    <div id="dp-truth" class="dp-truth"></div>
  </footer>
</div>`;
    this.$ = (id) => root.querySelector(`#${id}`);
    this.canvas = this.$("dp-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.decisions = [];
    this.samples = [];
    this.$("dp-collapse").onclick = () => this.toggleCollapsed();
  }
  toggleCollapsed(force) {
    const collapsed = force ?? !this.root.classList.contains("collapsed");
    this.root.classList.toggle("collapsed", collapsed);
    this.$("dp-collapse").setAttribute("aria-expanded", String(!collapsed));
  }
  setPilot(pilot, icon) {
    this.pilot = pilot;
    this.$("dp-icon").innerHTML = icon;
    this.$("dp-name").textContent = pilot.name.replace(/\s*\((ours|privileged|local|cloud)\)$/i, ""); // the tag says it
    this.$("dp-tag").textContent = pilot.tag;
    this.$("dp-tag").dataset.tag = pilot.tag.toLowerCase().replace(/\s+/g, "-");
    const camera = CAMERA_KINDS.includes(pilot.kind),
      withState = STATE_KINDS.includes(pilot.kind),
      teacher = TEACHER_KINDS.includes(pilot.kind);
    this.$("dp-camera").hidden = !camera;
    this.$("dp-camera").classList.toggle("tele", camera);
    this.$("dp-strip").hidden = !camera;
    this.$("dp-state").hidden = camera;
    this.$("dp-obs-wrap").hidden = !withState;
    this.$("dp-input-title").textContent = camera ? "Model input" : teacher ? "Privileged input" : "Jev input · structured state";
    this.$("dp-input-meta").textContent = camera
      ? "wide 640×384 t, t−0.5 s · tele 384×224 t"
      : teacher
        ? "true rule + 4.5 s rollouts of perceived road users, no camera"
        : "no camera";
    this.reset();
  }
  reset() {
    this.decisions = [];
    this.samples = [];
    this.$("dp-bars").replaceChildren();
    this.$("dp-question").textContent = "Engage to see live decisions";
    this.$("dp-confidence").textContent = "";
    this.$("dp-acceptance").textContent = "";
    this.$("dp-current").removeAttribute("src");
    this.$("dp-history").removeAttribute("src");
    this.$("dp-history7").removeAttribute("src");
    this.$("dp-tele").removeAttribute("src");
    this.$("dp-state").replaceChildren();
    this.$("dp-obs").innerHTML = "<dt>—</dt><dd>engage to see the observation the model reads</dd>";
    this.setLive("idle", "Idle");
    this.draw();
  }
  setLive(state, text) {
    const live = this.$("dp-live");
    live.dataset.state = state;
    live.textContent = text;
  }
  showPreview(dataUrl, teleUrl = null) {
    this.$("dp-current").src = dataUrl;
    this.$("dp-history").removeAttribute("src");
    this.$("dp-history7").removeAttribute("src");
    if (teleUrl) this.$("dp-tele").src = teleUrl;
    this.$("dp-camera").classList.remove("bootstrap");
    this.$("dp-camera").classList.add("preview");
  }
  showFrame(frame, history) {
    this.$("dp-camera").classList.remove("preview");
    this.$("dp-current").src = frame.dataUrl;
    if (history) this.$("dp-history").src = history.dataUrl;
    if (frame.tele) {
      this.$("dp-tele").src = frame.tele.dataUrl;
      if (history) this.$("dp-history7").src = history.dataUrl;
      else this.$("dp-history7").removeAttribute("src");
    }
    this.$("dp-camera").classList.toggle("bootstrap", !history);
  }
  /** Student observation summary: exactly the fields the model reads as text. */
  showObs(obs) {
    if (!obs) return;
    const { ego = {}, nav = {}, traffic = {} } = obs;
    const lead = traffic.lead,
      hazard = traffic.hazard,
      junction = traffic.junction;
    const control = { signal: "signal", stop: "stop sign" }[nav.junction_control] ?? nav.junction_control;
    const rows = [
      ["ego", `${num(ego.speed_mps, " m/s")} · stationary ${num(ego.stationary_s, " s")}`],
      [
        "junction",
        control
          ? `${control} · line ${num(nav.stop_line_ahead_m, " m")} · ${nav.stop_completed ? "stop done" : "no stop yet"}`
          : Number.isFinite(nav.stop_line_ahead_m)
            ? `uncontrolled · line ${num(nav.stop_line_ahead_m, " m")}`
            : "none ahead",
      ],
      ["lead", lead ? `${num(lead.gap_m, " m")} gap · ${num(lead.speed_mps, " m/s")}` : "none"],
      ["hazard", hazard ? `${hazard.type} ${hazard.side} · ${num(hazard.distance_m, " m")} · in ${num(hazard.in_s, " s")}` : "none predicted"],
      [
        "junction box",
        junction
          ? `${junction.vehicles_inside} inside · ${junction.cross_approaching} approaching · ${junction.pedestrians_crossing} peds${nav.junction_control === "stop" ? ` · ${junction.earlier_arrivals} arrived first` : ""}${junction.oncoming_eta_s != null ? ` · oncoming in ${junction.oncoming_eta_s} s` : ""}${junction.cross_eta_s != null ? ` · crossing in ${junction.cross_eta_s} s` : ""}`
          : "—",
      ],
    ];
    this.$("dp-obs").innerHTML = rows.map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`).join("");
    this.$("dp-obs").dataset.hazard = hazard ? "yes" : "no";
  }
  showState(rows) {
    this.$("dp-state").innerHTML = rows
      .map(([key, value]) => `<dt>${esc(key)}</dt><dd>${esc(value)}</dd>`)
      .join("");
  }
  bars(items) {
    this.$("dp-bars").innerHTML = items
      .map(
        (item) => `<li class="dp-bar${item.chosen ? " chosen" : ""}${item.executing ? " executing" : ""}" style="--bar:${item.color}">
  <span class="dp-bar-label"><b>${esc(item.label)}</b>${item.sub ? `<small>${esc(item.sub)}</small>` : ""}</span>
  <span class="dp-bar-track"><span class="dp-bar-fill" style="width:${Math.max(1.5, item.p * 100).toFixed(1)}%"></span></span>
  <span class="dp-bar-value">${pct(item.p)}</span>
  <span class="dp-bar-flag">${item.executing ? "▶ driving" : item.chosen ? "chosen" : ""}</span>
</li>`,
      )
      .join("");
  }
  /** DriveJev decision head or reference teacher over the offered behaviours. */
  semanticDecision(decision, simTime) {
    const executing = decision.executing;
    const items = decision.candidates
      .map((c) => {
        const style = actionStyle(c.action_type);
        return {
          id: c.candidate_id,
          label: style.label,
          sub: style.hint,
          color: style.color,
          p: decision.probabilities?.[c.candidate_id] ?? 0,
          chosen: c.candidate_id === decision.candidate_id,
          executing: c.action_type === executing,
        };
      })
      .sort((a, b) => b.p - a.p);
    this.bars(items);
    const best = items[0];
    this.$("dp-question").textContent = `Which driving action next? · ${decision.candidates.length} offered`;
    this.$("dp-confidence").textContent = best ? `top ${pct(best.p)}` : "";
    const a = decision.acceptance;
    const age = decision.applied_sim_time - decision.captured_sim_time;
    this.$("dp-acceptance").innerHTML = a.accepted
      ? `<span class="ok">✓ ${a.continued ? "kept" : "applied"}</span> ${esc(actionStyle(decision.candidates.find((c) => c.candidate_id === decision.candidate_id)?.action_type).label)} · ${(age * 1000).toFixed(0)} ms after capture`
      : `<span class="warn">✕ not applied</span> ${esc(REASONS[a.reason] ?? a.reason)}`;
    this.push(simTime, actionStyle(decision.candidates.find((c) => c.candidate_id === decision.candidate_id)?.action_type), best?.p ?? 0, a.accepted);
  }
  /** Jev / Kev: drive-or-stop question plus the conditional path choice. */
  jevDecision(data, plan, simTime) {
    const motion = data.answers?.motion?.probabilities ?? {};
    const selection = data.selection;
    const vectors = plan?.vectors ?? {};
    const items = [];
    for (const [id, p] of Object.entries(motion))
      items.push({ label: id === "drive" ? "Drive" : "Stop", sub: id === "drive" ? "move" : "wait", color: id === "drive" ? "#48a5ff" : "#e6a34b", p, chosen: data.answers.motion.choice === id });
    if (!items.length) items.push({ label: "Only one option", sub: "resolved locally", color: "#8f9bb3", p: 1, chosen: true });
    const paths = Object.entries(selection.probabilities)
      .filter(([id]) => vectors[id]?.velocity_mps)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 4)
      .map(([id, p]) => ({
        label: `${candidateName(vectors[id])} · ${(Math.abs(vectors[id].velocity_mps) * 3.6).toFixed(0)} km/h`,
        sub: `steer ${vectors[id].steering >= 0 ? "+" : ""}${vectors[id].steering.toFixed(2)} · ${vectors[id].stays_in_lane === false ? "leaves lane" : "in lane"}${vectors[id].collision_predicted ? " · conflict" : ""}`,
        color: vectors[id].collision_predicted ? "#e86940" : vectors[id].steering < -0.02 ? "#38bcd6" : "#48a5ff",
        p,
        chosen: id === selection.choice,
        executing: id === selection.choice,
      }));
    this.bars([...items, ...paths]);
    this.$("dp-question").textContent = data.decision_source === "only_eligible_action" ? "Single eligible action · no API call" : "Drive or stop? Then which path?";
    this.$("dp-confidence").textContent = `confidence ${pct(selection.confidence ?? 0)}`;
    this.$("dp-acceptance").innerHTML = `<span class="ok">✓ applied</span> ${esc(candidateName(vectors[selection.choice]))} · ${data.latency_ms} ms round trip`;
    const chosen = vectors[selection.choice];
    this.push(simTime, { label: candidateName(chosen), color: chosen?.velocity_mps ? "#48a5ff" : "#e6a34b" }, selection.probabilities[selection.choice] ?? 0, true);
  }
  error(message) {
    this.$("dp-acceptance").innerHTML = `<span class="warn">✕ error</span> ${esc(message)}`;
  }
  push(t, style, p, accepted) {
    this.decisions.push({ t, color: style?.color ?? "#8f9bb3", label: style?.label, p, accepted });
    while (this.decisions.length && this.decisions[0].t < t - WINDOW_S - 5) this.decisions.shift();
  }
  sample(t, speedRatio, signal) {
    const last = this.samples.at(-1);
    if (last && t < last.t) this.samples = [];
    this.samples.push({ t, speed: speedRatio, signal });
    while (this.samples.length && this.samples[0].t < t - WINDOW_S - 1) this.samples.shift();
  }
  metrics(html) {
    this.$("dp-metrics").innerHTML = html;
  }
  truth(signal, hazard = null) {
    this.$("dp-truth").innerHTML =
      (signal
        ? `<span class="dp-signal" style="--signal:${SIGNAL_COLORS[signal.color] ?? "#8f9bb3"}"></span>${esc(signal.control === "stop" ? "Stop sign" : `Signal ${signal.color?.toUpperCase()}`)} · ${signal.distance.toFixed(1)} m <small>ground truth · not a model input</small>`
        : `<small>No controlled junction on the route ahead</small>`) +
      (hazard ? `<span class="dp-hazard-truth">⚠ ${esc(hazard)} <small>scripted hazard · truth</small></span>` : "");
  }
  rate(text) {
    this.$("dp-rate").textContent = text;
  }
  draw(now = this.samples.at(-1)?.t ?? 0) {
    const ctx = this.ctx,
      W = this.canvas.width,
      H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);
    const x = (t) => W - ((now - t) / WINDOW_S) * W;
    const bandTop = 4,
      bandH = 14,
      plotTop = 24,
      plotH = H - 24 - 14,
      signalTop = H - 9;
    ctx.fillStyle = "#f3f4f6";
    ctx.fillRect(0, plotTop, W, plotH);
    ctx.strokeStyle = "#e3e5e8";
    ctx.lineWidth = 1;
    for (const level of [0.25, 0.5, 0.75]) {
      ctx.beginPath();
      ctx.moveTo(0, plotTop + plotH * (1 - level));
      ctx.lineTo(W, plotTop + plotH * (1 - level));
      ctx.stroke();
    }
    // Chosen-action band: each decision holds until the next one.
    for (let i = 0; i < this.decisions.length; i++) {
      const d = this.decisions[i],
        end = this.decisions[i + 1]?.t ?? now;
      if (end < now - WINDOW_S) continue;
      ctx.globalAlpha = d.accepted ? 1 : 0.35;
      ctx.fillStyle = d.color;
      ctx.fillRect(Math.max(0, x(d.t)), bandTop, Math.max(1, x(end) - Math.max(0, x(d.t)) - 0.5), bandH);
    }
    ctx.globalAlpha = 1;
    // Confidence of the chosen option.
    ctx.strokeStyle = "#007aff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    this.decisions.forEach((d, i) => {
      const px = x(d.t),
        py = plotTop + plotH * (1 - d.p);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.stroke();
    // Speed relative to the limit.
    ctx.strokeStyle = "#171a20";
    ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    this.samples.forEach((s, i) => {
      const px = x(s.t),
        py = plotTop + plotH * (1 - Math.min(1, s.speed));
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);
    // True signal state strip (reference only).
    for (let i = 0; i < this.samples.length; i++) {
      const s = this.samples[i],
        next = this.samples[i + 1]?.t ?? now;
      ctx.fillStyle = SIGNAL_COLORS[s.signal] ?? "#d0d3d8";
      ctx.fillRect(x(s.t), signalTop, Math.max(1, x(next) - x(s.t) + 0.5), 6);
    }
  }
  static latencyLine({ latencies = [], backbone = [] }) {
    const total = median(latencies),
      core = median(backbone);
    return total === null ? "—" : `${total.toFixed(0)} ms${core !== null ? ` (backbone ${core.toFixed(0)})` : ""}`;
  }
}
