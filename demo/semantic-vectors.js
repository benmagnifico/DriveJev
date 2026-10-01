// 3D overlay for semantic pilots (DriveJev / reference teacher): one ribbon per offered
// action, previewed with the real executor controller on a cloned ego car, plus
// probability labels. Styled after JevPilot's RoadVectors.
import * as THREE from "three";

const STEPS = 60;

export const ACTION_STYLE = {
  keep_route_cruise: { color: "#48a5ff", label: "Cruise", hint: "follow the route" },
  proceed_route: { color: "#38bcd6", label: "Proceed", hint: "start moving" },
  route_turn: { color: "#5b8cff", label: "Turn", hint: "take the turn" },
  stop_at_line: { color: "#e6a34b", label: "Stop at line", hint: "stop before the line" },
  yield_agent: { color: "#a35ee0", label: "Yield", hint: "stop for a crosser" },
  hold_stop: { color: "#d99a2b", label: "Hold stop", hint: "stay stopped" },
  continue_current: { color: "#8f9bb3", label: "Continue", hint: "keep current" },
  emergency_brake: { color: "#e86940", label: "Emergency brake", hint: "maximum braking" },
};
export const actionStyle = (type) =>
  ACTION_STYLE[type] ?? { color: "#8f9bb3", label: type, hint: type };

function ribbon() {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array((STEPS + 1) * 6), 3).setUsage(
      THREE.DynamicDrawUsage,
    ),
  );
  const progress = [],
    indices = [];
  for (let i = 0; i <= STEPS; i++) progress.push(i / STEPS, i / STEPS);
  geometry.setAttribute("progress", new THREE.Float32BufferAttribute(progress, 1));
  for (let i = 0; i < STEPS; i++) {
    const n = i * 2;
    indices.push(n, n + 1, n + 2, n + 1, n + 3, n + 2);
  }
  geometry.setIndex(indices);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      tint: { value: new THREE.Color("#007aff") },
      alpha: { value: 0.3 },
      time: { value: 0 },
      pulse: { value: 0 },
      ego: { value: new THREE.Vector3() },
      body: { value: new THREE.Vector2() },
    },
    vertexShader: `attribute float progress; varying float vProgress; varying vec2 vWorld; void main() { vProgress = progress; vWorld = (modelMatrix * vec4(position, 1.0)).xz; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
    fragmentShader: `uniform vec3 tint; uniform float alpha; uniform float time; uniform float pulse; uniform vec3 ego; uniform vec2 body; varying float vProgress; varying vec2 vWorld; void main() {
      vec2 delta = vWorld - ego.xy;
      float right = dot(delta, vec2(cos(ego.z), sin(ego.z)));
      float ahead = dot(delta, vec2(sin(ego.z), -cos(ego.z)));
      if (abs(right) < body.x && abs(ahead) < body.y) discard;
      float fade = (1.0 - smoothstep(0.72,1.0,vProgress)) * smoothstep(0.015,0.08,vProgress);
      float scan = 1.0 - pulse * (0.5 + 0.5 * sin(vProgress * 18.0 - time * 6.0));
      gl_FragColor = vec4(tint, alpha * fade * scan);
      #include <colorspace_fragment>
    }`,
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: false,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = 3;
  return mesh;
}

function updateRibbon(mesh, points, width, y) {
  const a = mesh.geometry.attributes.position;
  for (let i = 0; i <= STEPS; i++) {
    const p = points[Math.min(i, points.length - 1)],
      before = points[Math.max(0, Math.min(i, points.length - 1) - 1)],
      after = points[Math.min(points.length - 1, i + 1)],
      dx = after.x - before.x,
      dz = after.z - before.z,
      len = Math.hypot(dx, dz) || 1;
    a.setXYZ(i * 2, p.x - (dz / len) * width, y, p.z + (dx / len) * width);
    a.setXYZ(i * 2 + 1, p.x + (dz / len) * width, y, p.z - (dx / len) * width);
  }
  a.needsUpdate = true;
}

// Previews are world paths from the 4 Hz refresh; each frame drops the part the car
// has already driven so the ribbon stays attached to the front of the car.
function ahead(car, points) {
  let best = 0,
    bestDistance = Infinity;
  for (let i = 0; i < points.length; i++) {
    const d = (points[i].x - car.x) ** 2 + (points[i].z - car.z) ** 2;
    if (d < bestDistance) [best, bestDistance] = [i, d];
  }
  return points.slice(best);
}

export class SemanticVectors {
  constructor(scene, layer) {
    this.group = new THREE.Group();
    this.group.name = "semantic-action-previews";
    this.group.visible = false;
    scene.add(this.group);
    this.layer = layer;
    this.showCandidates = false;
    this.pool = Array.from({ length: 8 }, () => {
      const line = ribbon();
      const label = document.createElement("span");
      label.className = "vector-label semantic-label";
      label.hidden = true;
      layer.append(label);
      this.group.add(line);
      return { line, label };
    });
    this.selected = ribbon();
    this.selectedGlow = ribbon();
    this.group.add(this.selectedGlow, this.selected);
    this.state = null;
  }
  attach(scene) {
    this.group.removeFromParent();
    scene.add(this.group);
  }
  /** previews: [{candidate_id, action_type, points:[{x,z}], progress_m}] from controllerPreview(). */
  set({ origin, previews, probabilities, executing, chosen }) {
    this.state = {
      origin: { ...origin },
      previews: previews.map((p) => ({ ...p })),
      probabilities: probabilities ?? {},
      executing,
      chosen,
      received: performance.now(),
    };
  }
  clear() {
    this.state = null;
    this.group.visible = false;
    for (const item of this.pool) item.label.hidden = true;
  }
  dispose() {
    this.clear();
    for (const mesh of this.group.children) {
      mesh.geometry.dispose();
      mesh.material.dispose();
    }
    for (const item of this.pool) item.label.remove();
    this.group.removeFromParent();
  }
  render(car, camera, width, height, dt, active) {
    const state = this.state;
    this.group.visible = !!state && active;
    for (const item of this.pool) item.label.hidden = true;
    if (!state || !active) return;
    const time = performance.now() / 1000;
    for (const mesh of this.group.children) {
      mesh.material.uniforms.ego.value.set(car.x, car.z, car.heading);
      mesh.material.uniforms.body.value.set(car.width / 2 + 0.12, car.depth / 2 + 0.15);
      mesh.material.uniforms.time.value = time;
    }
    const executing = state.previews.find((p) => p.candidate_id === state.executing) ??
      state.previews.find((p) => p.action_type === state.executing);
    this.selected.visible = this.selectedGlow.visible = !!executing && executing.progress_m > 0.15;
    if (this.selected.visible) {
      const points = ahead(car, executing.points);
      updateRibbon(this.selected, points, 0.2, 0.25);
      updateRibbon(this.selectedGlow, points, 0.5, 0.195);
      this.selected.material.uniforms.tint.value.set("#007aff");
      this.selectedGlow.material.uniforms.tint.value.set("#007aff");
      this.selected.material.uniforms.alpha.value = 0.9;
      this.selectedGlow.material.uniforms.alpha.value = 0.15;
      this.selected.material.uniforms.pulse.value = state.chosen && state.chosen !== executing.candidate_id ? 0.6 : 0;
      this.selected.renderOrder = 5;
    }
    let index = 0;
    const labels = [];
    for (const item of this.pool) item.line.visible = false;
    for (const preview of state.previews) {
      const item = this.pool[index++];
      if (!item) break;
      const isExecuting = preview === executing;
      const probability = state.probabilities[preview.candidate_id];
      const style = actionStyle(preview.action_type);
      const points = ahead(car, preview.points);
      const moving = preview.progress_m > 0.15;
      if (this.showCandidates && moving && !isExecuting) {
        item.line.visible = true;
        updateRibbon(item.line, points, 0.055, 0.22);
        item.line.material.uniforms.tint.value.set(style.color);
        item.line.material.uniforms.alpha.value = 0.65;
      }
      const showLabel = this.showCandidates || isExecuting || preview.candidate_id === state.chosen;
      if (!showLabel || probability === undefined) continue;
      const anchor = moving
        ? points[Math.min(points.length - 1, Math.floor(STEPS * 0.5))]
        : { x: car.x + Math.sin(car.heading) * 3.2, z: car.z - Math.cos(car.heading) * 3.2 };
      const screen = new THREE.Vector3(anchor.x, 0.85, anchor.z).project(camera);
      if (screen.z > 1 || screen.z < 0 || Math.abs(screen.x) > 1 || Math.abs(screen.y) > 1) continue;
      const text = `${style.label} ${Math.round(probability * 100)}%`;
      labels.push({ item, isExecuting, probability, style, preview, text,
        x: (screen.x * 0.5 + 0.5) * width, y: (-0.5 * screen.y + 0.5) * height,
        w: text.length * (isExecuting ? 7.4 : 6.6) + 18, h: isExecuting ? 24 : 21 });
    }
    // Greedy de-overlap in screen space: executing first, then by probability.
    labels.sort((a, b) => b.isExecuting - a.isExecuting || b.probability - a.probability);
    const placed = [];
    for (const label of labels) {
      let { x, y } = label;
      for (let attempt = 0; attempt < 10; attempt++) {
        const hit = placed.find((r) => Math.abs(r.x - x) < (r.w + label.w) / 2 + 2 && Math.abs(r.y - y) < (r.h + label.h) / 2 + 2);
        if (!hit) break;
        y = hit.y + (hit.h + label.h) / 2 + 3;
      }
      placed.push({ x, y, w: label.w, h: label.h });
      const element = label.item.label;
      element.hidden = false;
      element.classList.toggle("selected", label.isExecuting);
      element.classList.toggle("chosen", label.preview.candidate_id === state.chosen && !label.isExecuting);
      element.style.setProperty("--action", label.style.color);
      element.textContent = label.text;
      element.title = label.preview.description ?? label.style.hint;
      element.style.opacity = label.isExecuting ? "1" : "0.88";
      element.style.transform = `translate(${x}px,${y}px) translate(-50%,-50%)`;
    }
  }
}
