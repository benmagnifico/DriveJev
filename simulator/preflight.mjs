/**
 * Preflight for evaluation suites: reads a JSON job list on stdin and reports seeds whose initial state already
 * overlaps an NPC footprint (a JevPilot spawn artefact that ends the drive at t = 0.05 s regardless of policy).
 *   node simulator/preflight.mjs < jobs.json   ->   {"overlap_seeds": [...]}
 */
import { InteractionWorld } from './interactions.mjs';
import { footprintClearance } from '../third_party/jevpilot/src/traffic-safety.js';

let text = '';
for await (const chunk of process.stdin) text += chunk;
const bad = [];
for (const cfg of JSON.parse(text)) {
  const sim = new InteractionWorld(cfg.seed, cfg.type, { randomRoute: cfg.randomRoute ?? true, hazards: cfg.hazards ?? [], hazardCooldown: cfg.hazardCooldown });
  const v = sim.player;
  if ([...sim.traffic, ...sim.pedestrians].some((o) => Math.hypot(o.x - v.x, o.z - v.z) < 8 && footprintClearance(v, o) < 0.3)) bad.push(cfg.seed);
}
console.log(JSON.stringify({ overlap_seeds: bad }));
