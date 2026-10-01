/**
 * DriveJev closed-loop runner: static server + model proxy + N headless Chrome workers.
 *
 *   node simulator/run.mjs --jobs eval/suites/test.json --out runs/test-ours [--workers 4] [--port 8981]
 *        [--model-url http://127.0.0.1:9031/predict] [--set policy=model,arm=default,aeb=false,realtime=false]
 *
 * Each job is a runEpisode()/runRealtime() config ({seed, type, hazards, max_time_s, ...}); `--set`
 * overrides fields for every job. Writes <out>/episodes.jsonl and <out>/run.json. <out> must be new.
 */
import http from 'node:http';
import fs from 'node:fs';
import fsp from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const own = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(own, '..');
const jevpilot = path.join(repo, 'third_party/jevpilot');
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg', '.hdr': 'application/octet-stream', '.glb': 'model/gltf-binary', '.woff2': 'font/woff2', '.css': 'text/css', '.webp': 'image/webp', '.ktx2': 'application/octet-stream' };
const arg = (k, d) => { const i = process.argv.indexOf(k); return i < 0 ? d : process.argv[i + 1]; };
const jobsPath = arg('--jobs'), outArg = arg('--out');
if (!jobsPath || !outArg) { console.error('usage: node simulator/run.mjs --jobs <suite.json> --out <new dir> [--workers 4] [--model-url URL] [--set k=v,...]'); process.exit(2); }
const out = path.resolve(outArg), workers = Number(arg('--workers', '4')), port = Number(arg('--port', '8981'));
const modelUrl = arg('--model-url', 'http://127.0.0.1:9031/predict');
const parse = (v) => (v === 'true' ? true : v === 'false' ? false : Number.isFinite(Number(v)) && v !== '' ? Number(v) : v);
const overrides = Object.fromEntries((arg('--set', '') || '').split(',').filter(Boolean).map((kv) => { const [k, v] = kv.split('='); return [k, parse(v)]; }));
if (fs.existsSync(out) && fs.readdirSync(out).length) throw Error(`${out} is not empty; choose a new output directory`);
fs.mkdirSync(out, { recursive: true });
const jobs = JSON.parse(fs.readFileSync(jobsPath, 'utf8')).map((j) => ({ policy: 'model', ...j, ...overrides }));
const episodes = fs.openSync(path.join(out, 'episodes.jsonl'), 'a');

function staticFile(p) {
  if (p === '/') return path.join(own, 'index.html');
  if (p.startsWith('/vendor/three/')) return path.join(jevpilot, 'node_modules/three', p.slice('/vendor/three/'.length));
  if (p.startsWith('/simulator/') || p.startsWith('/third_party/')) return path.join(repo, p);
  return path.join(jevpilot, 'public', p); // JevPilot models / textures
}
async function readBody(req) { const chunks = []; for await (const c of req) chunks.push(c); return Buffer.concat(chunks); }
const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://127.0.0.1');
    if (url.pathname === '/predict') {
      const upstream = await fetch(modelUrl, { method: 'POST', headers: { 'content-type': 'application/json' }, body: await readBody(req) });
      res.writeHead(upstream.status, { 'content-type': 'application/json' }); res.end(await upstream.text()); return;
    }
    if (url.pathname === '/favicon.ico') { res.writeHead(204); res.end(); return; }
    const file = path.resolve(staticFile(decodeURIComponent(url.pathname)));
    if (!file.startsWith(repo + path.sep)) throw Error('outside repository');
    const data = await fsp.readFile(file);
    res.writeHead(200, { 'content-type': mime[path.extname(file)] ?? 'application/octet-stream' }); res.end(data);
  } catch (error) { res.writeHead(404, { 'content-type': 'application/json' }); res.end(JSON.stringify({ error: String(error) })); }
});
await new Promise((r) => server.listen(port, '127.0.0.1', r));
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ? pathToFileURL(process.env.PLAYWRIGHT_MODULE).href : 'playwright');
const browser = await chromium.launch({ headless: true, executablePath: process.env.CHROME_PATH || undefined,
  args: ['--no-sandbox', '--disable-dev-shm-usage', '--ignore-gpu-blocklist', '--enable-gpu', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows'] });
let next = 0, done = 0, failed = 0;
const started = Date.now();
async function worker(w) {
  const page = await browser.newPage({ viewport: { width: 700, height: 420 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load', timeout: 60000 });
  await page.waitForFunction(() => window.drivejevReady, {}, { timeout: 120000 });
  while (next < jobs.length) {
    const index = next++, job = jobs[index];
    let result;
    try { result = await page.evaluate((cfg) => (cfg.realtime ? window.drivejev.runRealtime(cfg) : window.drivejev.runEpisode(cfg)), job); }
    catch (error) { result = { error: String(error).slice(0, 2000), config: job }; failed++; }
    Object.assign(result, { job_index: index, worker: w, page_errors: errors.splice(0) });
    fs.writeSync(episodes, JSON.stringify(result) + '\n');
    done++;
    const brief = result.error ? `ERROR ${result.error.slice(0, 160)}` : `${job.type}/${job.seed} arrived=${result.arrived} crash=${!!result.crash} violations=${result.violations + result.front_required_stop_violations} t=${result.sim_time_s.toFixed(1)}s`;
    console.log(`[${done}/${jobs.length}] ${brief}`);
  }
  await page.close();
}
await Promise.all(Array.from({ length: Math.min(workers, jobs.length) }, (_, w) => worker(w)));
fs.closeSync(episodes);
fs.writeFileSync(path.join(out, 'run.json'), JSON.stringify({ jobs: jobsPath, overrides, model_url: modelUrl, episodes: done, failed, wall_s: (Date.now() - started) / 1000, finished_at: new Date().toISOString() }, null, 2));
await browser.close(); server.close();
