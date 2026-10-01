#!/usr/bin/env python
"""Summarize DriveJev closed-loop runs (episodes.jsonl from simulator/run.mjs).

    python eval/summarize.py <run dir> [<run dir> ...] [--json out.json] [--markdown]
Success = arrived at the destination with no collision and no traffic violation
(JevPilot red-light/stop-sign count + front-bumper required-stop count).
"""
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def scenario(e):
    c = e["config"]
    if c["type"] == "highway":
        return "highway"
    return f"{c['type']}+hazards" if c.get("hazards") else f"{c['type']}"


def bootstrap(values, n=2000, seed=0):
    if not values:
        return [None, None]
    r = random.Random(seed)
    means = sorted(sum(r.choice(values) for _ in values) / len(values) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def metrics(rows):
    ok = [e for e in rows if "error" not in e]
    if not ok:
        return {"episodes": len(rows), "errors": len(rows)}
    viol = [e["violations"] + e.get("front_required_stop_violations", 0) for e in ok]
    success = [1.0 if e["arrived"] and not e["crash"] and v == 0 else 0.0 for e, v in zip(ok, viol)]
    hazards = sum(sum(1 for h in e.get("hazards", []) if "event" not in h) for e in ok)
    sim = sum(e["sim_time_s"] for e in ok)
    km = sum(e.get("route_completion", 0) * e.get("route_length_m", 0) for e in ok) / 1000
    return {
        "episodes": len(ok), "errors": len(rows) - len(ok),
        "success_rate": sum(success) / len(ok), "success_ci95": bootstrap(success),
        "arrival_rate": sum(e["arrived"] for e in ok) / len(ok),
        "collisions": sum(1 for e in ok if e["crash"]),
        "collision_types": sorted({(e["crash"] or {}).get("type") for e in ok if e["crash"]}),
        "red_or_stop_violations": sum(viol),
        "route_completion": sum(e.get("route_completion", 0) for e in ok) / len(ok),
        "km": km, "sim_hours": sim / 3600,
        "mean_speed_mps": sum(e["mean_speed_mps"] * e["sim_time_s"] for e in ok) / max(sim, 1e-9),
        "stuck_s_per_episode": sum(e.get("stuck_s", 0) for e in ok) / len(ok),
        "hard_brake_s_per_km": sum((e.get("hard_brake_steps") or 0) * 0.05 for e in ok) / max(km, 1e-9),
        "aeb_events": sum(e.get("aeb_events") or 0 for e in ok),
        "hazard_events": hazards,
        "latency_ms_p50": sorted(e["latency_ms_p50"] for e in ok if e.get("latency_ms_p50"))[len([1 for e in ok if e.get("latency_ms_p50")]) // 2] if any(e.get("latency_ms_p50") for e in ok) else None,
        "applied_hz": (sum(e.get("applied_hz", 0) for e in ok) / len(ok)) if any("applied_hz" in e for e in ok) else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--json")
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()
    table = {}
    for run in args.runs:
        rows = [json.loads(l) for l in (Path(run) / "episodes.jsonl").read_text().splitlines() if l.strip()]
        by = defaultdict(list)
        for e in rows:
            by[scenario(e) if "config" in e else "error"].append(e)
            by["all"].append(e)
        table[run] = {k: metrics(v) for k, v in sorted(by.items())}
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=1))
    for run, scen in table.items():
        print(f"== {run}")
        for k, m in scen.items():
            if "success_rate" not in m:
                print(f"  {k:16s} {m}")
                continue
            ci = m["success_ci95"]
            print(f"  {k:16s} n={m['episodes']:3d} success={m['success_rate']:.2f} [{ci[0]:.2f},{ci[1]:.2f}] arrive={m['arrival_rate']:.2f} "
                  f"crash={m['collisions']} viol={m['red_or_stop_violations']} compl={m['route_completion']:.2f} v={m['mean_speed_mps']:.1f} "
                  f"stuck={m['stuck_s_per_episode']:.1f}s hardbrake={m['hard_brake_s_per_km']:.1f}s/km aeb={m['aeb_events']} km={m['km']:.1f} lat={m['latency_ms_p50']}")


if __name__ == "__main__":
    main()
