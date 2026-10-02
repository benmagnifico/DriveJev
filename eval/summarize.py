#!/usr/bin/env python3
"""Summarize DriveJev closed-loop runs (episodes.jsonl from simulator/run.mjs).

    python eval/summarize.py <run dir> [<run dir> ...] [--json out.json]

Success = arrived, no collision, no violation (JevPilot red light / stop sign + front-bumper required stop).
Per hazard kind: how many scripted hazard events of that kind occurred and how many ended in a collision with
the hazard agent (crash_hazard). Stuck = stationary while the rule allows motion and no lead within 4 m;
stuck_clear additionally requires that the perceived scene predicts no conflict.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def scenario(e):
    c = e["config"]
    if c["type"] == "highway":
        return "highway"
    fam = c.get("family") or ("hazards" if c.get("hazards") else "plain")
    return f"{c['type']}+{fam}" if fam != "plain" else c["type"]


def bootstrap(values, n=2000, seed=0):
    if not values:
        return [None, None]
    r = random.Random(seed)
    means = sorted(sum(r.choice(values) for _ in values) / len(values) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def ok(e):
    return bool(e["arrived"] and not e["crash"] and e["violations"] + e.get("front_required_stop_violations", 0) == 0)


def metrics(rows):
    good = [e for e in rows if "error" not in e]
    if not good:
        return {"episodes": len(rows), "errors": len(rows)}
    success = [1.0 if ok(e) else 0.0 for e in good]
    sim = sum(e["sim_time_s"] for e in good)
    km = sum(e.get("route_completion", 0) * e.get("route_length_m", 0) for e in good) / 1000
    events = Counter(h["kind"] for e in good for h in e.get("hazards", []) if "event" not in h)
    hz_crash = Counter(e.get("crash_hazard") for e in good if e.get("crash") and e.get("crash_hazard"))
    lat = sorted(e["latency_ms_p50"] for e in good if e.get("latency_ms_p50"))
    return {
        "episodes": len(good), "errors": len(rows) - len(good),
        "success": int(sum(success)), "success_rate": sum(success) / len(good), "success_ci95": bootstrap(success),
        "arrival_rate": sum(e["arrived"] for e in good) / len(good),
        "collisions": sum(1 for e in good if e["crash"]),
        "hazard_collisions": sum(hz_crash.values()),
        "collision_types": dict(Counter((e["crash"] or {}).get("type") for e in good if e["crash"])),
        "violations": sum(e["violations"] + e.get("front_required_stop_violations", 0) for e in good),
        "route_completion": sum(e.get("route_completion", 0) for e in good) / len(good),
        "km": km, "sim_hours": sim / 3600,
        "mean_speed_mps": sum(e["mean_speed_mps"] * e["sim_time_s"] for e in good) / max(sim, 1e-9),
        "stuck_s_per_episode": sum(e.get("stuck_s", 0) for e in good) / len(good),
        "stuck_clear_s_per_episode": sum(e.get("stuck_clear_s", 0) or 0 for e in good) / len(good),
        "hard_brake_s_per_km": sum((e.get("hard_brake_steps") or 0) * 0.05 for e in good) / max(km, 1e-9),
        "aeb_events": sum(e.get("aeb_events") or 0 for e in good),
        "hazard_events": dict(events), "hazard_collisions_by_kind": dict(hz_crash),
        "latency_ms_p50": lat[len(lat) // 2] if lat else None,
        "applied_hz": (sum(e.get("applied_hz", 0) for e in good) / len(good)) if any("applied_hz" in e for e in good) else None,
    }


def load(run):
    return [json.loads(l) for l in (Path(run) / "episodes.jsonl").read_text().splitlines() if l.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--json")
    args = p.parse_args()
    table = {}
    for run in args.runs:
        rows = load(run)
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
                print(f"  {k:20s} {m}")
                continue
            ci = m["success_ci95"]
            print(f"  {k:20s} n={m['episodes']:3d} success={m['success']:2d} ({m['success_rate']:.2f} [{ci[0]:.2f},{ci[1]:.2f}]) arrive={m['arrival_rate']:.2f} "
                  f"crash={m['collisions']} (hazard {m['hazard_collisions']}) viol={m['violations']} compl={m['route_completion']:.2f} v={m['mean_speed_mps']:.1f} "
                  f"stuck={m['stuck_s_per_episode']:.1f}/{m['stuck_clear_s_per_episode']:.1f}s aeb={m['aeb_events']} lat={m['latency_ms_p50']}")
        hz = scen["all"].get("hazard_events", {})
        if hz:
            print("  hazards:", {k: f"{scen['all']['hazard_collisions_by_kind'].get(k, 0)}/{n}" for k, n in sorted(hz.items())})


if __name__ == "__main__":
    main()
