#!/usr/bin/env python3
"""Track H webux3 repair: guarded FPS re-measurement.
Fixes the harness hazard found in round-1/round-2 numbers: the 10.0s clip could END
mid-probe (follow probe covers video-time ~5.8-9.8s+), and an ended video lightens the
render loop and inflates FPS (trial-3 court=116.5 artifact). Guards: loop=true, playing
state asserted before/after every probe, alternating preset order, n=4 per preset."""
import json
from playwright.sync_api import sync_playwright

AUDIT = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/webux3_audit_20260716"
FIXL = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/webux3_fixes_20260716"
RICH = f"{AUDIT}/manifest_fresh_wolv_local.json"
BASE = "http://127.0.0.1:5173"
out = {"probes": [], "notes": "loop=true; state-guarded; alternating order; 4s rAF probes"}

def fps_probe(page, secs=4):
    return page.evaluate("(s)=>new Promise(res=>{let n=0;const t0=performance.now();function tick(){n++;if(performance.now()-t0<s*1000)requestAnimationFrame(tick);else res(n/s);}requestAnimationFrame(tick);})", secs)

def vstate(page):
    return page.evaluate("()=>{const v=document.querySelector('video');return {paused:v.paused,ended:v.ended,t:v.currentTime,loop:v.loop};}")

def ensure_playing(page):
    page.evaluate("()=>{const v=document.querySelector('video'); v.muted=true; v.loop=true; if(v.paused||v.ended){v.play();}}")
    page.wait_for_timeout(600)
    s = vstate(page)
    assert not s["paused"] and not s["ended"], f"video not playing: {s}"
    return s

def click_preset(page, name):
    try:
        page.get_by_role("button", name=name, exact=True).first.click(timeout=3000)
    except Exception:
        page.get_by_text(name, exact=True).first.click(timeout=3000)
    page.wait_for_timeout(700)

with sync_playwright() as p:
    b = p.chromium.launch(headless=False, args=["--use-angle=metal"])
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:200]))
    pg.goto(f"{BASE}/?manifest=/@fs{RICH}", wait_until="domcontentloaded")
    pg.wait_for_timeout(7000)
    pg.evaluate("()=>{const v=document.querySelector('video'); v.muted=true; v.loop=true; v.currentTime=0.2;}")

    orders = [("Court", "Follow player"), ("Follow player", "Court")] * 2
    court, follow = [], []
    for i, order in enumerate(orders):
        for preset in order:
            click_preset(pg, preset)
            s0 = ensure_playing(pg)
            fps = fps_probe(pg)
            s1 = vstate(pg)
            valid = (not s1["paused"]) and (not s1["ended"])
            out["probes"].append({"trial": i, "preset": preset, "fps": round(fps, 1),
                                  "t0": round(s0["t"], 2), "t1": round(s1["t"], 2), "valid": valid})
            print(f"trial {i} {preset:14s} fps={fps:6.1f} video t {s0['t']:.2f}->{s1['t']:.2f} valid={valid}")
            (court if preset == "Court" else follow).append((fps, valid))
    b.close()

cv = [f for f, v in court if v]; fv = [f for f, v in follow if v]
out["court_avg"] = sum(cv)/len(cv) if cv else None
out["follow_avg"] = sum(fv)/len(fv) if fv else None
out["ratio"] = (out["follow_avg"]/out["court_avg"]) if cv and fv else None
out["errors"] = errs
with open(f"{FIXL}/fps_recheck_result.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ("court_avg", "follow_avg", "ratio")}, indent=1))
print("VERDICT: follow>=0.8*court ->", (out["ratio"] or 0) >= 0.8)
