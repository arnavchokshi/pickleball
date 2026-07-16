#!/usr/bin/env python3
"""Track H webux3: manager AFTER-fix browser verification. Screenshots -> fixes lane shots_after/."""
import json, sys, time
from playwright.sync_api import sync_playwright

AUDIT = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/webux3_audit_20260716"
FIXL = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/webux3_fixes_20260716"
RICH = f"{AUDIT}/manifest_fresh_wolv_local.json"
DEGR = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/trk_flip_20260713/default_production/wolverine_mixed_0200_mid_steep_corner/replay_viewer_manifest.json"
VMORIG = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/demo_beststack_render_20260710/fresh_wolv/wolverine_mixed_0200_mid_steep_corner/replay_viewer_manifest.json"
BASE = "http://127.0.0.1:5173"
out = {"checks": [], "errors": [], "fps": {}}

def check(name, ok, detail=""):
    out["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:250]})
    print(("PASS " if ok else "FAIL "), name, "--", str(detail)[:160])

def fps_probe(page, secs=4):
    return page.evaluate("(s)=>new Promise(res=>{let n=0;const t0=performance.now();function tick(){n++;if(performance.now()-t0<s*1000)requestAnimationFrame(tick);else res(n/s);}requestAnimationFrame(tick);})", secs)

with sync_playwright() as p:
    b = p.chromium.launch(headless=False, args=["--use-angle=metal"])
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    pg.on("pageerror", lambda e: out["errors"].append(f"rich: {str(e)[:200]}"))

    # --- rich bundle ---
    pg.goto(f"{BASE}/?manifest=/@fs{RICH}", wait_until="domcontentloaded")
    pg.wait_for_timeout(7000)
    pg.screenshot(path=f"{FIXL}/shots_after/after_00_initial.png")

    # T1: video pane in first viewport; upload panel below/collapsed
    g = pg.evaluate("""() => {
      const v=document.querySelector('video'); const r=v?v.getBoundingClientRect():null;
      const up=[...document.querySelectorAll('button,summary')].find(e=>/predict court/i.test(e.innerText||''));
      const upVisible = up ? !!(up.offsetWidth||up.offsetHeight) : false;
      const upY = up ? up.getBoundingClientRect().y + window.scrollY : null;
      const vY = r ? r.y + window.scrollY : null;
      return {videoTop: vY, uploadY: upY, uploadVisible: upVisible, page: document.body.scrollHeight};
    }""")
    check("T1 video pane top < 700", g["videoTop"] is not None and g["videoTop"] < 700, g)
    check("T1 upload below video or collapsed", (g["uploadY"] is None) or (not g["uploadVisible"]) or (g["uploadY"] > (g["videoTop"] or 0)), g)

    # T2: FPS x3 trials
    trials = []
    for t in range(3):
        pg.evaluate("()=>{const v=document.querySelector('video'); v.muted=true; v.currentTime=0.2; v.play();}")
        pg.wait_for_timeout(700)
        court = fps_probe(pg)
        try:
            pg.get_by_role("button", name="Follow player", exact=True).click(timeout=3000)
        except Exception:
            pg.get_by_text("Follow player").first.click(timeout=3000)
        pg.wait_for_timeout(700)
        follow = fps_probe(pg)
        try:
            pg.get_by_role("button", name="Court", exact=True).first.click(timeout=3000)
        except Exception:
            pg.get_by_text("Court", exact=True).first.click(timeout=3000)
        pg.wait_for_timeout(500)
        trials.append({"court_play": court, "follow_play": follow})
        pg.evaluate("()=>document.querySelector('video')?.pause()")
    out["fps"]["trials"] = trials
    cavg = sum(t["court_play"] for t in trials)/3; favg = sum(t["follow_play"] for t in trials)/3
    out["fps"]["court_avg"] = cavg; out["fps"]["follow_avg"] = favg
    check("T2 court-play >= 37.5 (baseline)", cavg >= 37.5*0.95, f"court_avg={cavg:.1f} trials={[round(t['court_play'],1) for t in trials]}")
    check("T2 follow within 20% of court", favg >= 0.8*cavg, f"follow_avg={favg:.1f} vs court_avg={cavg:.1f}")
    pg.screenshot(path=f"{FIXL}/shots_after/after_01_playing.png")

    # T3: single timeline (no native controls), play button exists, markers/tooltip
    t3 = pg.evaluate("""() => {
      const v=document.querySelector('video');
      const strip=document.querySelector('.timeline-strip');
      const markers=strip?strip.querySelectorAll('[class*=marker]').length:0;
      const playBtn=[...document.querySelectorAll('button')].some(b=>/play|pause/i.test((b.getAttribute('aria-label')||b.innerText||'')));
      return {nativeControls: v?v.hasAttribute('controls'):null, hasStrip: !!strip, markers, playBtn};
    }""")
    check("T3 native video controls removed", t3["nativeControls"] is False, t3)
    check("T3 explicit play/pause present", t3["playBtn"], t3)
    check("T3 timeline strip + markers present", t3["hasStrip"] and t3["markers"] > 0, t3)

    # T4: camera framing screenshots (manual judgment)
    for preset in ["Court", "Follow player", "Free orbit"]:
        try:
            pg.get_by_role("button", name=preset, exact=True).first.click(timeout=3000)
            pg.wait_for_timeout(900)
            pg.screenshot(path=f"{FIXL}/shots_after/after_02_preset_{preset.replace(' ','_')}.png")
        except Exception as e:
            out["errors"].append(f"preset {preset}: {str(e)[:120]}")

    # T5/T6: in-pane badges + consolidated absence
    body = pg.evaluate("() => document.body.innerText")
    check("T5 trust vocab present", ("PREVIEW" in body or "LOW CONFIDENCE" in body), "vocab scan")
    pg.screenshot(path=f"{FIXL}/shots_after/after_03_badges.png")

    # T7: warnings expandable
    try:
        pg.get_by_text("notices", exact=False).first.click(timeout=2500)
        pg.wait_for_timeout(400)
        pg.screenshot(path=f"{FIXL}/shots_after/after_04_warnings_open.png")
    except Exception as e:
        out["errors"].append(f"warnings expand: {str(e)[:120]}")

    # T8: shots empty state
    try:
        pg.get_by_role("button", name="Shots", exact=True).click(timeout=2500)
        pg.wait_for_timeout(700)
        pg.screenshot(path=f"{FIXL}/shots_after/after_05_shots.png")
    except Exception as e:
        out["errors"].append(f"shots tab: {str(e)[:120]}")

    # bottom overview
    pg.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    pg.wait_for_timeout(400)
    pg.screenshot(path=f"{FIXL}/shots_after/after_06_bottom.png")

    # --- degraded bundle honesty ---
    pg2 = b.new_page(viewport={"width": 1600, "height": 1000})
    pg2.on("pageerror", lambda e: out["errors"].append(f"degraded: {str(e)[:200]}"))
    pg2.goto(f"{BASE}/?manifest=/@fs{DEGR}", wait_until="domcontentloaded")
    pg2.wait_for_timeout(5000)
    dbody = pg2.evaluate("() => document.body.innerText")
    check("degraded: preview/absence vocabulary intact", ("PREVIEW" in dbody), "scan")
    check("degraded: missing stays missing (ball)", ("missing" in dbody.lower()), "scan")
    pg2.screenshot(path=f"{FIXL}/shots_after/after_07_degraded.png")
    pg2.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    pg2.wait_for_timeout(300)
    pg2.screenshot(path=f"{FIXL}/shots_after/after_08_degraded_bottom.png")

    # --- T10: original VM manifest recovery (if landed) ---
    pg3 = b.new_page(viewport={"width": 1600, "height": 1000})
    pg3.goto(f"{BASE}/?manifest=/@fs{VMORIG}", wait_until="domcontentloaded")
    pg3.wait_for_timeout(7000)
    vbody = pg3.evaluate("() => document.body.innerText")
    has_video = pg3.evaluate("() => {const v=document.querySelector('video'); return v? v.readyState>=1 : false}")
    check("T10 VM manifest loads assets (stretch)", has_video, f"readyState>=1={has_video}")
    check("T10 loud manifest-relative notice (stretch)", ("manifest-relative" in vbody or "unreachable" in vbody), "scan")
    pg3.screenshot(path=f"{FIXL}/shots_after/after_09_vm_manifest.png")

    check("zero page errors", len(out["errors"]) == 0, out["errors"][:3])
    b.close()

with open(f"{FIXL}/manager_verify_result.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out["fps"], indent=1))
print("TOTAL:", sum(1 for c in out["checks"] if c["ok"]), "/", len(out["checks"]), "checks passed")
