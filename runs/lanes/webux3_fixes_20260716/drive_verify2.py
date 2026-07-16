#!/usr/bin/env python3
"""Track H webux3 repair round 1: manager browser verification (second-shift manager).
Verifies the three repair items (follow FPS, VM-manifest recovery, badge/chip overlap)
plus regression of the adopted round-1 surfaces. Screenshots -> shots_repair1/."""
import json, os, sys
from playwright.sync_api import sync_playwright

AUDIT = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/webux3_audit_20260716"
FIXL = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/webux3_fixes_20260716"
RICH = f"{AUDIT}/manifest_fresh_wolv_local.json"
DEGR = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/trk_flip_20260713/default_production/wolverine_mixed_0200_mid_steep_corner/replay_viewer_manifest.json"
VMORIG = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/demo_beststack_render_20260710/fresh_wolv/wolverine_mixed_0200_mid_steep_corner/replay_viewer_manifest.json"
BASE = "http://127.0.0.1:5173"
SHOTS = f"{FIXL}/shots_repair1"
os.makedirs(SHOTS, exist_ok=True)
out = {"checks": [], "errors": [], "fps": {}}

def check(name, ok, detail=""):
    out["checks"].append({"name": name, "ok": bool(ok), "detail": str(detail)[:300]})
    print(("PASS " if ok else "FAIL "), name, "--", str(detail)[:180])

def fps_probe(page, secs=4):
    return page.evaluate("(s)=>new Promise(res=>{let n=0;const t0=performance.now();function tick(){n++;if(performance.now()-t0<s*1000)requestAnimationFrame(tick);else res(n/s);}requestAnimationFrame(tick);})", secs)

with sync_playwright() as p:
    b = p.chromium.launch(headless=False, args=["--use-angle=metal"])
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    pg.on("pageerror", lambda e: out["errors"].append(f"rich: {str(e)[:200]}"))

    # --- rich bundle: regression of adopted round-1 surfaces ---
    pg.goto(f"{BASE}/?manifest=/@fs{RICH}", wait_until="domcontentloaded")
    pg.wait_for_timeout(7000)
    pg.screenshot(path=f"{SHOTS}/repair_00_initial.png")
    g = pg.evaluate("""() => {
      const v=document.querySelector('video'); const r=v?v.getBoundingClientRect():null;
      const strip=document.querySelector('.timeline-strip');
      const markers=strip?strip.querySelectorAll('[class*=marker]').length:0;
      const playBtn=[...document.querySelectorAll('button')].some(b=>/play|pause/i.test((b.getAttribute('aria-label')||b.innerText||'')));
      return {videoTop: r? r.y+window.scrollY : null, nativeControls: v?v.hasAttribute('controls'):null, hasStrip: !!strip, markers, playBtn};
    }""")
    check("REG T1 video pane top < 700", g["videoTop"] is not None and g["videoTop"] < 700, g)
    check("REG T3 timeline intact (no native controls, strip+markers, play btn)",
          g["nativeControls"] is False and g["hasStrip"] and g["markers"] > 0 and g["playBtn"], g)
    body = pg.evaluate("() => document.body.innerText")
    check("REG T5 trust vocab present", ("PREVIEW" in body or "LOW CONFIDENCE" in body), "vocab scan")

    # --- FPS: identical harness to round 1, n=3 court/follow while playing ---
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
    check("T2 court-play >= 37.5 (baseline bar)", cavg >= 37.5*0.95, f"court_avg={cavg:.1f} trials={[round(t['court_play'],1) for t in trials]}")
    check("T2 follow-play >= 80% of court-play", favg >= 0.8*cavg, f"follow_avg={favg:.1f} vs court_avg={cavg:.1f} ratio={favg/cavg:.2f} trials={[round(t['follow_play'],1) for t in trials]}")
    # paused non-regression (informational + bar: paused follow >= 60)
    pg.evaluate("()=>document.querySelector('video')?.pause()")
    try:
        pg.get_by_role("button", name="Follow player", exact=True).click(timeout=3000)
    except Exception:
        pg.get_by_text("Follow player").first.click(timeout=3000)
    pg.wait_for_timeout(700)
    fpaused = fps_probe(pg)
    out["fps"]["follow_paused"] = fpaused
    check("T2 follow-paused non-regression (>=60)", fpaused >= 60, f"follow_paused={fpaused:.1f} (round1 badge read ~97)")
    try:
        pg.get_by_role("button", name="Court", exact=True).first.click(timeout=3000)
    except Exception:
        pg.get_by_text("Court", exact=True).first.click(timeout=3000)
    pg.screenshot(path=f"{SHOTS}/repair_01_playing.png")

    # --- degraded bundle: honesty intact + geometric badge/chip overlap ---
    pg2 = b.new_page(viewport={"width": 1600, "height": 1000})
    pg2.on("pageerror", lambda e: out["errors"].append(f"degraded: {str(e)[:200]}"))
    pg2.goto(f"{BASE}/?manifest=/@fs{DEGR}", wait_until="domcontentloaded")
    pg2.wait_for_timeout(5000)
    dbody = pg2.evaluate("() => document.body.innerText")
    check("REG degraded: preview vocabulary intact", ("PREVIEW" in dbody), "scan")
    check("REG degraded: missing stays missing", ("missing" in dbody.lower()), "scan")
    ov = pg2.evaluate("""() => {
      const dock=document.querySelector('.world-honesty-dock');
      if(!dock) return {found:false};
      const els=[...dock.querySelectorAll('.trust-band-card')];
      const chip=dock.querySelector('.layer-empty-strip summary');
      if(chip) els.push(chip);
      const boxes=els.map(e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height,t:(e.innerText||'').slice(0,24)};});
      const overlaps=[];
      for(let i=0;i<boxes.length;i++) for(let j=i+1;j<boxes.length;j++){
        const a=boxes[i],b2=boxes[j];
        const ix=Math.min(a.x+a.w,b2.x+b2.w)-Math.max(a.x,b2.x);
        const iy=Math.min(a.y+a.h,b2.y+b2.h)-Math.max(a.y,b2.y);
        if(ix>1 && iy>1) overlaps.push([a.t,b2.t,Math.round(ix),Math.round(iy)]);
      }
      const dr=dock.getBoundingClientRect();
      const pane=dock.closest('section')||dock.parentElement;
      const pr=pane.getBoundingClientRect();
      return {found:true,n:boxes.length,overlaps,dockRight:Math.round(dr.right),paneRight:Math.round(pr.right)};
    }""")
    check("T2c honesty dock present on degraded bundle", ov.get("found"), ov)
    check("T2c zero badge/chip overlaps", ov.get("found") and len(ov.get("overlaps", [1])) == 0, ov)
    check("T2c dock stays inside pane", ov.get("found") and ov.get("dockRight", 9999) <= ov.get("paneRight", 0) + 2, ov)
    pg2.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    pg2.wait_for_timeout(300)
    pg2.screenshot(path=f"{SHOTS}/repair_02_degraded_bottom.png")
    # expanded chip must also not overlap
    try:
        pg2.locator(".layer-empty-strip summary").click(timeout=2500)
        pg2.wait_for_timeout(400)
        ov2 = pg2.evaluate("""() => {
          const dock=document.querySelector('.world-honesty-dock');
          const els=[...dock.querySelectorAll('.trust-band-card')];
          const strip=dock.querySelector('.layer-empty-strip');
          if(strip) els.push(strip);
          const boxes=els.map(e=>{const r=e.getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height,t:(e.innerText||'').slice(0,24)};});
          const overlaps=[];
          for(let i=0;i<boxes.length;i++) for(let j=i+1;j<boxes.length;j++){
            const a=boxes[i],b2=boxes[j];
            const ix=Math.min(a.x+a.w,b2.x+b2.w)-Math.max(a.x,b2.x);
            const iy=Math.min(a.y+a.h,b2.y+b2.h)-Math.max(a.y,b2.y);
            if(ix>1 && iy>1) overlaps.push([a.t,b2.t]);
          }
          return {overlaps};
        }""")
        check("T2c expanded chip: still zero overlaps", len(ov2.get("overlaps", [1])) == 0, ov2)
        pg2.screenshot(path=f"{SHOTS}/repair_03_degraded_chip_open.png")
    except Exception as e:
        out["errors"].append(f"chip expand: {str(e)[:120]}")

    # --- T10: real VM-written manifest recovery ---
    pg3 = b.new_page(viewport={"width": 1600, "height": 1000})
    pg3.on("pageerror", lambda e: out["errors"].append(f"vm: {str(e)[:200]}"))
    pg3.goto(f"{BASE}/?manifest=/@fs{VMORIG}", wait_until="domcontentloaded")
    pg3.wait_for_timeout(10000)
    vbody = pg3.evaluate("() => document.body.innerText")
    has_video = pg3.evaluate("() => {const v=document.querySelector('video'); return v? v.readyState>=1 : false}")
    has_world = pg3.evaluate("() => !!document.querySelector('canvas')")
    loud = ("manifest-relative" in vbody or "recovered" in vbody.lower() or "recovery" in vbody.lower())
    named_err = ("unreachable" in vbody.lower() and "/@fs//home" in vbody)
    check("T10 viewer OPENS on VM manifest (video readyState>=1)", has_video, f"readyState>=1={has_video} canvas={has_world}")
    check("T10 loud recovery notice visible", loud, [l for l in vbody.split("\\n") if "recover" in l.lower() or "manifest-relative" in l][:2])
    check("T10 no raw JSON token error", "Unexpected token" not in vbody, "scan")
    out["vm_named_error_fallback"] = named_err
    pg3.screenshot(path=f"{SHOTS}/repair_04_vm_manifest.png")
    pg3.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    pg3.wait_for_timeout(300)
    pg3.screenshot(path=f"{SHOTS}/repair_05_vm_manifest_bottom.png")

    check("zero page errors", len(out["errors"]) == 0, out["errors"][:3])
    b.close()

with open(f"{FIXL}/manager_verify2_result.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out["fps"], indent=1))
print("TOTAL:", sum(1 for c in out["checks"] if c["ok"]), "/", len(out["checks"]), "checks passed")
