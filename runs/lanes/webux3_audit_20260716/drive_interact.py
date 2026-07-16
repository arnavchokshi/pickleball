#!/usr/bin/env python3
"""Track H webux3 audit: interaction pass — exercise controls, measure FPS, screenshot each state."""
import json, sys, time
from playwright.sync_api import sync_playwright

LANE = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/webux3_audit_20260716"
MANIFEST = f"{LANE}/manifest_fresh_wolv_local.json"
URL = f"http://127.0.0.1:5173/?manifest=/@fs{MANIFEST}"
HEADLESS = "--headless" in sys.argv
TAG = "hl" if HEADLESS else "hd"

out = {"headless": HEADLESS, "steps": [], "console_errors": []}

def snap(page, name):
    page.screenshot(path=f"{LANE}/shots/int_{TAG}_{name}.png")

def fps_probe(page, seconds=4):
    return page.evaluate("""(secs) => new Promise(res => {
      let n = 0; const t0 = performance.now();
      function tick(){ n++; if (performance.now() - t0 < secs*1000) requestAnimationFrame(tick); else res(n/secs); }
      requestAnimationFrame(tick);
    })""", seconds)

def badge_fps(page):
    try:
        txt = page.evaluate("""() => {
          const els=[...document.querySelectorAll('div,span')];
          const hit=els.find(e=>e.childElementCount===0 && /^\\d+(\\.\\d+)?$/.test(e.innerText.trim()) && e.parentElement && /3D FPS/i.test(e.parentElement.innerText));
          return hit ? hit.innerText.trim() : null;
        }""")
        return txt
    except Exception:
        return None

with sync_playwright() as p:
    browser = p.chromium.launch(headless=HEADLESS, args=[] if HEADLESS else ["--use-angle=metal"])
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.on("pageerror", lambda e: out["console_errors"].append(str(e)[:200]))
    page.goto(URL, wait_until="domcontentloaded")
    page.wait_for_timeout(7000)

    def step(name, note=""):
        out["steps"].append({"name": name, "note": note, "badge_fps": badge_fps(page)})
        snap(page, name)

    step("00_loaded")

    # geometry: where are video, 3d canvas, timeline?
    geom = page.evaluate("""() => {
      const g = {};
      const v = document.querySelector('video'); if (v) g.video = v.getBoundingClientRect().toJSON();
      const c = document.querySelector('canvas'); if (c) g.canvas = c.getBoundingClientRect().toJSON();
      g.pageHeight = document.body.scrollHeight;
      const tl = [...document.querySelectorAll('div')].find(e => /rally/i.test(e.className||''));
      return g;
    }""")
    out["geometry"] = geom

    # 1. play video, measure fps while playing
    page.evaluate("() => { const v=document.querySelector('video'); if(v){v.muted=true; v.currentTime=0; v.play();} }")
    page.wait_for_timeout(1000)
    out["fps_playing_raf"] = fps_probe(page, 4)
    step("01_playing", f"raf_fps={out['fps_playing_raf']:.1f}")
    page.evaluate("() => document.querySelector('video')?.pause()")

    # 2. camera presets
    for preset in ["Follow player", "Free orbit", "Court"]:
        try:
            page.get_by_role("button", name=preset, exact=True).click(timeout=3000)
            page.wait_for_timeout(800)
            step(f"02_preset_{preset.replace(' ','_')}")
        except Exception as e:
            out["steps"].append({"name": f"02_preset_{preset}", "error": str(e)[:150]})

    # 3. entity toggles: solid meshes off, skeletons on; paddles off/on; ball trail off/on
    for label in ["Solid meshes", "Skeletons", "Paddles", "Ball trail", "Ghost positioning", "Contact surfaces", "Target zones", "Player trails"]:
        try:
            page.get_by_role("button", name=label, exact=True).first.click(timeout=2500)
            page.wait_for_timeout(400)
            step(f"03_toggle_{label.replace(' ','_')}")
        except Exception as e:
            out["steps"].append({"name": f"03_toggle_{label}", "error": str(e)[:150]})

    # 4. event nav
    try:
        page.get_by_role("button", name="Next Event", exact=True).click(timeout=2500)
        page.wait_for_timeout(500)
        page.get_by_role("button", name="Next Event", exact=True).click(timeout=2500)
        page.wait_for_timeout(500)
        step("04_next_event_x2")
    except Exception as e:
        out["steps"].append({"name": "04_next_event", "error": str(e)[:150]})

    # 5. isolate player / ball focus
    for label in ["Player 20", "Ball focus", "Clear"]:
        try:
            page.get_by_role("button", name=label, exact=True).click(timeout=2500)
            page.wait_for_timeout(500)
            step(f"05_isolate_{label.replace(' ','_')}")
        except Exception as e:
            out["steps"].append({"name": f"05_isolate_{label}", "error": str(e)[:150]})

    # 6. tabs: Shots, Court map, back to 3D
    for label in ["Shots", "Court map", "3D"]:
        try:
            page.get_by_role("button", name=label, exact=True).click(timeout=2500)
            page.wait_for_timeout(900)
            step(f"06_tab_{label.replace(' ','_')}")
        except Exception as e:
            out["steps"].append({"name": f"06_tab_{label}", "error": str(e)[:150]})

    # 7. playback rate toggle
    for label in ["original", "2x FPS (interpolated)"]:
        try:
            page.get_by_role("button", name=label, exact=True).click(timeout=2500)
            page.wait_for_timeout(400)
            step(f"07_playback_{label.split(' ')[0]}")
        except Exception as e:
            out["steps"].append({"name": f"07_playback_{label}", "error": str(e)[:150]})

    # 8. scroll to timeline, click 60% across the marker strip
    try:
        page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(400)
        step("08_bottom_scrolled")
    except Exception as e:
        out["steps"].append({"name": "08_bottom", "error": str(e)[:150]})

    # 9. fps while playing with meshes at follow preset (worst case)
    try:
        page.get_by_role("button", name="Follow player", exact=True).click(timeout=2500)
        page.evaluate("() => { const v=document.querySelector('video'); if(v){v.muted=true; v.currentTime=1; v.play();} }")
        page.wait_for_timeout(800)
        out["fps_follow_playing_raf"] = fps_probe(page, 4)
        out["badge_after_follow_play"] = badge_fps(page)
        step("09_follow_playing", f"raf={out['fps_follow_playing_raf']:.1f}")
    except Exception as e:
        out["steps"].append({"name": "09_follow_playing", "error": str(e)[:150]})

    browser.close()

with open(f"{LANE}/interact_{TAG}_result.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({"headless": HEADLESS, "fps_playing_raf": out.get("fps_playing_raf"), "fps_follow": out.get("fps_follow_playing_raf"), "badge_last": out.get("badge_after_follow_play"), "errors": out["console_errors"], "geometry": out.get("geometry")}, indent=1))
for s in out["steps"]:
    print(s.get("name"), s.get("badge_fps"), s.get("error", ""))
