#!/usr/bin/env python3
"""Track H webux3 audit: recon pass — load real bundle, screenshot, enumerate controls."""
import json, sys, time
from playwright.sync_api import sync_playwright

LANE = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/webux3_audit_20260716"
MANIFEST = f"{LANE}/manifest_fresh_wolv_local.json"
URL = f"http://127.0.0.1:5173/?manifest=/@fs{MANIFEST}"
HEADED = "--headed" in sys.argv

out = {"url": URL, "console": [], "page_errors": [], "requests_failed": []}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=not HEADED, args=["--use-angle=metal"] if HEADED else [])
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.on("console", lambda m: out["console"].append(f"[{m.type}] {m.text[:300]}"))
    page.on("pageerror", lambda e: out["page_errors"].append(str(e)[:300]))
    page.on("requestfailed", lambda r: out["requests_failed"].append(f"{r.url[:160]} :: {r.failure}"))
    t0 = time.time()
    page.goto(URL, wait_until="domcontentloaded")
    page.wait_for_timeout(1500)
    page.screenshot(path=f"{LANE}/shots/recon_00_initial.png")
    # wait longer for assets (mesh index is big)
    page.wait_for_timeout(6000)
    out["load_wall_s"] = round(time.time() - t0, 2)
    page.screenshot(path=f"{LANE}/shots/recon_01_after_8s.png")

    # enumerate interactive controls
    ctrl = page.evaluate("""() => {
      const els = [...document.querySelectorAll('button, input, select, [role=button], [role=checkbox], a, summary')];
      return els.map(e => ({
        tag: e.tagName.toLowerCase(),
        type: e.getAttribute('type'),
        text: (e.innerText || e.value || e.getAttribute('aria-label') || '').trim().slice(0,80),
        title: e.getAttribute('title'),
        disabled: e.disabled === true,
        visible: !!(e.offsetWidth || e.offsetHeight),
        rect: (r => ({x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height)}))(e.getBoundingClientRect()),
      }));
    }""")
    out["controls"] = ctrl
    out["body_text_head"] = page.evaluate("() => document.body.innerText.slice(0, 4000)")
    canvases = page.evaluate("() => [...document.querySelectorAll('canvas')].map(c => ({w:c.width,h:c.height,cw:c.clientWidth,ch:c.clientHeight}))")
    videos = page.evaluate("() => [...document.querySelectorAll('video')].map(v => ({dur:v.duration, ready:v.readyState, w:v.videoWidth, h:v.videoHeight, paused:v.paused}))")
    out["canvases"] = canvases
    out["videos"] = videos
    page.screenshot(path=f"{LANE}/shots/recon_02_full.png", full_page=True)
    browser.close()

with open(f"{LANE}/recon_result.json", "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps({k: out[k] for k in ["load_wall_s", "page_errors", "requests_failed", "canvases", "videos"]}, indent=1))
print("console lines:", len(out["console"]))
print("controls:", len(out["controls"]))
