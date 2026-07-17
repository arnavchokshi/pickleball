import json
from playwright.sync_api import sync_playwright
URL = ("http://127.0.0.1:5199/?manifest=/@fs/Users/arnavchokshi/Desktop/pickleball/runs/lanes/"
       "trackH_evidpack_20260716/one_world_viewer_manifest.json")
OUT = "/Users/arnavchokshi/Desktop/visual_evidence_20260716/assets"
res = {"errors": []}
with sync_playwright() as p:
    b = p.chromium.launch(headless=False, args=["--use-angle=metal"])
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    pg.on("pageerror", lambda e: res["errors"].append(str(e)[:200]))
    pg.goto(URL, wait_until="domcontentloaded")
    pg.wait_for_timeout(9000)
    res["state"] = pg.evaluate("""() => ({
      video: (()=>{const v=document.querySelector('video'); return v? v.readyState : -1})(),
      canvas: !!document.querySelector('canvas'),
      badges: document.querySelectorAll('.trust-band-card').length,
      body: document.body.innerText.slice(0, 1400)
    })""")
    pg.evaluate("()=>{const v=document.querySelector('video'); if(v){v.muted=true; v.currentTime=3.0;}}")
    pg.wait_for_timeout(2500)
    pg.screenshot(path=f"{OUT}/fused_world_viewer.png")
    b.close()
print("video readyState:", res["state"]["video"], "| canvas:", res["state"]["canvas"], "| badges:", res["state"]["badges"])
print("errors:", res["errors"][:2])
print("---BODY---")
print(res["state"]["body"][:900])
