import json
from playwright.sync_api import sync_playwright
IDX = "file:///Users/arnavchokshi/Desktop/visual_evidence_20260716/index.html"
res = {"errors": []}
with sync_playwright() as p:
    b = p.chromium.launch(headless=False)
    pg = b.new_page(viewport={"width": 1280, "height": 950})
    pg.on("pageerror", lambda e: res["errors"].append(str(e)[:150]))
    pg.goto(IDX, wait_until="load")
    pg.wait_for_timeout(2500)
    res["images"] = pg.evaluate("""() => [...document.images].map(i => ({src: i.src.split('/').pop(), ok: i.complete && i.naturalWidth > 0}))""")
    pg.evaluate("() => document.querySelector('video').play()")
    pg.wait_for_timeout(1500)
    res["video"] = pg.evaluate("""() => { const v = document.querySelector('video');
        return {ready: v.readyState, playing: !v.paused, dur: Math.round(v.duration*10)/10}; }""")
    res["viewer_link"] = pg.evaluate("() => document.querySelector('.viewerbox a').href")
    pg.screenshot(path="/Users/arnavchokshi/Desktop/pickleball/runs/lanes/trackH_evidpack_20260716/pack_top.png")
    pg.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    pg.wait_for_timeout(600)
    pg.screenshot(path="/Users/arnavchokshi/Desktop/pickleball/runs/lanes/trackH_evidpack_20260716/pack_bottom.png")
    b.close()
bad = [i for i in res["images"] if not i["ok"]]
print("images:", len(res["images"]), "loaded;", len(bad), "FAILED", bad)
print("video:", res["video"])
print("viewer link:", res["viewer_link"][:80])
print("page errors:", res["errors"])
print("VERDICT:", "PASS" if not bad and res["video"]["ready"] >= 2 and res["video"]["playing"] and not res["errors"] else "FAIL")
