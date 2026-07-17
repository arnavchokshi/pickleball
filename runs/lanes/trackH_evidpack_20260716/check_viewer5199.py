import json
from playwright.sync_api import sync_playwright
URL = "http://127.0.0.1:5199/?manifest=/@fs/Users/arnavchokshi/Desktop/pickleball/runs/lanes/webux3_audit_20260716/manifest_fresh_wolv_local.json"
OUT = "/Users/arnavchokshi/Desktop/visual_evidence_20260716/assets"
res = {"url": URL, "checks": []}
with sync_playwright() as p:
    b = p.chromium.launch(headless=False, args=["--use-angle=metal"])
    pg = b.new_page(viewport={"width": 1600, "height": 1000})
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:150]))
    pg.goto(URL, wait_until="domcontentloaded")
    pg.wait_for_timeout(8000)
    body = pg.evaluate("() => document.body.innerText")
    st = pg.evaluate("""() => ({
      signin: !!document.querySelector('[class*=sign], [class*=Sign]') && /sign\\s?in/i.test(document.body.innerText.slice(0,2000)),
      video: (()=>{const v=document.querySelector('video'); return v? v.readyState : -1})(),
      canvas: !!document.querySelector('canvas'),
      strip: !!document.querySelector('.timeline-strip'),
      badges: document.querySelectorAll('.trust-band-card').length
    })""")
    res["checks"].append({"replay_not_signin": (not st["signin"]) and st["video"] >= 1 and st["canvas"] and st["strip"], "detail": st})
    res["page_errors"] = errs
    pg.screenshot(path=f"{OUT}/world_now_live_viewer.png")
    b.close()
print(json.dumps(res, indent=1))
