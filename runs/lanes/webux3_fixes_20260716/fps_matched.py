import json
from playwright.sync_api import sync_playwright
AUDIT = "/Users/arnavchokshi/Desktop/pickleball/runs/lanes/webux3_audit_20260716"
RICH = f"{AUDIT}/manifest_fresh_wolv_local.json"
BASE = "http://127.0.0.1:5173"
def fps_probe(page, secs=3.5):
    return page.evaluate("(s)=>new Promise(res=>{let n=0;const t0=performance.now();function tick(){n++;if(performance.now()-t0<s*1000)requestAnimationFrame(tick);else res(n/s);}requestAnimationFrame(tick);})", secs)
def click_preset(page, name):
    try: page.get_by_role("button", name=name, exact=True).first.click(timeout=3000)
    except Exception: page.get_by_text(name, exact=True).first.click(timeout=3000)
    page.wait_for_timeout(600)
res=[]
with sync_playwright() as p:
    b = p.chromium.launch(headless=False, args=["--use-angle=metal"])
    pg = b.new_page(viewport={"width":1600,"height":1000})
    pg.goto(f"{BASE}/?manifest=/@fs{RICH}", wait_until="domcontentloaded")
    pg.wait_for_timeout(7000)
    for seg in [0.2, 5.0, 0.2, 5.0]:
        pair={}
        for preset in ["Court","Follow player"]:
            click_preset(pg, preset)
            pg.evaluate(f"()=>{{const v=document.querySelector('video'); v.muted=true; v.loop=true; v.currentTime={seg}; v.play();}}")
            pg.wait_for_timeout(600)
            f=fps_probe(pg)
            st=pg.evaluate("()=>{const v=document.querySelector('video');return [v.paused,v.ended,v.currentTime];}")
            pair[preset]=round(f,1)
            print(f"seg={seg} {preset:14s} fps={f:6.1f} vstate={st}")
        pair["seg"]=seg; res.append(pair)
    b.close()
c=sum(r["Court"] for r in res)/len(res); f=sum(r["Follow player"] for r in res)/len(res)
print(json.dumps(res))
print(f"court_avg={c:.1f} follow_avg={f:.1f} ratio={f/c:.3f} PASS={f>=0.8*c}")
