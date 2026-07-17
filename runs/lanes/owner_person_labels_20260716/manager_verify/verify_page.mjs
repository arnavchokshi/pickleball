// Track F manager functional verification of the owner person-box labeling page.
// Run: npx playwright test? No — plain script: node manager_verify/verify_page.mjs
// Uses playwright's chromium. Every check prints PASS/FAIL; exit 1 on any FAIL.
import { chromium } from 'playwright';

const PAGE = 'file:///Users/arnavchokshi/Desktop/person_labels_20260716/START_HERE.html';
const LS_KEY = 'person_labels_20260716';
let failures = 0;
const check = (name, ok, extra = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}: ${name}${extra ? ' — ' + extra : ''}`);
  if (!ok) failures++;
};

const browser = await chromium.launch();
const ctx = await browser.newContext({ acceptDownloads: true });
const page = await ctx.newPage();
const consoleErrors = [];
page.on('pageerror', (e) => consoleErrors.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });

await page.goto(PAGE, { waitUntil: 'load' });
await page.waitForTimeout(1500);

// 1. loads w/o errors
check('page loads without JS errors', consoleErrors.length === 0, consoleErrors.slice(0, 2).join(' | '));

// 2. no native video
check('no <video> elements', (await page.locator('video').count()) === 0);

// 3. stratum/clip-name blindness in DOM
const dom = await page.content();
const leaks = ['scratch', 'stratum', 'spectator', 'IMG_', 'gameplay', 'empty_sparse']
  .filter((w) => dom.toLowerCase().includes(w.toLowerCase()));
check('DOM leaks no stratum/clip info', leaks.length === 0, 'leaked: ' + leaks.join(','));

// 4. a frame image renders
const imgOk = await page.evaluate(() => {
  const el = document.querySelector('canvas, img');
  if (!el) return false;
  const r = el.getBoundingClientRect();
  return r.width > 300 && r.height > 150;
});
check('frame image/canvas renders at size', imgOk);

// helper: read app state box count via localStorage after interactions
const lsState = async () => page.evaluate((k) => localStorage.getItem(k), LS_KEY);

// 5. draw a box with mouse in new-box mode (W)
await page.keyboard.press('w');
const target = page.locator('canvas, img').first();
const bb = await target.boundingBox();
await page.mouse.move(bb.x + bb.width * 0.3, bb.y + bb.height * 0.3);
await page.mouse.down();
await page.mouse.move(bb.x + bb.width * 0.45, bb.y + bb.height * 0.55, { steps: 8 });
await page.mouse.up();
await page.waitForTimeout(400);
let st = await lsState();
check('drawing a box autosaves to localStorage', !!st && st.length > 10);
const stateHasBox = st && /"(boxes|labels)"/.test(st);
check('saved state contains a boxes/labels field', !!stateHasBox);

// 6. class toggle + delete keys do not throw
await page.keyboard.press('c');
await page.keyboard.press('x');
await page.waitForTimeout(200);
check('class-toggle/delete keys no JS errors', consoleErrors.length === 0, consoleErrors.slice(-1).join(''));

// 7. navigation + progress counter
const progressBefore = await page.evaluate(() => document.body.innerText.match(/\d+\s*(\/|of)\s*\d+/)?.[0] || '');
const t0 = Date.now();
await page.keyboard.press('d');
await page.waitForTimeout(600);
const navMs = Date.now() - t0;
const progressAfter = await page.evaluate(() => document.body.innerText.match(/\d+\s*(\/|of)\s*\d+/)?.[0] || '');
check('progress counter exists', progressBefore !== '');
check('next-frame navigation updates progress', progressAfter !== progressBefore, `${progressBefore} -> ${progressAfter}`);
check('frame-to-frame nav <= 2s', navMs <= 2000, navMs + 'ms');
await page.keyboard.press('a');
await page.waitForTimeout(300);

// 8. empty-confirm key
await page.keyboard.press('d');
await page.waitForTimeout(300);
await page.keyboard.press('e');
await page.waitForTimeout(300);
st = await lsState();
check('empty-confirm persists', !!st && /empt/i.test(st));

// 9. reload restores state
await page.reload({ waitUntil: 'load' });
await page.waitForTimeout(1200);
const stAfter = await lsState();
check('reload preserves localStorage state', !!stAfter && stAfter.length > 10);
check('reload no JS errors', consoleErrors.length === 0, consoleErrors.slice(-1).join(''));

// 10. export produces valid JSON
let exportOk = false, exportParsed = null;
try {
  const dlPromise = page.waitForEvent('download', { timeout: 5000 });
  const btn = page.locator('button, a').filter({ hasText: /export/i }).first();
  await btn.click();
  const dl = await dlPromise;
  const path = await dl.path();
  const fs = await import('fs');
  exportParsed = JSON.parse(fs.readFileSync(path, 'utf8'));
  exportOk = true;
} catch (e) { exportOk = false; }
check('Export downloads valid JSON', exportOk, exportOk ? Object.keys(exportParsed).slice(0, 5).join(',') : 'no download/parse');

// 11. Save button exists and is big/obvious (width heuristic)
const saveBtn = page.locator('button').filter({ hasText: /save/i }).first();
const saveVisible = await saveBtn.isVisible().catch(() => false);
let saveBig = false;
if (saveVisible) { const b = await saveBtn.boundingBox(); saveBig = b && b.width >= 80 && b.height >= 32; }
check('Save button visible and large', saveVisible && saveBig);

await browser.close();
console.log(failures === 0 ? 'ALL CHECKS PASSED' : `${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
