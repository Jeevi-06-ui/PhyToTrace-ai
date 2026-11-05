const p = require("../frontend/node_modules/puppeteer");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const out = "reports/evidence_repair";
const failed = "reports/debug_segmentation/benchmark_whatsapp_orig.jpg";
const vertices = [[215,168],[264,171],[298,186],[307,210],[309,229],[293,253],[272,244],
  [259,268],[265,289],[247,294],[266,326],[265,347],[251,367],[227,377],[198,399],
  [173,385],[155,368],[151,345],[142,329],[138,314],[151,285],[155,267],[123,280],
  [123,265],[143,248],[143,222],[151,195],[174,178]];

(async () => {
  const browser = await p.launch();
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 1100 });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    const results = {};
    for (const [name, file, manual] of [
      ["whatsapp-auto", failed, false], ["whatsapp-polygon", failed, true],
      ["reviewed-positive", "data/raw/original_photos/5558326.jpg", false],
      ["reviewed-negative", "data/raw/original_photos/BLK_3_1002_PL010_H.JPG", false],
    ]) {
      await page.goto("http://localhost:3007/analyze", { waitUntil: "networkidle0" });
      await (await page.$('input[type="file"]')).uploadFile(path.resolve(file));
      const img = await page.waitForSelector('img[alt="Selected leaf preview"]');
      await page.waitForFunction(() => document.querySelector('img[alt="Selected leaf preview"]').naturalWidth > 0);
      if (manual) {
        await page.locator('button::-p-text(Outline target leaflet)').click();
        await img.evaluate(el => el.scrollIntoView({ block: "center" }));
        const rect = await img.boundingBox();
        for (const [x, y] of vertices) await page.mouse.click(rect.x + rect.width * x / 420, rect.y + rect.height * y / 560);
        await page.locator('button::-p-text(Use outline)').click();
        assert.equal(await page.$$eval('svg[aria-label="Selected leaflet outline"] circle', els => els.length), vertices.length);
        await page.screenshot({ path: path.join(out, "selected-before-inference.png"), fullPage: true });
      }
      const [response] = await Promise.all([
        page.waitForResponse(r => r.url().endsWith("/api/v1/analyze"), { timeout: 60000 }),
        page.locator('button::-p-text(Run Deep AI Diagnosis)').click(),
      ]);
      assert.equal(response.status(), 200);
      const result = await response.json();
      assert.equal(result.damage_percentage, null);
      assert.equal(result.damage_regions.length, 0);
      if (manual) {
        assert.equal(result.roi_mode, "manual_polygon");
        assert.equal(result.leaf_polygon.length, vertices.length);
        assert.equal(result.leaf_boundary_source, "user_polygon");
        const [x1,y1,x2,y2] = result.crop_box;
        for (const c of result.rejected_candidates)
          for (const poly of c.raw_polygons)
            for (const [x,y] of poly) assert.ok(x >= x1 && x < x2 && y >= y1 && y < y2);
      }
      await page.waitForFunction(() => document.body.innerText.includes("Measurement Unavailable"));
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: path.join(out, name + ".png"), fullPage: true });
      const { image_urls, ...summary } = result;
      results[name] = summary;
      console.log(name, result.failure_classification, result.diagnostic_telemetry.mine_segmenter);
    }
    fs.writeFileSync(path.join(out, "browser_errors.json"), JSON.stringify(errors));
    fs.writeFileSync(path.join(out, "browser_results.json"), JSON.stringify(results));
    assert.deepEqual(errors, []);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
