const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const puppeteer = require("../frontend/node_modules/puppeteer");

const repo = path.resolve(__dirname, "..");
const base = process.env.PHYTOTRACE_BROWSER_URL || "http://localhost:3007";
const out = path.join(repo, "reports/astra_cv/browser");
const greenhouse = "data/inspections/raw/INSP_20261105_024600_44601_PLANT_TOMATO_A07_orig.jpg";

(async () => {
  fs.mkdirSync(out, { recursive: true });
  const browser = await puppeteer.launch({ headless: true });
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1440, height: 1100 });
    const errors = [];
    page.on("pageerror", (error) => errors.push(error.message));
    const scans = {};
    async function scan(name, photo, manual = false) {
      await page.goto(`${base}/analyze`, { waitUntil: "networkidle0" });
      await (await page.$('input[type=file]')).uploadFile(path.join(repo, photo));
      await page.waitForSelector('img[alt="Selected leaf preview"]');
      if (manual) {
        await page.locator('button::-p-text(Focus Leaf ROI (Crop))').click();
        const bounds = await (await page.$('img[alt="Selected leaf preview"]')).boundingBox();
        const x = (f) => bounds.x + bounds.width * f;
        const y = (f) => bounds.y + bounds.height * f;
        await page.mouse.move(x(0.27), y(0.29));
        await page.mouse.down();
        await page.mouse.move(x(0.75), y(0.73), { steps: 10 });
        await page.mouse.up();
        await page.waitForFunction(() => document.body.innerText.includes("Manual Focus ROI: ["));
      }
      const [response] = await Promise.all([
        page.waitForResponse((r) => /\/api\/(predict|v1\/analyze)$/.test(r.url()) && r.request().method() === "POST", { timeout: 90000 }),
        page.locator('button::-p-text(Run Deep AI Diagnosis)').click(),
      ]);
      assert.equal(response.status(), 200);
      const result = await response.json();
      const { image_urls, ...summary } = result;
      scans[name] = summary;
      assert.equal(result.inspection_id, undefined, "guest scan must not be persisted");
      await page.waitForFunction((state) => document.body.innerText.includes(state), {}, result.failure_classification);
      if (manual) {
        assert.equal(result.roi_mode, "manual_crop");
        const [x1, y1, x2, y2] = result.crop_box;
        assert.ok(x1 > 750 && y1 > 1100 && x2 < 2400 && y2 < 3100);
        assert.ok(result.rejected_candidates.length > 0);
        for (const candidate of result.rejected_candidates) {
          for (const polygon of candidate.raw_polygons) {
            for (const [x, y] of polygon) assert.ok(x >= x1 && x < x2 && y >= y1 && y < y2);
          }
        }
        for (const r of result.damage_regions) {
          for (const [x, y] of r.polygon) assert.ok(x >= x1 && x < x2 && y >= y1 && y < y2);
        }
      }
      if (result.damage_regions.length) {
        assert.ok(result.damage_regions[0].label);
        await page.waitForSelector('svg[aria-label="Model evidence outlines"] polygon');
        await page.locator('svg[aria-label="Model evidence outlines"] polygon').click();
        await page.waitForFunction(() => document.body.innerText.includes("Experimental model prediction"));
      }
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: path.join(out, `${name}.png`), fullPage: true });
    }
    await scan("reviewed-positive", "data/raw/original_photos/5558326.jpg");
    await scan("greenhouse-manual", greenhouse, true);
    await page.setViewport({ width: 390, height: 844, isMobile: true, hasTouch: true });
    for (const [locale, title] of [["en", "AI Leaf Scanner"], ["kn", "AI ಎಲೆ ಸ್ಕ್ಯಾನರ್"], ["hi", "AI पत्ती स्कैनर"]]) {
      await page.evaluate((language) => localStorage.setItem("phytotrace_locale", language), locale);
      await page.goto(`${base}/analyze`, { waitUntil: "networkidle0" });
      await page.waitForFunction((language, heading) => document.documentElement.lang === language && document.querySelector("h1")?.textContent.includes(heading), {}, locale, title);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.screenshot({ path: path.join(out, `mobile-${locale}.png`) });
    }

    await page.setViewport({ width: 1440, height: 1100 });
    await page.evaluate(() => localStorage.setItem("phytotrace_locale", "en"));
    await browser.defaultBrowserContext().setPermission(base, { permission: { name: "geolocation" }, state: "granted" });
    await page.setGeolocation({ latitude: 12.9716, longitude: 77.5946 });
    await page.goto(`${base}/weather`, { waitUntil: "networkidle0" });
    const [weatherResponse] = await Promise.all([
      page.waitForResponse((r) => r.url().includes("/api/v1/weather/current?"), { timeout: 25000 }),
      page.locator('button::-p-text(Use My Current Location)').click(),
    ]);
    assert.equal(weatherResponse.status(), 200);
    const weather = await weatherResponse.json();
    assert.ok(["Google Weather API", "Open-Meteo"].includes(weather.source));
    assert.equal(weather.latitude, 12.9716);
    await page.waitForFunction((provider) => document.body.innerText.includes(provider), {}, weather.source);
    await page.screenshot({ path: path.join(out, "live-weather.png"), fullPage: true });
    fs.writeFileSync(path.join(out, "weather.json"), JSON.stringify(weather, null, 2));
    fs.writeFileSync(path.join(out, "responses.json"), JSON.stringify({ scans, errors }, null, 2));
    console.log(Object.fromEntries(Object.entries(scans).map(([name, scan]) => [
      name, { state: scan.failure_classification, percent: scan.damage_percentage, regions: scan.damage_regions.length, crop: scan.crop_box },
    ])));
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
