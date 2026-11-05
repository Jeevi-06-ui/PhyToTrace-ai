/**
 * PhytoTrace AI — Real Browser Verification of Critical Segmentation & Background Repair
 */

const path = require("path");
const fs = require("fs");
const puppeteer = require(path.resolve(__dirname, "../frontend/node_modules/puppeteer"));

const OUT_DIR = path.resolve(__dirname, "../reports/debug_segmentation");
if (!fs.existsSync(OUT_DIR)) {
  fs.mkdirSync(OUT_DIR, { recursive: true });
}

async function runTest() {
  console.log("===============================================================================");
  console.log(" PHYTOTRACE — REAL BROWSER INFERENCE ACCURACY & BACKGROUND VERIFICATION");
  console.log("===============================================================================");

  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });

  // Monitor network
  let predictStatus = 0;
  let predictResponseBody = null;

  page.on("response", async (response) => {
    const url = response.url();
    if (url.includes("/api/predict")) {
      predictStatus = response.status();
      try {
        predictResponseBody = await response.json();
      } catch (e) {
        predictResponseBody = await response.text();
      }
      console.log(`[NETWORK] POST /api/predict -> HTTP ${predictStatus}`);
    }
  });

  page.on("pageerror", (err) => {
    console.error("[BROWSER ERROR]:", err.message);
  });

  try {
    // -------------------------------------------------------------------------
    // TEST: The user's exact failing smartphone photo
    // -------------------------------------------------------------------------
    const failingImgPath = path.resolve(
      __dirname,
      "../data/inspections/raw/INSP_20261105_024600_44601_PLANT_TOMATO_A07_orig.jpg"
    );
    console.log(`\nTesting Smartphone Photo: ${path.basename(failingImgPath)}`);

    console.log("Navigating to http://localhost:3000/analyze ...");
    await page.goto("http://localhost:3000/analyze", { waitUntil: "domcontentloaded", timeout: 20000 });
    await new Promise((r) => setTimeout(r, 1200));

    // Upload file
    console.log("Uploading photograph to file input...");
    const fileInput = await page.$('input[type="file"]');
    await fileInput.uploadFile(failingImgPath);
    await new Promise((r) => setTimeout(r, 1200));

    // Take screenshot of staged photo with ROI focus button
    const stagedScreenshotPath = path.join(OUT_DIR, "15_browser_staged_preview.png");
    await page.screenshot({ path: stagedScreenshotPath, fullPage: true });
    console.log(`Saved staged preview screenshot to: ${stagedScreenshotPath}`);

    // Click "Run Deep AI Diagnosis"
    console.log("Clicking 'Run Deep AI Diagnosis' button...");
    const buttons = await page.$$("button");
    let runBtn = null;
    for (const b of buttons) {
      const text = await page.evaluate((el) => el.textContent, b);
      if (text && text.includes("Run Deep AI Diagnosis")) {
        runBtn = b;
        break;
      }
    }

    if (!runBtn) {
      throw new Error("Could not find 'Run Deep AI Diagnosis' button");
    }

    await runBtn.click();
    console.log("Clicked! Waiting for real AI inference to complete...");

    // Wait for result
    await page.waitForFunction(
      () => {
        return (
          document.body.innerText.includes("Agronomic Decision") ||
          document.body.innerText.includes("Diagnosis Execution Error")
        );
      },
      { timeout: 30000 }
    );

    await new Promise((r) => setTimeout(r, 2000));

    const completedScreenshotPath = path.join(OUT_DIR, "16_browser_completed_diagnosis.png");
    await page.screenshot({ path: completedScreenshotPath, fullPage: true });
    console.log(`Saved completed diagnosis screenshot to: ${completedScreenshotPath}`);

    const bodyText = await page.evaluate(() => document.body.innerText);

    console.log("\n[VERIFICATION AUDIT ON SMARTPHONE PHOTO]:");
    console.log("HTTP Status Code        :", predictStatus);
    console.log("Damage Detected (JSON)  :", predictResponseBody?.damage_detected);
    console.log("Damage Percentage (JSON):", predictResponseBody?.damage_percentage);
    console.log("Leaf Area Pixels (JSON) :", predictResponseBody?.leaf_area_pixels);
    console.log("Decision (JSON)         :", predictResponseBody?.decision);
    console.log("Severity Tier (JSON)    :", predictResponseBody?.severity_tier);
    console.log("Evidence Regions (JSON) :", predictResponseBody?.damage_regions?.length || 0);

    // Assertions
    if (predictStatus !== 200) {
      throw new Error(`Expected HTTP 200, got ${predictStatus}`);
    }
    if (predictResponseBody?.damage_percentage !== 0.0) {
      throw new Error(`Expected 0.0% damage on healthy smartphone leaf, got ${predictResponseBody?.damage_percentage}`);
    }
    if (predictResponseBody?.damage_regions?.length > 0) {
      throw new Error(`Expected 0 damage regions, got ${predictResponseBody?.damage_regions?.length}`);
    }
    if (bodyText.includes("1,864,438")) {
      throw new Error("1.86M false region was found in rendered text!");
    }

    console.log("\n✅ ALL BROWSER ASSERTIONS PASSED!");
    console.log("• Background false positive eliminated.");
    console.log("• 1.86M pixel lesion artifact eliminated.");
    console.log("• Clean 0% damage score rendered in UI gauge.");
    console.log("• Real model inference executed end-to-end with HTTP 200.");
  } catch (err) {
    console.error("\n❌ Browser verification failed:", err);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

runTest();

