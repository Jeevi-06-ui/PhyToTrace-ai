/**
 * End-to-end browser verification of PhytoTrace AI diagnostic repair.
 * Proves that real image uploads in the browser execute without errors,
 * capture network telemetry, and render the trained model overlays.
 */

const path = require("path");
const fs = require("fs");
const puppeteer = require(path.resolve(__dirname, "../frontend/node_modules/puppeteer"));

const SCREENSHOT_DIR = path.resolve(__dirname, "../reports/screenshots");
if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

async function runBrowserVerification() {
  console.log("===============================================================================");
  console.log(" PHYTOTRACE — REAL BROWSER END-TO-END DIAGNOSTIC TEST");
  console.log("===============================================================================");

  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });

  // Listen to network responses
  const networkLogs = [];
  page.on("response", async (response) => {
    const url = response.url();
    if (url.includes("/api/predict") || url.includes("/api/v1/analyze")) {
      const status = response.status();
      let bodyText = "";
      try {
        bodyText = await response.text();
      } catch (e) {
        bodyText = "<non-text>";
      }
      networkLogs.push({ url, status, body: bodyText });
      console.log(`[NETWORK CAPTURE] ${response.request().method()} ${url} -> HTTP ${status}`);
    }
  });

  page.on("pageerror", (err) => {
    console.error("[BROWSER CONSOLE ERROR]:", err.message);
  });

  try {
    // -------------------------------------------------------------------------
    // TEST 1: Real Image 1 (BLK_3_1002_PL010_H.JPG)
    // -------------------------------------------------------------------------
    const img1Path = path.resolve(__dirname, "../data/raw/original_photos/BLK_3_1002_PL010_H.JPG");
    console.log("\n[TEST 1] Testing Image 1:", path.basename(img1Path));
    console.log("Navigating to http://localhost:3000/analyze...");
    await page.goto("http://localhost:3000/analyze", { waitUntil: "domcontentloaded", timeout: 20000 });
    await new Promise((r) => setTimeout(r, 1000));

    // Find the file input
    const fileInput = await page.$("input[type=file]");
    if (!fileInput) throw new Error("File input element not found on /analyze page");
    await fileInput.uploadFile(img1Path);
    console.log("File uploaded into browser input element.");
    await new Promise((r) => setTimeout(r, 800));

    // Verify preview is displayed
    const previewImg = await page.$("img[alt='Selected leaf preview']");
    if (!previewImg) throw new Error("Selected leaf preview image did not render!");
    console.log("Leaf preview rendered in browser DOM.");

    // Click "Run Deep AI Diagnosis" button
    console.log("Clicking 'Run Deep AI Diagnosis' CTA button...");
    const buttons = await page.$$("button");
    let runButton = null;
    for (const b of buttons) {
      const text = await page.evaluate((el) => el.innerText, b);
      if (text && text.includes("Run Deep AI Diagnosis")) {
        runButton = b;
        break;
      }
    }
    if (!runButton) throw new Error("Could not find 'Run Deep AI Diagnosis' button!");
    await runButton.click();

    console.log("Waiting for inference execution & LayerToggleViewer rendering...");
    // Wait for the result viewer or error
    await page.waitForFunction(
      () => {
        const text = document.body.innerText || "";
        return text.includes("Export Agronomic Report") || text.includes("Diagnosis Execution Error");
      },
      { timeout: 35000 }
    );

    // Check for error banner
    const pageText = await page.evaluate(() => document.body.innerText);
    if (pageText.includes("Diagnosis Execution Error")) {
      const match = pageText.match(/Diagnosis Execution Error\s*([^\n]+)/);
      throw new Error(`Diagnosis Execution Error appeared on page: ${match ? match[1] : ""}`);
    }

    console.log("[TEST 1 SUCCESS] Inference completed without errors!");
    await new Promise((r) => setTimeout(r, 1500));

    // Save Screenshot 1
    const screenshot1Path = path.join(SCREENSHOT_DIR, "browser_verification_test1.png");
    await page.screenshot({ path: screenshot1Path, fullPage: false });
    console.log("Saved verification screenshot:", screenshot1Path);

    // Verify Network Capture for Test 1
    const test1Net = networkLogs[networkLogs.length - 1];
    if (!test1Net || test1Net.status !== 200) {
      throw new Error(`Expected HTTP 200 from prediction endpoint, got: ${JSON.stringify(test1Net)}`);
    }
    const test1Json = JSON.parse(test1Net.body);
    console.log("Image 1 Result - Inspection ID:", test1Json.inspection_id);
    console.log("Image 1 Result - Damage Percentage:", test1Json.damage_percentage + "%");
    console.log("Image 1 Result - Severity Tier:", test1Json.severity_tier);
    console.log("Image 1 Result - Detected Regions:", test1Json.damage_regions ? test1Json.damage_regions.length : 0);

    // -------------------------------------------------------------------------
    // TEST 2: Real Image 2 (BLK_3_1002_PL021_H.JPG)
    // -------------------------------------------------------------------------
    const img2Path = path.resolve(__dirname, "../data/raw/original_photos/BLK_3_1002_PL021_H.JPG");
    console.log("\n[TEST 2] Testing Image 2:", path.basename(img2Path));
    console.log("Reloading http://localhost:3000/analyze for fresh scan...");
    await page.goto("http://localhost:3000/analyze", { waitUntil: "domcontentloaded", timeout: 20000 });
    await new Promise((r) => setTimeout(r, 1000));

    const fileInput2 = await page.$("input[type=file]");
    await fileInput2.uploadFile(img2Path);
    console.log("Second file uploaded into browser input element.");
    await new Promise((r) => setTimeout(r, 800));

    const buttons2 = await page.$$("button");
    let runButton2 = null;
    for (const b of buttons2) {
      const text = await page.evaluate((el) => el.innerText, b);
      if (text && text.includes("Run Deep AI Diagnosis")) {
        runButton2 = b;
        break;
      }
    }
    await runButton2.click();
    console.log("Clicking 'Run Deep AI Diagnosis' for Image 2...");

    await page.waitForFunction(
      () => {
        const text = document.body.innerText || "";
        return text.includes("Export Agronomic Report") || text.includes("Diagnosis Execution Error");
      },
      { timeout: 35000 }
    );

    const pageText2 = await page.evaluate(() => document.body.innerText);
    if (pageText2.includes("Diagnosis Execution Error")) {
      const match = pageText2.match(/Diagnosis Execution Error\s*([^\n]+)/);
      throw new Error(`Diagnosis Execution Error appeared on Image 2: ${match ? match[1] : ""}`);
    }

    console.log("[TEST 2 SUCCESS] Inference completed for Image 2 without errors!");
    await new Promise((r) => setTimeout(r, 1500));

    // Save Screenshot 2
    const screenshot2Path = path.join(SCREENSHOT_DIR, "browser_verification_test2.png");
    await page.screenshot({ path: screenshot2Path, fullPage: false });
    console.log("Saved verification screenshot:", screenshot2Path);

    const test2Net = networkLogs[networkLogs.length - 1];
    const test2Json = JSON.parse(test2Net.body);
    console.log("Image 2 Result - Inspection ID:", test2Json.inspection_id);
    console.log("Image 2 Result - Damage Percentage:", test2Json.damage_percentage + "%");
    console.log("Image 2 Result - Severity Tier:", test2Json.severity_tier);
    console.log("Image 2 Result - Detected Regions:", test2Json.damage_regions ? test2Json.damage_regions.length : 0);

    console.log("\n===============================================================================");
    console.log(" ALL BROWSER WORKFLOW TESTS PASSED 100% SUCCESFULLY!");
    console.log("===============================================================================");
  } finally {
    await browser.close();
  }
}

runBrowserVerification().catch((err) => {
  console.error("\n[FATAL TEST FAILURE]:", err.message);
  process.exit(1);
});
