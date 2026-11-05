const path = require("path");
const fs = require("fs");

const puppeteer = require(path.resolve(__dirname, "../frontend/node_modules/puppeteer"));

async function run() {
  console.log("Launching browser for Real Leaflet Damage Localization Verification...");
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
    ],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1024 });

  const artifactDir = "/Users/jeevithp/.gemini/antigravity/brain/004b200c-2290-4550-9f8d-f64a7a1e3cca/screenshots";
  if (!fs.existsSync(artifactDir)) {
    fs.mkdirSync(artifactDir, { recursive: true });
  }

  try {
    console.log("1. Navigating to http://localhost:3000/analyze ...");
    await page.goto("http://localhost:3000/analyze", { waitUntil: "networkidle2" });
    await new Promise((r) => setTimeout(r, 1000));

    console.log("2. Uploading /Users/jeevithp/Downloads/Image.png ...");
    const fileInput = await page.$("input[type=file][accept*='jpeg']");
    if (!fileInput) {
      throw new Error("File input not found!");
    }
    await fileInput.uploadFile("/Users/jeevithp/Downloads/Image.png");
    await new Promise((r) => setTimeout(r, 1500));

    console.log("3. Clicking 'Run Deep AI Diagnosis' for initial scan...");
    const runBtn = await page.evaluateHandle(() => {
      const btns = Array.from(document.querySelectorAll("button"));
      return btns.find((b) => b.textContent && (b.textContent.includes("Run Deep AI Diagnosis") || b.textContent.includes("Analyzing"))) || null;
    });

    if (!runBtn) throw new Error("Run button not found");
    await runBtn.asElement().click();

    // Wait for the analysis to complete
    console.log("Waiting for initial analysis completion...");
    await page.waitForFunction(
      () => {
        const text = document.body.innerText;
        return text.includes("Canopy Boundary Uncalibrated") || text.includes("DAMAGE_DETECTED");
      },
      { timeout: 30000 }
    );

    await new Promise((r) => setTimeout(r, 1500));
    const step1Shot = path.join(artifactDir, "real_leaflet_01_uncalibrated_banner.png");
    await page.screenshot({ path: step1Shot, fullPage: true });
    console.log("Saved Step 1 screenshot:", step1Shot);

    console.log("4. Finding and clicking 'Calibrate Boundary & Run Scan' button...");
    const calibrateBtn = await page.evaluateHandle(() => {
      const btns = Array.from(document.querySelectorAll("button"));
      return btns.find((b) => b.textContent && b.textContent.includes("Calibrate Boundary & Run Scan")) || null;
    });

    if (!calibrateBtn || !calibrateBtn.asElement()) {
      throw new Error("'Calibrate Boundary & Run Scan' button not found in banner!");
    }

    await calibrateBtn.asElement().click();
    console.log("Clicked Calibrate & Run. Waiting for deep diagnosis results...");

    await page.waitForFunction(
      () => {
        const text = document.body.innerText;
        return text.includes("DAMAGE_DETECTED") && !text.includes("Analyzing Leaf Canopy");
      },
      { timeout: 35000 }
    );

    await new Promise((r) => setTimeout(r, 2000));

    const finalShot = path.join(artifactDir, "real_leaflet_02_damage_detected.png");
    await page.screenshot({ path: finalShot, fullPage: true });
    console.log("Saved Step 2 screenshot:", finalShot);

    // Extract displayed metrics
    const metrics = await page.evaluate(() => {
      const body = document.body.innerText;
      return {
        hasDamageDetected: body.includes("DAMAGE_DETECTED"),
        hasManuallyAssisted: body.includes("Manually assisted analysis"),
        bodyTextSnippet: body.substring(0, 1500),
      };
    });

    console.log("Browser Verification Succeeded:", JSON.stringify(metrics, null, 2));
  } catch (err) {
    console.error("Test failed:", err);
    process.exit(1);
  } finally {
    await browser.close();
  }
}

run();

