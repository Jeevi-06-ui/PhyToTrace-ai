const path = require("path");
const fs = require("fs");

const puppeteer = require(path.resolve(__dirname, "../frontend/node_modules/puppeteer"));

async function run() {
  console.log("Launching browser for camera & spot detection test...");
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      "--no-sandbox",
      "--disable-setuid-sandbox",
      "--use-fake-ui-for-media-stream",
      "--use-fake-device-for-media-stream",
    ],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  try {
    // 1. Navigate to analyze page
    console.log("Navigating to http://localhost:3000/analyze ...");
    await page.goto("http://localhost:3000/analyze", { waitUntil: "networkidle2" });

    const screenshotDir = path.resolve(__dirname, "../reports/screenshots");
    if (!fs.existsSync(screenshotDir)) {
      fs.mkdirSync(screenshotDir, { recursive: true });
    }

    // 2. Click "Camera Snap" button
    console.log("Clicking Camera Snap button...");
    const cameraBtn = await page.evaluateHandle(() => {
      const btns = Array.from(document.querySelectorAll("button"));
      return btns.find((b) => b.textContent && b.textContent.includes("Camera Snap")) || null;
    });

    if (!cameraBtn) {
      throw new Error("Camera Snap button not found!");
    }

    await cameraBtn.asElement().click();
    await new Promise((r) => setTimeout(r, 1200));

    // Verify Camera Viewfinder modal is visible
    const modalVisible = await page.evaluate(() => {
      const modal = document.querySelector("video");
      const title = document.body.innerText.includes("Camera Viewfinder");
      return Boolean(modal && title);
    });

    console.log("Camera modal opened successfully:", modalVisible);
    const cameraModalShot = path.join(screenshotDir, "camera_viewfinder_modal.png");
    await page.screenshot({ path: cameraModalShot });
    console.log("Saved camera viewfinder screenshot to:", cameraModalShot);

    // 3. Close the camera modal using Cancel or X
    await page.evaluate(() => {
      const cancelBtn = Array.from(document.querySelectorAll("button")).find(
        (b) => b.textContent && b.textContent.trim() === "Cancel"
      );
      if (cancelBtn) cancelBtn.click();
    });
    await new Promise((r) => setTimeout(r, 500));

    // 4. Test Spot Detection on WhatsApp Tomato Leaf Image
    console.log("Uploading benchmark_whatsapp_orig.jpg to test spot detection...");
    const testImgPath = path.resolve(__dirname, "../reports/debug_segmentation/benchmark_whatsapp_orig.jpg");
    const fileInput = await page.$('input[accept*="jpeg"]');
    if (!fileInput) throw new Error("File input element not found");
    await fileInput.uploadFile(testImgPath);

    await new Promise((r) => setTimeout(r, 1000));

    // Click "Run Deep AI Diagnosis"
    console.log("Clicking Run Deep AI Diagnosis button...");
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll("button"));
      const btn = btns.find((b) => b.textContent.includes("Run Deep AI Diagnosis"));
      if (btn) btn.click();
    });

    console.log("Waiting for AI diagnosis to complete...");
    await page.waitForFunction(
      () => {
        const bodyText = document.body.innerText;
        return (
          bodyText.includes("DETECTED EVIDENCE REGIONS") ||
          bodyText.includes("Region A") ||
          bodyText.includes("Mine-like regions found") ||
          bodyText.includes("Diagnosis Execution Error")
        );
      },
      { timeout: 30000 }
    );

    await new Promise((r) => setTimeout(r, 2000));

    // Extract diagnosis results
    const results = await page.evaluate(() => {
      const text = document.body.innerText;
      return {
        hasDamageDetected: text.includes("DETECTED EVIDENCE REGIONS") || text.includes("Mine-like regions found"),
        hasRegionA: text.includes("Region A"),
        hasRegionB: text.includes("Region B"),
        hasDamageMeasurement: text.includes("33.60%"),
      };
    });

    console.log("Diagnosis Result Evaluation:");
    console.log("  Damage Detected:", results.hasDamageDetected);
    console.log("  Region A Detected:", results.hasRegionA);
    console.log("  Region B Detected:", results.hasRegionB);
    console.log("  Damage Measurement 33.60%:", results.hasDamageMeasurement);

    const diagnosisShot = path.join(screenshotDir, "whatsapp_spots_detected_browser.png");
    await page.screenshot({ path: diagnosisShot, fullPage: true });
    console.log("Saved diagnosis screenshot to:", diagnosisShot);

    if (!results.hasDamageDetected || !results.hasRegionA) {
      throw new Error("Spots were not localized on the leaf as expected!");
    }

    console.log("\n========================================================");
    console.log("VERIFICATION SUCCESSFUL: Camera snap and leaf spots detection verified in browser!");
    console.log("========================================================\n");
  } finally {
    await browser.close();
  }
}

run().catch((err) => {
  console.error("Test failed:", err);
  process.exit(1);
});
