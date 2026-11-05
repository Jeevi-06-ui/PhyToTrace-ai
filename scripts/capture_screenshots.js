const path = require("path");
const fs = require("fs");

// Require puppeteer from frontend directory
const puppeteer = require(path.resolve(__dirname, "../frontend/node_modules/puppeteer"));

const SCREENSHOT_DIR = path.resolve(__dirname, "../reports/screenshots");
if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

async function capture() {
  console.log("Launching headless Chrome for Visual QA screenshots...");
  const browser = await puppeteer.launch({
    headless: true,
    args: ["--no-sandbox", "--disable-setuid-sandbox"],
  });

  const page = await browser.newPage();

  // Desktop viewport: 1440x900
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2 });

  // 1. Landing Page
  console.log("1. Capturing Landing Page...");
  await page.goto("http://localhost:3000/", { waitUntil: "domcontentloaded", timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1200));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "01_landing_page.png"), fullPage: false });

  // 2. Dashboard
  console.log("2. Capturing Farmer Dashboard...");
  await page.goto("http://localhost:3000/dashboard", { waitUntil: "domcontentloaded", timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1200));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "02_dashboard.png"), fullPage: false });

  // 3. AI Scanner Before Upload
  console.log("3. Capturing AI Scanner before upload...");
  await page.goto("http://localhost:3000/analyze", { waitUntil: "domcontentloaded", timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1200));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "03_ai_scanner_before_upload.png"), fullPage: false });

  // 4. AI Scanner After Inference (using real recorded inspection)
  console.log("4. Capturing AI Scanner with genuine diagnostic results...");
  await page.goto("http://localhost:3000/analyze?id=INSP_20261104_231538_56129_PLANT_TOMATO_A07", {
    waitUntil: "domcontentloaded",
    timeout: 15000,
  });
  await new Promise((r) => setTimeout(r, 1800));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "04_ai_scanner_after_inference.png"), fullPage: false });

  // 5. Inspection History
  console.log("5. Capturing Inspection History...");
  await page.goto("http://localhost:3000/history", { waitUntil: "domcontentloaded", timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1200));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "05_inspection_history.png"), fullPage: false });

  // 6. Research Dashboard
  console.log("6. Capturing Research Dashboard...");
  await page.goto("http://localhost:3000/research", { waitUntil: "domcontentloaded", timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1200));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "06_research_dashboard.png"), fullPage: false });

  // 7. Mobile Layout (390x844)
  console.log("7. Capturing Mobile Layout...");
  await page.setViewport({ width: 390, height: 844, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
  await page.goto("http://localhost:3000/analyze?id=INSP_20261104_231538_56129_PLANT_TOMATO_A07", {
    waitUntil: "domcontentloaded",
    timeout: 15000,
  });
  await new Promise((r) => setTimeout(r, 1500));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "07_mobile_layout.png"), fullPage: false });

  // Reset to desktop viewport for weather & assistant
  await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 2, isMobile: false, hasTouch: false });

  // 8. Weather Intelligence
  console.log("8. Capturing Weather Intelligence...");
  await page.goto("http://localhost:3000/weather", { waitUntil: "domcontentloaded", timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1500));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "08_weather_intelligence.png"), fullPage: false });

  // 9. Agri Assistant
  console.log("9. Capturing Agri Assistant...");
  await page.goto("http://localhost:3000/assistant", { waitUntil: "domcontentloaded", timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1500));
  await page.screenshot({ path: path.join(SCREENSHOT_DIR, "09_agri_assistant.png"), fullPage: false });

  await browser.close();
  console.log("==> All screenshots captured successfully in reports/screenshots/!");
}

capture().catch((err) => {
  console.error("Screenshot capture error:", err);
  process.exit(1);
});
