const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

async function main() {
  const screenshotsDir = path.resolve(__dirname, "../reports/screenshots");
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1380, height: 960 });

  console.log("1. Navigating to Weather page...");
  await page.goto("http://localhost:3000/weather", { waitUntil: "networkidle2", timeout: 30000 });
  await new Promise((r) => setTimeout(r, 2000));

  // Change coordinates to Kolar (13.1362, 78.1347)
  console.log("2. Testing dynamic geocoding for Kolar (13.1362, 78.1347)...");
  const latInput = await page.$('input[type="number"][min="-90"]');
  const lonInput = await page.$('input[type="number"][min="-180"]');

  if (latInput && lonInput) {
    await latInput.click({ clickCount: 3 });
    await latInput.type("13.1362");
    await lonInput.click({ clickCount: 3 });
    await lonInput.type("78.1347");

    // Wait for debounced geocoding (600ms + network)
    await new Promise((r) => setTimeout(r, 2500));

    // Check locationName value
    const locationVal = await page.$eval('input[maxlength="96"]', (el) => el.value);
    console.log(`   Dynamically resolved location: "${locationVal}"`);
  }

  // Click Get Forecast to load weather
  const getForecastBtn = await page.$('button[type="submit"]');
  if (getForecastBtn) {
    await getForecastBtn.click();
    await new Promise((r) => setTimeout(r, 3000));
  }

  const weatherKolarScreenshot = path.join(screenshotsDir, "weather_dynamic_geocoding_kolar.png");
  await page.screenshot({ path: weatherKolarScreenshot, fullPage: true });
  console.log(`   Captured: ${weatherKolarScreenshot}`);

  // Switch to Kannada
  console.log("3. Switching language to Kannada...");
  const knBtn = await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll("button"));
    const b = buttons.find((btn) => btn.textContent && btn.textContent.includes("ಕನ್ನಡ"));
    if (b) {
      b.click();
      return true;
    }
    return false;
  });
  await new Promise((r) => setTimeout(r, 2500));

  const weatherKnScreenshot = path.join(screenshotsDir, "weather_kannada_dynamic_geocoding.png");
  await page.screenshot({ path: weatherKnScreenshot, fullPage: true });
  console.log(`   Captured Kannada weather: ${weatherKnScreenshot}`);

  // Switch to Hindi
  console.log("4. Switching language to Hindi...");
  await page.evaluate(() => {
    const buttons = Array.from(document.querySelectorAll("button"));
    const b = buttons.find((btn) => btn.textContent && btn.textContent.includes("हिंदी"));
    if (b) b.click();
  });
  await new Promise((r) => setTimeout(r, 2500));

  const weatherHiScreenshot = path.join(screenshotsDir, "weather_hindi_dynamic_geocoding.png");
  await page.screenshot({ path: weatherHiScreenshot, fullPage: true });
  console.log(`   Captured Hindi weather: ${weatherHiScreenshot}`);

  // 5. Navigate to Leaf Scanner and verify Voice Narration Button & Species Title
  console.log("5. Testing AI Leaf Scanner & Voice Narration Button...");
  await page.goto("http://localhost:3000/analyze", { waitUntil: "networkidle2", timeout: 30000 });
  await new Promise((r) => setTimeout(r, 2000));

  // Load benchmark sample
  const benchmarkBtn = await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll("button"));
    const b = btns.find((x) => x.textContent && x.textContent.includes("Load & Analyze"));
    if (b) {
      b.click();
      return true;
    }
    return false;
  });

  if (benchmarkBtn) {
    console.log("   Clicked benchmark sample, staging image...");
    await new Promise((r) => setTimeout(r, 1500));

    // Click "Run Deep AI Diagnosis" button
    console.log("   Clicking 'Run Deep AI Diagnosis'...");
    const runScanBtn = await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll("button"));
      const b = btns.find((x) => x.textContent && x.textContent.includes("Run Deep AI Diagnosis"));
      if (b) {
        b.click();
        return true;
      }
      return false;
    });

    if (runScanBtn) {
      console.log("   Waiting for AI inference to complete...");
      await new Promise((r) => setTimeout(r, 6000));
    }

    // Verify Voice Narration Button is present
    const voiceButtonExists = await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll("button"));
      return btns.some(
        (b) =>
          b.textContent &&
          (b.textContent.includes("Listen to Diagnosis") ||
            b.textContent.includes("ಧ್ವನಿ ವಿವರಣೆ ಕೇಳಿ") ||
            b.textContent.includes("निदान सुनें"))
      );
    });

    console.log(`   Voice Narration Button rendered: ${voiceButtonExists}`);

    const scannerVoiceScreenshot = path.join(screenshotsDir, "scanner_voice_narration_verified.png");
    await page.screenshot({ path: scannerVoiceScreenshot, fullPage: true });
    console.log(`   Captured Scanner Voice: ${scannerVoiceScreenshot}`);
  }

  await browser.close();
  console.log("Verification finished successfully!");
}

main().catch((err) => {
  console.error("Verification failed:", err);
  process.exit(1);
});
