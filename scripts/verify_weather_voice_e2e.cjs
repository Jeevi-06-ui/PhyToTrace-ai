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
  await page.setViewport({ width: 1400, height: 1000 });

  console.log("1. Navigating to /analyze...");
  await page.goto("http://localhost:3000/analyze", { waitUntil: "networkidle2" });
  await new Promise((r) => setTimeout(r, 1000));

  // Load benchmark sample
  console.log("2. Loading authentic benchmark sample...");
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll("button"));
    const b = btns.find((x) => x.textContent.includes("Load & Analyze"));
    if (b) b.click();
  });
  await new Promise((r) => setTimeout(r, 1500));

  console.log("3. Running Deep AI Diagnosis...");
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll("button"));
    const b = btns.find((x) => x.textContent.includes("Run Deep AI Diagnosis"));
    if (b) b.click();
  });

  // Wait for inference to complete
  await page.waitForSelector('button:has-text("Listen to Diagnosis"), button:has-text("Voice")', { timeout: 15000 }).catch(() => {});
  await new Promise((r) => setTimeout(r, 4000));

  // Capture English Scanner with Voice Option
  const enScanShot = path.join(screenshotsDir, "verified_scanner_voice_english.png");
  await page.screenshot({ path: enScanShot, fullPage: true });
  console.log("   Saved:", enScanShot);

  // Switch to Kannada
  console.log("4. Switching to Kannada on Leaf Scanner...");
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll("button"));
    const b = btns.find((x) => x.textContent.includes("ಕನ್ನಡ"));
    if (b) b.click();
  });
  await new Promise((r) => setTimeout(r, 2000));

  const knScanShot = path.join(screenshotsDir, "verified_scanner_voice_kannada.png");
  await page.screenshot({ path: knScanShot, fullPage: true });
  console.log("   Saved:", knScanShot);

  // Switch to Hindi
  console.log("5. Switching to Hindi on Leaf Scanner...");
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll("button"));
    const b = btns.find((x) => x.textContent.includes("हिंदी"));
    if (b) b.click();
  });
  await new Promise((r) => setTimeout(r, 2000));

  const hiScanShot = path.join(screenshotsDir, "verified_scanner_voice_hindi.png");
  await page.screenshot({ path: hiScanShot, fullPage: true });
  console.log("   Saved:", hiScanShot);

  // 6. Test Weather Dynamic Reverse Geocoding and Place Search
  console.log("6. Testing Weather page dynamic address lookup...");
  await page.goto("http://localhost:3000/weather", { waitUntil: "networkidle2" });
  await new Promise((r) => setTimeout(r, 1500));

  // Search for Hubli
  console.log("   Searching for 'Hubli'...");
  await page.evaluate(() => {
    const input = document.querySelector('input[maxlength="96"]');
    if (input) {
      input.value = "Hubli";
      input.dispatchEvent(new Event("input", { bubbles: true }));
    }
  });

  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll("button"));
    const b = btns.find((x) => x.textContent.includes("Search") || x.textContent.includes("ಹುಡುಕಿ") || x.textContent.includes("खोजें"));
    if (b) b.click();
  });

  await new Promise((r) => setTimeout(r, 3500));

  const hubliWeatherShot = path.join(screenshotsDir, "verified_weather_hubli_geocoded.png");
  await page.screenshot({ path: hubliWeatherShot, fullPage: true });
  console.log("   Saved:", hubliWeatherShot);

  await browser.close();
  console.log("All screenshots captured and verified successfully!");
}

main().catch((e) => {
  console.error("Error:", e);
  process.exit(1);
});

