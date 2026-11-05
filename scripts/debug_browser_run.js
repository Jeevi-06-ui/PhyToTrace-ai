const path = require("path");
const puppeteer = require(path.resolve(__dirname, "../frontend/node_modules/puppeteer"));

(async () => {
  const browser = await puppeteer.launch({ headless: true });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  await page.goto("http://localhost:3000/analyze", { waitUntil: "domcontentloaded" });
  await new Promise(r => setTimeout(r, 1000));

  const fileInput = await page.$('input[type="file"]');
  await fileInput.uploadFile(path.resolve(__dirname, "../data/inspections/raw/INSP_20261105_024600_44601_PLANT_TOMATO_A07_orig.jpg"));
  await new Promise(r => setTimeout(r, 1000));
  
  const buttons = await page.$$("button");
  for (const b of buttons) {
    const text = await page.evaluate(el => el.textContent, b);
    if (text && text.includes("Run Deep AI Diagnosis")) {
      await b.click();
      break;
    }
  }

  await new Promise(r => setTimeout(r, 6000));
  const fullText = await page.evaluate(() => document.body.innerText);

  console.log("--- SEARCHING IN FULL TEXT ---");
  console.log("Contains 'No Visible Mine':", fullText.toLowerCase().includes("no visible mine"));
  console.log("Contains '0.00%':", fullText.includes("0.00%"));
  console.log("Contains 'No Serpentine':", fullText.includes("No Serpentine"));
  console.log("Contains '1,864,438':", fullText.includes("1,864,438"));

  const idx = fullText.indexOf("Export Agronomic Report");
  if (idx !== -1) {
    console.log("\n--- SLICE OF DIAGNOSTICS SECTION ---");
    console.log(fullText.slice(idx, idx + 1500));
  }

  const ssPath = path.resolve(__dirname, "../reports/debug_segmentation/16_browser_completed_diagnosis.png");
  await page.screenshot({ path: ssPath, fullPage: true });
  console.log("Saved full-page screenshot to:", ssPath);

  await browser.close();
})();
