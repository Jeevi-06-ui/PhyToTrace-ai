const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1080 });

  console.log('===> Loading http://localhost:3000/analyze ...');
  await page.goto('http://localhost:3000/analyze', { waitUntil: 'networkidle2', timeout: 30000 });

  // 1. Check Engine Telemetry
  const telemetryOnline = await page.evaluate(() => {
    const text = document.body.innerText;
    return text.includes('Engine Telemetry') && text.includes('Online');
  });
  console.log('Engine Telemetry is Online:', telemetryOnline);

  // 2. Upload Image.png
  const imagePath = '/Users/jeevithp/Downloads/Image.png';
  if (!fs.existsSync(imagePath)) {
    throw new Error('Image.png not found at ' + imagePath);
  }

  const fileInput = await page.$('input[type="file"]');
  await fileInput.uploadFile(imagePath);
  console.log('Uploaded Image.png');

  await page.waitForFunction(() => {
    const btn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Run Deep AI Diagnosis'));
    return btn && !btn.disabled;
  }, { timeout: 10000 });

  console.log('Clicking Run Deep AI Diagnosis...');
  const runBtn = await page.evaluateHandle(() => {
    return Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Run Deep AI Diagnosis'));
  });
  await runBtn.click();

  console.log('Waiting for AI Diagnosis result...');
  await page.waitForFunction(() => {
    const text = document.body.innerText;
    return text.includes('AI-Based Mixed Infestation Detection') || text.includes('Diagnosis Execution Error');
  }, { timeout: 20000 });

  const hasError = await page.evaluate(() => document.body.innerText.includes('Diagnosis Execution Error'));
  if (hasError) {
    const errorText = await page.evaluate(() => {
      const el = document.querySelector('.text-\\[\\#F17878\\]');
      return el ? el.innerText : 'Unknown error';
    });
    throw new Error('Encountered Diagnosis Execution Error: ' + errorText);
  }

  // Check diagnosis name and accuracy
  const diagnosisData = await page.evaluate(() => {
    const text = document.body.innerText;
    const matchAcc = text.match(/([0-9.]+)%\s*Accuracy/);
    const hasLiriomyza = text.includes('Liriomyza trifolii');
    const hasTuta = text.includes('Tuta absoluta');
    const hasMixed = text.includes('Mixed Infestation');
    return {
      accuracy: matchAcc ? matchAcc[1] : null,
      hasLiriomyza,
      hasTuta,
      hasMixed,
      containsSpeciesBreakdown: text.includes('Species Evidence Breakdown'),
    };
  });

  console.log('Diagnosis Data:', diagnosisData);

  const screenshotPath = '/Users/jeevithp/Desktop/PhyToTrace/reports/screenshots/user_issue_image_png_verified.png';
  fs.mkdirSync(path.dirname(screenshotPath), { recursive: true });
  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log('Saved screenshot to:', screenshotPath);

  // Also copy to artifacts directory
  const artifactPath = '/Users/jeevithp/.gemini/antigravity/brain/004b200c-2290-4550-9f8d-f64a7a1e3cca/screenshots/user_issue_image_png_verified.png';
  fs.copyFileSync(screenshotPath, artifactPath);

  await browser.close();
  console.log('Verification completed successfully!');
})().catch(err => {
  console.error('Test failed:', err);
  process.exit(1);
});

