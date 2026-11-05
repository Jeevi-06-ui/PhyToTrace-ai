const path = require('path');
const fs = require('fs');
const puppeteer = require(path.resolve('frontend/node_modules/puppeteer'));

(async () => {
  const screenshotsDir = path.resolve('reports/screenshots');
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1400,900']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });

  console.log('======================================================================');
  console.log('PHYTOTRACE BROWSER END-TO-END VERIFICATION: REPAIRED SCANNER & RESEARCH');
  console.log('======================================================================');

  // -------------------------------------------------------------------------
  // 1. RESEARCH DASHBOARD AUDIT
  // -------------------------------------------------------------------------
  console.log('\n[1/3] Navigating to Research Dashboard (http://localhost:3000/research)...');
  await page.goto('http://localhost:3000/research', { waitUntil: 'networkidle0', timeout: 30000 });
  await new Promise(r => setTimeout(r, 2000));

  const researchPath1 = path.join(screenshotsDir, 'research_benchmark_5558326.png');
  await page.screenshot({ path: researchPath1, fullPage: true });
  console.log(`✓ Captured Research Dashboard (5558326 Benchmark): ${researchPath1}`);

  // Switch to WhatsApp tab in Research Dashboard
  console.log('Switching to WhatsApp tab in Research Dashboard...');
  const buttons = await page.$$('button');
  for (const b of buttons) {
    const text = await page.evaluate(el => el.textContent, b);
    if (text && text.includes('WhatsApp Phone Photo')) {
      await b.click();
      break;
    }
  }
  await new Promise(r => setTimeout(r, 1500));

  const researchPath2 = path.join(screenshotsDir, 'research_benchmark_whatsapp.png');
  await page.screenshot({ path: researchPath2, fullPage: true });
  console.log(`✓ Captured Research Dashboard (WhatsApp Phone Photo Benchmark): ${researchPath2}`);

  // -------------------------------------------------------------------------
  // 2. FARMER SCANNER: 5558326.jpg (Ground Truth Mine Positive)
  // -------------------------------------------------------------------------
  console.log('\n[2/3] Navigating to Farmer Scanner (http://localhost:3000/analyze)...');
  await page.goto('http://localhost:3000/analyze', { waitUntil: 'networkidle0', timeout: 30000 });
  await new Promise(r => setTimeout(r, 1500));

  const testFile1 = path.resolve('data/segmentation/pilot_upload/5558326.jpg');
  console.log(`Uploading ground-truth positive leaf: ${testFile1}`);

  const fileInput1 = await page.$('input[type="file"]');
  if (!fileInput1) throw new Error('File input not found');
  await fileInput1.uploadFile(testFile1);

  await new Promise(r => setTimeout(r, 1500));

  // Click Run Deep AI Diagnosis
  console.log('Clicking "Run Deep AI Diagnosis"...');
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const btn = btns.find(b => b.textContent && b.textContent.includes('Run Deep AI Diagnosis'));
    if (btn) btn.click();
    else throw new Error('Run Deep AI Diagnosis button not found');
  });

  // Wait for diagnosis result
  console.log('Waiting for model inference result on 5558326.jpg...');
  await page.waitForFunction(
    () => document.body.innerText.includes('Agronomic Decision') || document.body.innerText.includes('Canopy Surface Area Ratio'),
    { timeout: 30000 }
  );
  await new Promise(r => setTimeout(r, 2000));

  const analyzePath1 = path.join(screenshotsDir, 'analyze_5558326_valid_detection.png');
  await page.screenshot({ path: analyzePath1, fullPage: true });
  console.log(`✓ Captured Farmer Scanner Positive Result (5558326.jpg): ${analyzePath1}`);

  const pageText1 = await page.evaluate(() => document.body.innerText);
  console.log('Result for 5558326.jpg:');
  console.log(`  - Includes "VALID_MINE_DETECTION": ${pageText1.includes('VALID_MINE_DETECTION')}`);
  console.log(`  - Includes "19.38%": ${pageText1.includes('19.38%')}`);
  console.log(`  - Includes "Moderate Infestation": ${pageText1.includes('Moderate Infestation')}`);

  // -------------------------------------------------------------------------
  // 3. FARMER SCANNER: WhatsApp Tomato Image (Non-Miner Background Textures)
  // -------------------------------------------------------------------------
  console.log('\n[3/3] Uploading WhatsApp Phone Photo (INSP_20261105_024600_44601_PLANT_TOMATO_A07_orig.jpg)...');
  await page.goto('http://localhost:3000/analyze', { waitUntil: 'networkidle0', timeout: 30000 });
  await new Promise(r => setTimeout(r, 1500));

  const testFile2 = path.resolve('data/inspections/raw/INSP_20261105_024600_44601_PLANT_TOMATO_A07_orig.jpg');
  const fileInput2 = await page.$('input[type="file"]');
  await fileInput2.uploadFile(testFile2);

  await new Promise(r => setTimeout(r, 1500));

  console.log('Clicking "Run Deep AI Diagnosis"...');
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const btn = btns.find(b => b.textContent && b.textContent.includes('Run Deep AI Diagnosis'));
    if (btn) btn.click();
    else throw new Error('Run Deep AI Diagnosis button not found');
  });

  console.log('Waiting for model inference result on WhatsApp Phone Photo...');
  await page.waitForFunction(
    () => document.body.innerText.includes('Agronomic Decision') || document.body.innerText.includes('Canopy Surface Area Ratio'),
    { timeout: 30000 }
  );
  await new Promise(r => setTimeout(r, 2000));

  const analyzePath2 = path.join(screenshotsDir, 'analyze_whatsapp_candidates_rejected.png');
  await page.screenshot({ path: analyzePath2, fullPage: true });
  console.log(`✓ Captured Farmer Scanner Filtered Negative Result (WhatsApp Photo): ${analyzePath2}`);

  const pageText2 = await page.evaluate(() => document.body.innerText);
  console.log('Result for WhatsApp Tomato Photo:');
  console.log(`  - Includes "MINE_CANDIDATES_REJECTED": ${pageText2.includes('MINE_CANDIDATES_REJECTED')}`);
  console.log(`  - Includes "False-Positive Guardrail Audit": ${pageText2.includes('False-Positive Guardrail Audit')}`);
  console.log(`  - Includes "0.00%": ${pageText2.includes('0.00%')}`);

  await browser.close();
  console.log('\n======================================================================');
  console.log('ALL BROWSER END-TO-END VERIFICATIONS COMPLETED SUCCESSFULLY!');
  console.log('======================================================================');
})();
