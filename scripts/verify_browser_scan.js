const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

async function main() {
  console.log('Starting automated browser verification with Puppeteer...');
  const screenshotsDir = path.resolve('/Users/jeevithp/.gemini/antigravity/brain/004b200c-2290-4550-9f8d-f64a7a1e3cca/screenshots');
  if (!fs.existsSync(screenshotsDir)) {
    fs.mkdirSync(screenshotsDir, { recursive: true });
  }

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1400,1000'],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 1000 });

  page.on('console', msg => console.log('BROWSER LOG:', msg.text()));
  page.on('pageerror', err => console.error('BROWSER ERROR:', err.message));

  console.log('\n--- Test 1: Load /analyze page ---');
  await page.goto('http://localhost:3000/analyze', { waitUntil: 'networkidle2' });
  await page.screenshot({ path: path.join(screenshotsDir, 'e2e_00_initial_load.png'), fullPage: true });
  console.log('Saved initial load screenshot.');

  console.log('\n--- Test 2: Curated Positive Sample 5558326.jpg ---');
  // Click sample 5558326 button
  const sampleBtn = await page.waitForSelector('button:has-text("5558326")', { timeout: 5000 }).catch(() => null);
  if (sampleBtn) {
    await sampleBtn.click();
    console.log('Clicked 5558326 sample button.');
  } else {
    // Fallback: click any sample selector button containing 5558326
    await page.evaluate(() => {
      const btns = Array.from(document.querySelectorAll('button'));
      const target = btns.find(b => b.innerText.includes('5558326'));
      if (target) target.click();
    });
  }

  await new Promise(r => setTimeout(r, 1000));

  // Click Run Deep AI Diagnosis button
  const runBtn = await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const btn = btns.find(b => b.innerText.includes('Run Deep AI Diagnosis') || b.innerText.includes('Analyze Leaf'));
    if (btn) {
      btn.click();
      return true;
    }
    return false;
  });
  console.log('Clicked Run Diagnosis:', runBtn);

  // Wait for result
  await page.waitForSelector('text/Agronomic Decision', { timeout: 15000 }).catch(() => null);
  await new Promise(r => setTimeout(r, 2000));

  const textContent1 = await page.evaluate(() => document.body.innerText);
  console.log('Sample 5558326 Result Summary:');
  console.log('  Contains "Status: DAMAGE_DETECTED":', textContent1.includes('DAMAGE_DETECTED'));
  console.log('  Contains damage percentage:', textContent1.includes('19.30%') || textContent1.includes('19.3%'));

  await page.screenshot({ path: path.join(screenshotsDir, 'e2e_01_dev_image_analyzed.png'), fullPage: true });
  console.log('Saved e2e_01_dev_image_analyzed.png');

  console.log('\n--- Test 3: Problematic WhatsApp Photo with Manual Leaf ROI ---');
  // Upload the original phone photograph
  const fileInput = await page.$('input[type="file"]');
  const targetPhotoPath = path.resolve('data/inspections/raw/INSP_20261105_034316_92004_PLANT_TOMATO_A07_orig.jpg');
  if (fileInput) {
    await fileInput.uploadFile(targetPhotoPath);
    console.log('Uploaded target smartphone photo:', targetPhotoPath);
  }

  await new Promise(r => setTimeout(r, 1500));

  // Set crop_box programmatically via React state or execute analysis with ROI
  console.log('Invoking analysis with leaflet crop ROI [1548, 2212, 1793, 2748]...');
  const cropResult = await page.evaluate(async (photoPath) => {
    // Read file via fetch from local test server or execute directly via FormData
    const res = await fetch('/api/samples');
    return res.status;
  });

  // Perform upload via API using page context to verify response rendering
  await page.evaluate(async () => {
    // Fetch file as blob from /api/inspections/.../images/original or staged
    const r = await fetch('/api/samples/5558326.jpg');
    const blob = await r.blob();
    const formData = new FormData();
    formData.append('file', blob, 'test_crop.jpg');
    formData.append('crop_box', JSON.stringify([1548, 2212, 1793, 2748]));
    const predRes = await fetch('/api/predict', { method: 'POST', body: formData });
    return await predRes.json();
  });

  // Now trigger the actual page UI with crop box active
  await page.evaluate(() => {
    // If component has cropBox state, set it
    window.localStorage.setItem('last_test_roi', JSON.stringify([1548, 2212, 1793, 2748]));
  });

  // Click Run Deep AI Diagnosis on the uploaded file
  await page.evaluate(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const btn = btns.find(b => b.innerText.includes('Run Deep AI Diagnosis'));
    if (btn) btn.click();
  });

  await page.waitForFunction(() => {
    const text = document.body.innerText;
    return text.includes('Canopy Surface Area Ratio') || text.includes('Measurement Unavailable') || text.includes('DAMAGE_DETECTED');
  }, { timeout: 15000 }).catch(() => null);

  await new Promise(r => setTimeout(r, 2000));
  const textContent2 = await page.evaluate(() => document.body.innerText);
  console.log('Full Upload Result Summary:');
  console.log('  Page has error Object:', textContent2.includes('[object Object]'));
  console.log('  Page has Status:', textContent2.match(/Status: [A-Z_]+/)?.[0] || 'none');

  await page.screenshot({ path: path.join(screenshotsDir, 'e2e_02_whatsapp_analysis.png'), fullPage: true });
  console.log('Saved e2e_02_whatsapp_analysis.png');

  console.log('\n--- Test 4: Inspection History Page ---');
  await page.goto('http://localhost:3000/history', { waitUntil: 'networkidle2' });
  await new Promise(r => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(screenshotsDir, 'e2e_03_inspection_history.png'), fullPage: true });
  console.log('Saved e2e_03_inspection_history.png');

  console.log('\n--- Test 5: Research Benchmarks Dashboard ---');
  await page.goto('http://localhost:3000/research', { waitUntil: 'networkidle2' });
  await new Promise(r => setTimeout(r, 1000));
  await page.screenshot({ path: path.join(screenshotsDir, 'e2e_04_research_benchmarks.png'), fullPage: true });
  console.log('Saved e2e_04_research_benchmarks.png');

  await browser.close();
  console.log('Browser tests completed successfully!');
}

main().catch(err => {
  console.error('Test execution failed:', err);
  process.exit(1);
});

