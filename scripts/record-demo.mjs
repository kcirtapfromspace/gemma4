import { chromium } from 'playwright';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CLINIQ_URL = 'http://192.168.150.41:30081';
const EICR = readFileSync(join(__dirname, '..', 'data', 'eicr-samples', 'sample_eicr_01.xml'), 'utf-8');

async function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({
  recordVideo: { dir: join(__dirname, '..'), size: { width: 1280, height: 720 } },
  viewport: { width: 1280, height: 720 },
});
const page = await context.newPage();

console.log('1. Opening ClinIQ dashboard...');
await page.goto(CLINIQ_URL);
await page.waitForLoadState('networkidle');
await sleep(3000);

// Click the first case to show details
console.log('2. Clicking latest case to show extraction details...');
const firstCase = page.locator('.case-row, tr, [class*="case"]').first();
if (await firstCase.isVisible().catch(() => false)) {
  await firstCase.click();
  await sleep(3000);
}

// Click through tabs
for (const tab of ['Summary', 'Model Output', 'Entities', 'FHIR', 'Timeline']) {
  const tabEl = page.locator(`text="${tab}"`).first();
  if (await tabEl.isVisible().catch(() => false)) {
    console.log(`3. Viewing ${tab} tab...`);
    await tabEl.click();
    await sleep(2000);
  }
}

// Go back to Summary
const summaryTab = page.locator('text="Summary"').first();
if (await summaryTab.isVisible().catch(() => false)) {
  await summaryTab.click();
  await sleep(1000);
}

// Click Convert eICR
console.log('4. Opening Convert modal...');
const convertBtn = page.locator('text="Convert eICR"').first();
if (await convertBtn.isVisible().catch(() => false)) {
  await convertBtn.click();
  await sleep(1500);
}

// Find textarea and paste sample
console.log('5. Loading eICR XML sample...');
const loadSampleBtn = page.locator('#convert-modal text="Load Sample"').first();
if (await loadSampleBtn.isVisible().catch(() => false)) {
  await loadSampleBtn.click({ force: true });
  await sleep(1000);
} else {
  const textarea = page.locator('#convert-modal textarea').first();
  if (await textarea.isVisible().catch(() => false)) {
    await textarea.fill(EICR);
    await sleep(1000);
  }
}

// Submit
console.log('6. Submitting for inference (will take ~2 min)...');
const submitBtn = page.locator('#convert-modal button:has-text("Submit"), #convert-modal button:has-text("Convert"), #convert-modal button[type="submit"]').first();
if (await submitBtn.isVisible().catch(() => false)) {
  await submitBtn.click({ force: true });
}

// Wait for inference result
console.log('   Waiting for model response...');
await sleep(180000); // 3 minutes max

// Scroll to see results
await page.evaluate(() => window.scrollTo(0, 0));
await sleep(2000);

// Click on the new case if it appears
const newCase = page.locator('.case-row, tr, [class*="case"]').first();
if (await newCase.isVisible().catch(() => false)) {
  await newCase.click();
  await sleep(3000);
}

// Show FHIR output
const fhirTab = page.locator('text="FHIR"').first();
if (await fhirTab.isVisible().catch(() => false)) {
  console.log('7. Showing FHIR Bundle output...');
  await fhirTab.click();
  await sleep(3000);
}

console.log('8. Recording complete. Saving...');
await page.close();
await context.close();
await browser.close();

console.log('Demo video saved!');
