#!/usr/bin/env python3
"""PhytoTrace AI — Headless Browser Verification of Fixed Smartphone Leaf Inference."""

import os
import sys
import time
from playwright.sync_api import sync_playwright

def verify():
    img_path = os.path.abspath("data/inspections/raw/INSP_20261105_024600_44601_PLANT_TOMATO_A07_orig.jpg")
    out_dir = os.path.abspath("reports/debug_segmentation")
    os.makedirs(out_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("1. Navigating to http://localhost:3000/analyze ...")
        page.goto("http://localhost:3000/analyze", wait_until="networkidle")
        time.sleep(1)

        print("2. Uploading smartphone leaf image ...")
        file_input = page.locator('input[type="file"]').first
        file_input.set_input_files(img_path)
        time.sleep(1)

        # Take screenshot of staged photograph
        page.screenshot(path=os.path.join(out_dir, "15_browser_staged_preview.png"), full_page=True)
        print("  Saved staged preview screenshot.")

        print("3. Clicking 'Run Deep AI Diagnosis' ...")
        diagnose_btn = page.locator("button:has-text('Run Deep AI Diagnosis')")
        diagnose_btn.click()

        # Wait for results
        print("4. Waiting for AI inference and UI update ...")
        page.wait_for_selector("text=Agronomic Decision", timeout=20000)
        time.sleep(2)

        # Check for key fixed assertions
        body_text = page.inner_text("body")
        assert "0.00%" in body_text or "0%" in body_text, "Damage percentage should be 0%"
        assert "No Visible Mine" in body_text or "No Serpentine" in body_text, "Decision should be no visible mine"
        assert "1,864,438" not in body_text, "1.86M false region should NOT appear!"

        print("  Confirmed: 0% damage, no visible mine, 1.86M px false region eliminated!")

        # Take screenshot of completed diagnosis
        page.screenshot(path=os.path.join(out_dir, "16_browser_completed_diagnosis.png"), full_page=True)
        print("  Saved completed diagnosis screenshot.")

        browser.close()
        print("✅ Headless browser verification successfully completed!")

if __name__ == "__main__":
    verify()

