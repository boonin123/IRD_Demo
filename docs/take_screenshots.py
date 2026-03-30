#!/usr/bin/env python3
"""
Capture screenshots of all IRL Explorer app states for the README.

Usage
-----
1. Install dependencies (one-time):
       pip install playwright
       playwright install chromium

2. Start the Streamlit app in a separate terminal:
       streamlit run app.py

3. Run this script:
       python docs/take_screenshots.py

Screenshots are written to docs/screenshots/.
"""

import asyncio
import time
from pathlib import Path

from playwright.async_api import async_playwright, Page

APP_URL = "http://localhost:8501"
OUT_DIR = Path(__file__).parent / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIEWPORT = {"width": 1400, "height": 900}


# ── Helpers ────────────────────────────────────────────────────────────────────

async def wait_for_idle(page: Page, timeout: int = 90_000) -> None:
    """Wait for Streamlit to finish any running computation."""
    # Give Streamlit a moment to register the click and start running
    await page.wait_for_timeout(600)
    # Wait for the 'Running...' status widget to appear, then disappear
    try:
        await page.wait_for_selector(
            "[data-testid='stStatusWidget']", timeout=3_000
        )
    except Exception:
        pass
    try:
        await page.wait_for_selector(
            "[data-testid='stStatusWidget']", state="hidden", timeout=timeout
        )
    except Exception:
        pass
    # Small settle buffer for final renders
    await page.wait_for_timeout(800)


async def click_tab(page: Page, label: str) -> None:
    await page.get_by_role("tab", name=label).click()
    await wait_for_idle(page, timeout=5_000)


async def select_option(page: Page, label: str, option: str) -> None:
    """Open a Streamlit selectbox by its label and choose an option."""
    # Click the selectbox container that sits near its label
    box = page.locator(f"label:has-text('{label}')").locator("..").locator(
        "[data-testid='stSelectbox']"
    )
    if await box.count() == 0:
        # Fallback: find any selectbox in the visible form area
        box = page.locator("[data-testid='stSelectbox']").filter(has_text=label)
    await box.click()
    await page.wait_for_timeout(300)
    await page.get_by_role("option", name=option, exact=True).click()
    await page.wait_for_timeout(300)


async def click_button(page: Page, label: str) -> None:
    await page.get_by_role("button", name=label).click()


async def screenshot(page: Page, name: str) -> None:
    path = OUT_DIR / f"{name}.png"
    await page.screenshot(path=str(path), full_page=False)
    print(f"  saved {path.name}")


# ── Screenshot routines ────────────────────────────────────────────────────────

async def tab1_preview(page: Page) -> None:
    """Tab 1 — default state (layout preview, not yet trained)."""
    print("Tab 1: layout preview")
    await click_tab(page, "Grid World RL")
    await page.wait_for_timeout(500)
    await screenshot(page, "tab1_preview")


async def tab1_trained(page: Page) -> None:
    """Tab 1 — train Q-Learning on default 5×5 grid and screenshot result."""
    print("Tab 1: training Q-Learning agent")
    await click_tab(page, "Grid World RL")

    # Open the Configuration expander (it auto-closes after training)
    expander = page.locator("[data-testid='stExpander']").first
    if await expander.get_by_text("Configuration").is_visible():
        await expander.click()
        await page.wait_for_timeout(300)

    await click_button(page, "Train Agent")
    await wait_for_idle(page, timeout=60_000)
    await screenshot(page, "tab1_trained")


async def tab1_lava(page: Page) -> None:
    """Tab 1 — Lava Field preset: agent trained then walks into lava."""
    print("Tab 1: Lava Field trajectory")
    await click_tab(page, "Grid World RL")

    # Reset any prior training
    try:
        await click_button(page, "Reset")
        await wait_for_idle(page, timeout=5_000)
    except Exception:
        pass

    # Open Configuration
    expander = page.locator("[data-testid='stExpander']").first
    try:
        await expander.click()
        await page.wait_for_timeout(300)
    except Exception:
        pass

    # Select Lava Field preset
    await select_option(page, "Preset", "Lava Field")

    # Train
    await click_button(page, "Train Agent")
    await wait_for_idle(page, timeout=60_000)

    # Run greedy episode (lava activates at test time)
    await click_button(page, "Run Greedy Episode")
    await wait_for_idle(page, timeout=10_000)
    await screenshot(page, "tab1_lava_trajectory")


async def tab2_ird(page: Page) -> None:
    """Tab 2 — IRD comparison (requires Lava Field agent to be trained first)."""
    print("Tab 2: running IRD")
    await click_tab(page, "Inverse Reward Design")
    await page.wait_for_timeout(500)

    # Run IRD with defaults
    await click_button(page, "Run IRD")
    await wait_for_idle(page, timeout=120_000)

    # Screenshot the full three-column comparison
    await screenshot(page, "tab2_ird_comparison")

    # Scroll down to capture reward heatmaps + convergence chart
    await page.evaluate("window.scrollBy(0, 500)")
    await page.wait_for_timeout(400)
    await screenshot(page, "tab2_reward_heatmaps")

    # Scroll back to top
    await page.evaluate("window.scrollTo(0, 0)")


async def tab3_challenge(page: Page) -> None:
    """Tab 3 — Robustness Challenge with H-Wall preset."""
    print("Tab 3: robustness challenge")
    await click_tab(page, "Robustness Challenge")
    await page.wait_for_timeout(500)

    # Load the H-Wall preset
    await click_button(page, "H-Wall")
    await page.wait_for_timeout(500)

    # Open IRD Parameters expander and run
    await click_button(page, "Run Challenge")
    await wait_for_idle(page, timeout=120_000)
    await screenshot(page, "tab3_challenge")


# ── Main ───────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"Connecting to {APP_URL} ...")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT)
        page = await context.new_page()

        # Load the app and wait for Streamlit to be ready
        await page.goto(APP_URL, wait_until="networkidle")
        await page.wait_for_timeout(2_000)

        # Confirm the app title is visible before proceeding
        try:
            await page.wait_for_selector("text=IRL Explorer", timeout=15_000)
        except Exception:
            print("ERROR: Could not find 'IRL Explorer' title. Is the app running?")
            await browser.close()
            return

        print("App is ready. Taking screenshots...\n")

        await tab1_preview(page)
        await tab1_trained(page)
        await tab1_lava(page)      # leaves the Lava Field agent in session state
        await tab2_ird(page)       # Tab 2 picks up the trained agent automatically
        await tab3_challenge(page) # Tab 3 also reuses the same agent

        await browser.close()

    print(f"\nDone! {len(list(OUT_DIR.glob('*.png')))} screenshots in {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
