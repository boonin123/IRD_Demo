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
from pathlib import Path

from playwright.async_api import async_playwright, Page

APP_URL = "http://localhost:8502"
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


async def click_button(page: Page, label: str, force: bool = False) -> None:
    btn = page.get_by_role("button", name=label)
    await btn.scroll_into_view_if_needed()
    await page.wait_for_timeout(200)
    await btn.click(force=force)


async def screenshot(page: Page, name: str) -> None:
    path = OUT_DIR / f"{name}.png"
    # timeout=0 skips the font-loading wait that can stall headless Chrome
    await page.screenshot(path=str(path), full_page=False, timeout=0)
    print(f"  saved {path.name}")


# ── Screenshot routines ────────────────────────────────────────────────────────

async def tab1_preview(page: Page) -> None:
    """Tab 1 — default state (layout preview, not yet trained)."""
    print("Tab 1: layout preview")
    await click_tab(page, "Grid World RL")
    await page.wait_for_timeout(500)
    await screenshot(page, "tab1_preview")


async def tab1_trained_and_lava(page: Page) -> None:
    """
    Tab 1 — single training run on the Lava Field preset.
    Screenshots:
      tab1_trained          — policy + value function after training
      tab1_lava_trajectory  — greedy episode where agent walks into lava
    """
    print("Tab 1: training on Lava Field preset")
    await click_tab(page, "Grid World RL")

    # Configuration expander is open on fresh load (trained=False → expanded=True)
    await page.wait_for_selector("[data-testid='stExpander']", timeout=8_000)
    await page.wait_for_timeout(500)

    # Select Lava Field preset
    await select_option(page, "Preset", "Lava Field")
    await page.wait_for_timeout(400)

    # Train — use JS click to avoid any header/overlay interception
    await page.evaluate("""
        () => {
            const btn = [...document.querySelectorAll('button')]
                .find(b => b.innerText.trim() === 'Train Agent');
            if (btn) btn.click();
        }
    """)
    await wait_for_idle(page, timeout=60_000)
    await screenshot(page, "tab1_trained")

    print("Tab 1: Lava Field trajectory")
    # Run greedy episode — lava is activated at test time
    await page.evaluate("""
        () => {
            const btn = [...document.querySelectorAll('button')]
                .find(b => b.innerText.trim() === 'Run Greedy Episode');
            if (btn) btn.click();
        }
    """)
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

        # Load the app — Streamlit renders via WebSocket after initial HTML load
        await page.goto(APP_URL)
        # Wait for Streamlit's main content wrapper to exist in the DOM
        try:
            await page.wait_for_selector(
                "[data-testid='stAppViewContainer']", timeout=20_000
            )
        except Exception:
            print("ERROR: Streamlit app container not found. Is the app running?")
            await browser.close()
            return
        # Give React time to finish rendering
        await page.wait_for_timeout(3_000)
        # Confirm the app title rendered
        title = await page.title()
        print(f"  page title: {title}")

        print("App is ready. Taking screenshots...\n")

        await tab1_preview(page)
        await tab1_trained_and_lava(page)  # trains on Lava Field, then runs episode
        await tab2_ird(page)               # Tab 2 picks up the trained agent automatically
        await tab3_challenge(page)         # Tab 3 also reuses the same agent

        await browser.close()

    print(f"\nDone! {len(list(OUT_DIR.glob('*.png')))} screenshots in {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
