"""Drive the running app with Playwright to capture README screenshots and the demo video.

Usage:  streamlit run app.py &   then   python scripts/capture_media.py [url]
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8501"
DOCS = Path(__file__).resolve().parent.parent / "docs"
SHOTS = DOCS / "screenshots"
SHOTS.mkdir(parents=True, exist_ok=True)

STEPS = [
    ("Which region generated the highest revenue?", "01-top-region"),
    ("Show monthly sales trends.", "02-monthly-trend"),
    ("Which products are underperforming?", "03-underperformers"),
    ("Detect anomalies in the dataset.", "04-anomalies"),
    ("Generate SQL for this analysis.", "05-sql-followup"),
]


def ask(page, text, shot):
    n = page.locator('[data-testid="stChatMessage"]').count()
    box = page.locator('[data-testid="stChatInputTextArea"]')
    box.click()
    box.fill(text)
    box.press("Enter")
    page.wait_for_function(
        "n => document.querySelectorAll('[data-testid=\"stChatMessage\"]').length >= n + 2", arg=n, timeout=60000)
    page.wait_for_timeout(2500)
    page.locator('[data-testid="stChatMessage"]').last.scroll_into_view_if_needed()
    page.wait_for_timeout(1200)
    page.screenshot(path=str(SHOTS / f"{shot}.png"))


def main():
    tmp = Path(tempfile.mkdtemp())
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=str(tmp),
                            record_video_size={"width": 1440, "height": 900})
        page = ctx.new_page()
        page.goto(URL)
        page.get_by_role("button", name="Load sample dataset").click()
        page.wait_for_timeout(2500)
        page.screenshot(path=str(SHOTS / "00-loaded-sample.png"))
        for text, shot in STEPS:
            ask(page, text, shot)
        # data tab screenshot, if present
        for label, shot in (("Data & quality", "06-data-quality"), ("Dashboard", "07-dashboard")):
            tab = page.get_by_role("tab", name=label)
            if tab.count():
                tab.first.click()
                page.wait_for_timeout(2500)
                page.screenshot(path=str(SHOTS / f"{shot}.png"))
        ctx.close()
        b.close()
    webm = next(tmp.glob("*.webm"))
    raw = DOCS / "demo_raw.webm"
    shutil.move(webm, raw)
    print("raw video:", raw)


if __name__ == "__main__":
    main()
