"""Scrape every player's BPR rating from https://evanmiya.com/?player_ratings.

The site is an R Shiny app. The full player table (all ~5,500 players) is pushed to the
browser in a single websocket message, so we load the page in headless Chrome, switch to
the "Advanced" view (adds position/role/class), and read that message directly instead of
paging through the table or downloading 1,000-row CSVs.

Usage:  .venv/bin/python scrape_evanmiya_bpr.py [-o evanmiya_bpr.csv]
"""
import argparse
import json
import time

import pandas as pd
from playwright.sync_api import sync_playwright

URL = "https://evanmiya.com/?player_ratings"
OUTPUT_ID = "player_ratings_page-player_data"
ADVANCED_RADIO = 'input[name="player_ratings_page-advanced"][value="2"]'
YEAR_SELECT = "#player_ratings_page-year"
DROP_COLS = ["color_O", "color_D", "color_Diff"]
RENAME = {"players": "miya_player_id", "awaiting_decision": "in_portal", "class": "class_year"}


def find_table(frames, required_col):
    """Return the newest player-table payload that contains `required_col`, else None."""
    for raw in reversed(frames):
        if OUTPUT_ID not in raw or '"values"' not in raw:
            continue
        payload = json.loads(raw)["values"].get(OUTPUT_ID)
        if not payload:
            continue
        data = payload["x"]["tag"]["attribs"]["data"]
        if required_col in data:
            return data
    return None


def scrape(timeout_s=90):
    frames = []
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome", headless=True)
        except Exception:
            browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1200})
        page.on(
            "websocket",
            lambda ws: ws.on("framereceived", lambda f: frames.append(f if isinstance(f, str) else f.decode("utf8"))),
        )
        page.goto(URL, wait_until="load")

        deadline = time.time() + timeout_s
        while find_table(frames, "bpr") is None and time.time() < deadline:
            page.wait_for_timeout(500)
        page.locator(ADVANCED_RADIO).check(force=True)
        data = None
        while data is None and time.time() < deadline:
            page.wait_for_timeout(500)
            data = find_table(frames, "position")
        season = page.locator(YEAR_SELECT).evaluate("el => el.value")
        browser.close()

    if data is None:
        raise RuntimeError("Timed out waiting for the player table from evanmiya.com")
    df = pd.DataFrame(data).drop(columns=DROP_COLS).rename(columns=RENAME)
    df.insert(0, "season", season)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="evanmiya_bpr.csv")
    args = ap.parse_args()
    df = scrape()
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} players (season {df['season'].iloc[0]}) to {args.output}")
