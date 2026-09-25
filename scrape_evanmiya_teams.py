"""Scrape every team's rating from https://evanmiya.com/?team_ratings.

Same approach as scrape_evanmiya_bpr.py: the site is an R Shiny app that pushes the full
team table (all ~365 D1 teams) to the browser in one websocket message, so we load the
page in headless Chrome and read that message directly.

Usage:  .venv/bin/python scrape_evanmiya_teams.py [-o evanmiya_team_ratings.csv]
"""
import argparse
import json
import time

import pandas as pd
from playwright.sync_api import sync_playwright

URL = "https://evanmiya.com/?team_ratings"
OUTPUT_ID = "team_ratings_page-team_ratings"
YEAR_SELECT = "#team_ratings_page-year"
DROP_COLS = [
    "tooltip_team", "tooltip_str_Diff", "tooltip_tempo_Diff",
    "color_O", "color_D", "color_Diff", "color_str_Diff", "color_tempo_Diff",
]
RENAME = {
    "rank_inj": "rank_excl_ineligible",
    "pre_roster_rank": "preseason_roster_rank",
    "runs_per_game": "kill_shots_per_game",
    "runs_conceded_per_game": "kill_shots_conceded_per_game",
    "runs_margin": "kill_shots_margin_per_game",
    "runs_total": "kill_shots_total",
    "runs_conceded_total": "kill_shots_conceded_total",
}


def find_table(frames):
    """Return the newest team-table payload seen on the websocket, else None."""
    for raw in reversed(frames):
        if OUTPUT_ID not in raw or '"values"' not in raw:
            continue
        payload = json.loads(raw)["values"].get(OUTPUT_ID)
        if payload:
            return payload["x"]["tag"]["attribs"]["data"]
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
        data = None
        while data is None and time.time() < deadline:
            page.wait_for_timeout(500)
            data = find_table(frames)
        season = page.locator(YEAR_SELECT).evaluate("el => el.value")
        browser.close()

    if data is None:
        raise RuntimeError("Timed out waiting for the team table from evanmiya.com")
    df = pd.DataFrame(data)
    assert (df["team"] == df["tooltip_team"]).all()
    df = df.drop(columns=DROP_COLS).rename(columns=RENAME)
    df.insert(0, "season", season)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="evanmiya_team_ratings.csv")
    args = ap.parse_args()
    df = scrape()
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} teams (season {df['season'].iloc[0]}) to {args.output}")
