"""Download NIL values for every men's basketball player from https://thesideline.co/nil-tracker/top-players.

That page renders 25 rows at a time and filters by sport in the browser, but all players
come from one static JSON file (the same file the page loads). We fetch it and keep the
men's basketball rows (sport code "B", the site's "CBB").

Usage:  .venv/bin/python scrape_sideline_nil.py [-o sideline_mbb_player_nil.csv]
"""
import argparse
import time

import pandas as pd
import requests

INDEX_URL = "https://thesideline.co/nil-player-index.json"
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
SOURCE = {"a": "public", "m": "modeled"}


def scrape():
    resp = requests.get(f"{INDEX_URL}?v={int(time.time() // 600)}", headers=HEADERS, timeout=60)
    resp.raise_for_status()
    index = resp.json()
    schools = index["schools"]

    rows = []
    for p in index["players"]:
        if p.get("sp") != "B":
            continue
        school = schools.get(str(p["sc"]), {})
        rows.append(
            {
                "sideline_player_id": p["i"],
                "espn_athlete_id": p["i"] if str(p["i"]).isdigit() else None,
                "player_name": p["n"],
                "position": p.get("p"),
                "class_year": p.get("y"),
                "jersey": p.get("j"),
                "school_id": p["sc"],
                "school": school.get("n"),
                "school_slug": school.get("s"),
                "conference": school.get("c"),
                "nil_value_usd": p["v"],
                "nil_source": SOURCE.get(p.get("src"), p.get("src")),
                "recruit_stars": p.get("st"),
                "sideline_overall_rank": p.get("gr"),
            }
        )
    df = pd.DataFrame(rows).sort_values("nil_value_usd", ascending=False, kind="stable", ignore_index=True)
    return df, index["generatedAt"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="sideline_mbb_player_nil.csv")
    args = ap.parse_args()
    df, generated_at = scrape()
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} men's basketball players to {args.output} (site data generated {generated_at})")
