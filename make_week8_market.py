#!/usr/bin/env python3
import os
import argparse
import requests
import pandas as pd
import nfl_data_py as nfl

# Optional: load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

API_KEY = os.getenv("ODDS_API_KEY")  # set this in your environment
SPORT   = "americanfootball_nfl"
REGION  = "us"
MARKETS = "spreads,totals"
BOOKS   = "draftkings,betmgm,caesars,fanduel,pointsbetus"

# Map NFL abbreviations -> Odds API full team names
ABBR_TO_NAME = {
    "ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills",
    "CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns",
    "DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GB":"Green Bay Packers",
    "HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KC":"Kansas City Chiefs",
    "LV":"Las Vegas Raiders","LAC":"Los Angeles Chargers","LA":"Los Angeles Rams","MIA":"Miami Dolphins",
    "MIN":"Minnesota Vikings","NE":"New England Patriots","NO":"New Orleans Saints","NYG":"New York Giants",
    "NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SF":"San Francisco 49ers",
    "SEA":"Seattle Seahawks","TB":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders"
}

def load_week_matchups(season=2025, week=8) -> pd.DataFrame:
    sched = nfl.import_schedules([season])
    w = sched[sched["week"] == week].copy()
    if w.empty:
        raise SystemExit(f"No Week {week} games found for {season}.")
    return w[["away_team","home_team"]].reset_index(drop=True)

def fetch_odds(api_key: str):
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = {
        "regions": REGION,
        "markets": MARKETS,
        "oddsFormat": "american",
        "bookmakers": BOOKS,
        "apiKey": api_key
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def pick_consensus_spread_total(game_json):
    """Return (spread_home, total) by taking the median of available books."""
    spreads, totals = [], []
    home_name = game_json.get("home_team", "")
    for bk in game_json.get("bookmakers", []):
        for mk in bk.get("markets", []):
            key = mk.get("key")
            if key == "spreads":
                for o in mk.get("outcomes", []):
                    if o.get("name") == home_name and "point" in o:
                        try:
                            spreads.append(float(o["point"]))
                        except Exception:
                            pass
            elif key == "totals":
                # Over/Under share the same 'point'; take the first valid
                for o in mk.get("outcomes", []):
                    if "point" in o:
                        try:
                            totals.append(float(o["point"]))
                        except Exception:
                            pass
                        break
    if not spreads or not totals:
        return None, None
    spreads.sort(); totals.sort()
    return spreads[len(spreads)//2], totals[len(totals)//2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", dest="api_key", default=None,
                        help="Odds API key (overrides env var ODDS_API_KEY)")
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week", type=int, default=8)
    parser.add_argument("--outfile", default="week8_market.csv")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("ODDS_API_KEY")
    if api_key:
        print("Using API key from", ("--api-key" if args.api_key else "environment/.env"))
    else:
        print("No ODDS_API_KEY found → will write a template CSV (no odds filled).")

    w = load_week_matchups(args.season, args.week)
    rows = []

    if api_key:
        try:
            data = fetch_odds(api_key)
            # Build a quick index: (home_name, away_name) -> game_json
            index = {(g.get("home_team",""), g.get("away_team","")): g for g in data}

            unmatched = []
            for _, g in w.iterrows():
                away_abbr, home_abbr = g["away_team"], g["home_team"]
                home_name = ABBR_TO_NAME.get(home_abbr, home_abbr)
                away_name = ABBR_TO_NAME.get(away_abbr, away_abbr)

                match = index.get((home_name, away_name))
                if not match:
                    # Try case-insensitive match as fallback (some APIs tweak spacing)
                    match = next((gj for (h,a), gj in index.items()
                                  if h.lower()==home_name.lower() and a.lower()==away_name.lower()), None)

                if match:
                    s, t = pick_consensus_spread_total(match)
                    rows.append({"away": away_abbr, "home": home_abbr,
                                 "spread_home": "" if s is None else s,
                                 "total": "" if t is None else t})
                else:
                    unmatched.append(f"{away_abbr} @ {home_abbr}")
                    rows.append({"away": away_abbr, "home": home_abbr, "spread_home":"", "total":""})

            if unmatched:
                print("WARNING: Unmatched games (template left blank):")
                for m in unmatched: print("  -", m)

        except Exception as e:
            print("Odds fetch failed → writing template instead:", e)
            rows = [{"away": a, "home": h, "spread_home":"", "total":""}
                    for a,h in zip(w["away_team"], w["home_team"])]
    else:
        rows = [{"away": a, "home": h, "spread_home":"", "total":""}
                for a,h in zip(w["away_team"], w["home_team"])]

    out = pd.DataFrame(rows)
    out.to_csv(args.outfile, index=False)
    print(f"Saved {args.outfile} with {len(out)} rows")

if __name__ == "__main__":
    main()
