#!/usr/bin/env python3
import os, datetime as dt
import requests
import pandas as pd
import nfl_data_py as nfl

# Optional: load .env (not required for Open-Meteo)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

VENUE = {
  "KC":  (39.0490, -94.4839, "OUTDOOR", "America/Chicago"),
  "WAS": (38.9077, -76.8645, "OUTDOOR", "America/New_York"),
  "BAL": (39.2780, -76.6227, "OUTDOOR", "America/New_York"),
  "CHI": (41.8623, -87.6167, "OUTDOOR", "America/Chicago"),
  "ATL": (33.7555, -84.4009, "DOME",    "America/New_York"),
  "MIA": (25.9580, -80.2389, "OUTDOOR", "America/New_York"),
  "BUF": (42.7738, -78.7869, "OUTDOOR", "America/New_York"),
  "CAR": (35.2258, -80.8528, "OUTDOOR", "America/New_York"),
  "CIN": (39.0954, -84.5160, "OUTDOOR", "America/New_York"),
  "NYJ": (40.8136, -74.0746, "OUTDOOR", "America/New_York"),
  "HOU": (29.6847, -95.4107, "RETRACTABLE", "America/Chicago"),
  "SF":  (37.4030, -121.9700, "OUTDOOR", "America/Los_Angeles"),
  "NE":  (42.0909, -71.2643, "OUTDOOR", "America/New_York"),
  "CLE": (41.5061, -81.6996, "OUTDOOR", "America/New_York"),
  "PHI": (39.9008, -75.1675, "OUTDOOR", "America/New_York"),
  "NYG": (40.8136, -74.0746, "OUTDOOR", "America/New_York"),
  "NO":  (29.9509, -90.0812, "DOME", "America/Chicago"),
  "TB":  (27.9759, -82.5033, "OUTDOOR", "America/New_York"),
  "DEN": (39.7439, -105.0201, "OUTDOOR", "America/Denver"),
  "DAL": (32.7473, -97.0945, "RETRACTABLE", "America/Chicago"),
  "IND": (39.7601, -86.1639, "RETRACTABLE", "America/Indiana/Indianapolis"),
  "TEN": (36.1665, -86.7713, "OUTDOOR", "America/Chicago"),
  "PIT": (40.4468, -80.0158, "OUTDOOR", "America/New_York"),
  "GB":  (44.5013, -88.0622, "OUTDOOR", "America/Chicago"),
  "LAC": (33.9535, -118.3392, "RETRACTABLE", "America/Los_Angeles"),
  "MIN": (44.9740, -93.2575, "RETRACTABLE", "America/Chicago"),
  "JAX": (30.3239, -81.6373, "OUTDOOR", "America/New_York"),
  "SEA": (47.5952, -122.3316, "OUTDOOR", "America/Los_Angeles"),
  "LV":  (36.0909, -115.1830, "DOME", "America/Los_Angeles"),
  "DET": (42.3400, -83.0456, "DOME", "America/Detroit"),
  "ARI": (33.5276, -112.2626, "RETRACTABLE", "America/Phoenix"),
  "LA":  (33.9535, -118.3392, "RETRACTABLE", "America/Los_Angeles")
}

def load_week(season=2025, week=8) -> pd.DataFrame:
    sched = nfl.import_schedules([season])
    w = sched[sched["week"] == week].copy()
    if w.empty:
        raise SystemExit(f"No Week {week} games found for {season}.")
    return w.reset_index(drop=True)

def parse_local_dt(row) -> tuple[str, str]:
    # Date
    for c in ("gameday", "game_date", "start_date"):
        if c in row and pd.notna(row[c]):
            try:
                d = pd.to_datetime(row[c]).date()
                date_iso = d.isoformat()
                break
            except Exception:
                pass
    else:
        date_iso = "2025-10-26"  # safe fallback (Week 8 Sun)
    # Time
    time_hhmm = None
    for c in ("start_time", "gametime", "game_time_eastern", "game_time"):
        if c in row and pd.notna(row[c]):
            s = str(row[c]).strip()
            for fmt in ("%H:%M", "%I:%M %p"):
                try:
                    t = pd.to_datetime(s, format=fmt).time()
                    time_hhmm = f"{t.hour:02d}:{t.minute:02d}"
                    break
                except Exception:
                    continue
            if time_hhmm: break
    if not time_hhmm:
        weekday = str(row.get("weekday","SUN")).upper()
        time_hhmm = "20:15" if weekday in ("THU","MON") else "13:00"
    return date_iso, time_hhmm

def open_meteo_hourly(lat, lon, date_iso, tz):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "windspeed_10m,precipitation_probability",
        "timezone": tz,          # Open-Meteo returns local times (no tz offset in strings)
        "start_date": date_iso, "end_date": date_iso
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def pick_nearest_hour(payload, local_naive):
    # Parse Open-Meteo times as naive; compare to naive kickoff
    times = [pd.to_datetime(t) for t in payload["hourly"]["time"]]
    winds = payload["hourly"]["windspeed_10m"]
    precp = payload["hourly"]["precipitation_probability"]
    i = min(range(len(times)), key=lambda k: abs(times[k] - local_naive))
    return float(winds[i]), float(precp[i])

def main(season=2025, week=8, outfile="week8_weather.csv"):
    w = load_week(season, week)
    rows = []
    for _, g in w.iterrows():
        away, home = g["away_team"], g["home_team"]
        lat, lon, roof, tz = VENUE.get(home, (None, None, "OUTDOOR", "UTC"))

        wind_out = ""; precip_out = ""
        if lat is not None and lon is not None:
            date_iso, time_hhmm = parse_local_dt(g)
            # Keep kickoff as NAIVE dt to avoid tz-aware/naive subtraction
            local_naive = pd.to_datetime(f"{date_iso} {time_hhmm}")
            try:
                payload = open_meteo_hourly(lat, lon, date_iso, tz)
                wind, precip = pick_nearest_hour(payload, local_naive)
                wind_out = round(wind, 1)
                precip_out = int(round(precip))
            except Exception as e:
                print(f"Weather fetch failed for {away} @ {home}: {e}")

        rows.append({"away": away, "home": home,
                     "wind_mph": wind_out, "precip": precip_out, "roof": roof})

    pd.DataFrame(rows).to_csv(outfile, index=False)
    print(f"Saved {outfile} with {len(rows)} rows")

if __name__ == "__main__":
    import argparse
    pd.options.mode.chained_assignment = None
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--week", type=int, default=8)
    parser.add_argument("--outfile", default=None)
    args = parser.parse_args()
    outfile = args.outfile or f"week{args.week}_weather.csv"
    main(season=args.season, week=args.week, outfile=outfile)

