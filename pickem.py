#!/usr/bin/env python3
"""
NFL Week X Monte Carlo — Auto CSV + D-model + EWMA + Weather + Market Blend
- Robust to seasons where play-by-play (PBP) parquet isn't available (404/bug)
- Avoids pandas groupby.apply deprecation
- Prints clean, aligned table sorted by win prob
- Exposes a tie-breaker (median total) for EVERY matchup

CLI:
  python pickem.py --season 2025 --week 8 --sims 10000
"""

# ================================================================
# --- Imports & env/parquet engine hints ---
# ================================================================
import os
import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
import nfl_data_py as nfl

# Prefer fastparquet on Py3.13 if pyarrow wheels aren't available
try:
    import fastparquet  # noqa: F401
    pd.options.io.parquet.engine = "fastparquet"
except Exception:
    # If pyarrow is installed, pandas will use it automatically
    pass

# ================================================================
# --- CLI & config knobs ---
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--season", type=int, default=2025)
parser.add_argument("--week", type=int, default=8)
parser.add_argument("--sims", type=int, default=10_000)
args = parser.parse_args()

SEASON = args.season
TARGET_WEEK = args.week
N_SIMS = args.sims
RANDOM_SEED = 42

# Model knobs
HOME_FIELD_POINTS = 1.5
RECENT_WEEKS_QB = 3
EPA_TO_POINTS_PER_0P1 = 6.0
PACE_SCALER = 0.10
W_EPA, W_PFPA, W_QB = 0.55, 0.25, 0.20
W_MARKET, W_MODEL = 0.60, 0.40

# Recency weighting (EWMA)
EWMA_ALPHA_EPA  = 0.50   # EPA + Pace (recent weeks matter more)
EWMA_ALPHA_PFPA = 0.35   # PF/PA smoother

# Weather knobs (applied only if weather CSV present)
WIND_THRESHOLD = 12                # mph; below this, no wind penalty
WIND_PENALTY_PER_MPH = 0.15        # pts per team per mph above threshold
PRECIP_PENALTY = 0.8               # flat pts per team if precip truthy
DOME_TAGS = {"DOME"}               # no weather impact
RETRACTABLE_TAGS = {"RETRACTABLE"} # treat like outdoor unless you edit

np.random.seed(RANDOM_SEED)

# ================================================================
# --- Utilities ---
# ================================================================
def ensure_csvs(file_stem: str, season: int, week: int):
    """
    If market/weather CSVs are missing, call helper scripts to generate them.
    """
    market_csv  = f"{file_stem}_market.csv"
    weather_csv = f"{file_stem}_weather.csv"

    need_market  = not os.path.exists(market_csv)
    need_weather = not os.path.exists(weather_csv)

    if need_market:
        print(f"{market_csv} not found → generating…")
        subprocess.run(
            [sys.executable, "make_week8_market.py",
             "--season", str(season), "--week", str(week),
             "--outfile", market_csv],
            check=False
        )

    if need_weather:
        print(f"{weather_csv} not found → generating…")
        subprocess.run(
            [sys.executable, "make_week8_weather.py",
             "--season", str(season), "--week", str(week),
             "--outfile", weather_csv],
            check=False
        )

def ewma_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    s = weights.sum()
    if s <= 0: return 0.0
    return float((values * weights).sum() / s)

def build_ewma_weights(weeks: np.ndarray, alpha: float, last_week: int) -> np.ndarray:
    gaps = (last_week - weeks).clip(min=0)
    return (1.0 - alpha) ** gaps  # newer weeks → larger weight

def pretty_score(x: float) -> int:
    return int(round(x))

def edge_label(pct: float) -> str:
    if pct >= 70: return "Heavy edge"
    if pct >= 60: return "Solid edge"
    if pct >= 52: return "Slight edge"
    return "Coin-flip"

def market_note(spread_home):
    if spread_home is None or spread_home == "":
        return ""
    try:
        sh = float(spread_home)
    except Exception:
        return ""
    if abs(sh) < 0.25:
        return " | Market: pick’em"
    if sh < 0:
        return f" | Market: Home by {abs(sh):.1f}"
    else:
        return f" | Market: Away by {abs(sh):.1f}"

# ================================================================
# --- Load schedules & training window ---
# ================================================================
file_stem = f"week{TARGET_WEEK}"
ensure_csvs(file_stem, SEASON, TARGET_WEEK)

print(f"Loading schedules for {SEASON}...")
sched = nfl.import_schedules([SEASON])
completed = sched.dropna(subset=["home_score","away_score"]).copy()
if completed.empty:
    raise SystemExit("No completed games found for this season yet.")
latest_completed_week = int(completed["week"].max())
print(f"Latest completed week: {latest_completed_week}")

train_weeks = min(TARGET_WEEK - 1, latest_completed_week)
train = completed[completed["week"] <= train_weeks].copy()
if train.empty:
    train = completed.copy()

# ================================================================
# --- Market & Weather CSVs (optional) ---
# ================================================================
market_df = None
market_path = f"{file_stem}_market.csv"
if os.path.exists(market_path):
    try:
        tmp = pd.read_csv(market_path)
        cols = {c.lower() for c in tmp.columns}
        if {"away","home","spread_home","total"}.issubset(cols):
            tmp.columns = [c.lower() for c in tmp.columns]
            market_df = tmp[["away","home","spread_home","total"]].copy()
    except Exception as e:
        print("Warning: could not read market CSV:", e)

weather_df = None
weather_path = f"{file_stem}_weather.csv"
if os.path.exists(weather_path):
    try:
        w = pd.read_csv(weather_path)
        cols = {c.lower() for c in w.columns}
        if {"away","home","wind_mph","precip","roof"}.issubset(cols):
            w.columns = [c.lower() for c in w.columns]
            weather_df = w[["away","home","wind_mph","precip","roof"]].copy()
    except Exception as e:
        print("Warning: could not read weather CSV:", e)

# ================================================================
# --- PF/PA per game with EWMA (deprecation-safe) ---
# ================================================================
home_g = train[["week","home_team","home_score","away_score"]].rename(
    columns={"home_team":"team","home_score":"pf","away_score":"pa"}
)
away_g = train[["week","away_team","away_score","home_score"]].rename(
    columns={"away_team":"team","away_score":"pf","home_score":"pa"}
)
long_pg = pd.concat([home_g, away_g], ignore_index=True)
weights_pfpa = build_ewma_weights(long_pg["week"].to_numpy(), EWMA_ALPHA_PFPA, train_weeks)
long_pg["w"] = weights_pfpa

def _agg_pfpa(df):
    w = df["w"].to_numpy()
    pf = df["pf"].to_numpy()
    pa = df["pa"].to_numpy()
    pf_g = ewma_weighted_mean(pf, w)
    pa_g = ewma_weighted_mean(pa, w)
    return pd.Series({"pf_g": pf_g, "pa_g": pa_g})

# NOTE: group_keys=False silences the future deprecation
pfpa = long_pg.groupby("team", group_keys=False).apply(_agg_pfpa).reset_index()
league_avg_pf = float(pfpa["pf_g"].mean())

# ================================================================
# --- Robust PBP + EPA/pace + QB form (works even if PBP missing) ---
# ================================================================
def _ewma_team_metric(df, col, alpha, last_week):
    w = build_ewma_weights(df["week"].to_numpy(), alpha, last_week)
    v = df[col].to_numpy()
    return ewma_weighted_mean(v, w)

def qb_recent_epa_from_pbp(pbp_all: pd.DataFrame, season: int, train_wk: int, recent_window: int) -> pd.DataFrame:
    if pbp_all is None or pbp_all.empty or train_wk <= 0:
        return pd.DataFrame({"team": pfpa["team"], "qb_epa_play": 0.0})

    p = pbp_all[(pbp_all["season"] == season) & (pbp_all["week"] <= train_wk)].copy()
    if p.empty:
        return pd.DataFrame({"team": pfpa["team"], "qb_epa_play": 0.0})

    if "pass" in p.columns and p["pass"].dtype != bool:
        p["pass"] = p["pass"].astype(int)
    p = p[p["pass"] == 1]
    if p.empty:
        return pd.DataFrame({"team": pfpa["team"], "qb_epa_play": 0.0})

    min_week = max(0, train_wk - recent_window)
    p = p[p["week"] > min_week] if recent_window > 0 else p
    if p.empty:
        return pd.DataFrame({"team": pfpa["team"], "qb_epa_play": 0.0})

    name_col = "passer_player_name" if "passer_player_name" in p.columns else None
    if not name_col:
        qb_team = p.groupby("posteam")["epa"].mean().rename("qb_epa_play").reset_index().rename(columns={"posteam":"team"})
        if qb_team.empty:
            return pd.DataFrame({"team": pfpa["team"], "qb_epa_play": 0.0})
        return qb_team

    atts = (
        p.groupby(["posteam", name_col]).size().rename("atts").reset_index()
         .sort_values(["posteam","atts"], ascending=[True, False])
         .groupby("posteam").head(1)
         .rename(columns={"posteam":"team", name_col:"qb"})
    )
    qb_epa = (
        p.groupby(["posteam", name_col])["epa"].mean()
         .rename("qb_epa_play").reset_index()
         .rename(columns={"posteam":"team", name_col:"qb"})
    )
    out = atts.merge(qb_epa, on=["team","qb"], how="left")
    out["qb_epa_play"] = out["qb_epa_play"].fillna(0.0)
    return out[["team","qb_epa_play"]]

def build_rates_from_pbp(pbp_all: pd.DataFrame, train_wk: int) -> pd.DataFrame:
    if pbp_all is None or pbp_all.empty or train_wk <= 0:
        return pd.DataFrame({
            "team": pfpa["team"],
            "epa_off": 0.0,
            "epa_def": 0.0,
            "pace": 60.0
        })

    pbp_use = pbp_all[pbp_all["week"] <= train_wk].copy()
    if pbp_use.empty:
        return pd.DataFrame({
            "team": pfpa["team"],
            "epa_off": 0.0,
            "epa_def": 0.0,
            "pace": 60.0
        })

    if "pass" in pbp_use.columns and pbp_use["pass"].dtype != bool:
        pbp_use["pass"] = pbp_use["pass"].astype(int)

    off_week = (
        pbp_use.groupby(["posteam","week"])["epa"].mean()
               .rename("epa_off_wk").reset_index().rename(columns={"posteam":"team"})
    )
    def_week = (
        pbp_use.groupby(["defteam","week"])["epa"].mean()
               .rename("epa_def_wk").reset_index().rename(columns={"defteam":"team"})
    )
    plays_team_game = (
        pbp_use.dropna(subset=["game_id"])
               .groupby(["game_id","posteam","week"]).size().rename("plays").reset_index()
               .rename(columns={"posteam":"team"})
    )
    pace_week = plays_team_game.groupby(["team","week"])["plays"].mean().rename("pace_wk").reset_index()

    epa_off = off_week.groupby("team").apply(lambda d: _ewma_team_metric(d, "epa_off_wk", EWMA_ALPHA_EPA,  train_wk)).rename("epa_off")
    epa_def = def_week.groupby("team").apply(lambda d: _ewma_team_metric(d, "epa_def_wk", EWMA_ALPHA_EPA,  train_wk)).rename("epa_def")
    pace    = pace_week.groupby("team").apply(lambda d: _ewma_team_metric(d, "pace_wk",    EWMA_ALPHA_EPA,  train_wk)).rename("pace")

    out = (
        pd.DataFrame(epa_off).reset_index()
          .merge(pd.DataFrame(epa_def).reset_index(), on="team", how="outer")
          .merge(pd.DataFrame(pace).reset_index(),    on="team", how="outer")
          .fillna({"epa_off":0.0, "epa_def":0.0, "pace":60.0})
    )
    out = pfpa[["team"]].merge(out, on="team", how="left").fillna({"epa_off":0.0,"epa_def":0.0,"pace":60.0})
    return out

# Try to import PBP (nfl_data_py may 404 / bug on Py3.13)
try:
    pbp = nfl.import_pbp_data([SEASON])
except Exception as e:
    print(f"Warning: PBP not available for {SEASON} (reason: {e}). Proceeding without PBP features.")
    pbp = pd.DataFrame(columns=["season","week","posteam","defteam","epa","game_id","pass","passer_player_name"])

if not pbp.empty:
    pbp = pbp[pbp["week"] <= train_weeks].copy()

rates  = build_rates_from_pbp(pbp, train_weeks)
qbform = qb_recent_epa_from_pbp(pbp, SEASON, train_weeks, RECENT_WEEKS_QB)
if qbform.empty:
    qbform = pd.DataFrame({"team": pfpa["team"].tolist(), "qb_epa_play":[0.0]*len(pfpa)})

# ================================================================
# --- Week schedule ---
# ================================================================
week_df = sched[sched["week"] == TARGET_WEEK].copy()
if week_df.empty:
    raise SystemExit(f"No schedule entries found for week {TARGET_WEEK} in {SEASON}.")

# ================================================================
# --- Modeling helpers (points, market, weather, sims) ---
# ================================================================
def expected_points_from_components(home, away):
    rH = rates[rates["team"] == home].iloc[0] if (rates["team"] == home).any() else pd.Series({"epa_off":0.0,"epa_def":0.0,"pace":60.0})
    rA = rates[rates["team"] == away].iloc[0] if (rates["team"] == away).any() else pd.Series({"epa_off":0.0,"epa_def":0.0,"pace":60.0})
    pH = pfpa[pfpa["team"] == home].iloc[0] if (pfpa["team"] == home).any() else pd.Series({"pf_g":league_avg_pf,"pa_g":league_avg_pf})
    pA = pfpa[pfpa["team"] == away].iloc[0] if (pfpa["team"] == away).any() else pd.Series({"pf_g":league_avg_pf,"pa_g":league_avg_pf})
    qH = qbform[qbform["team"] == home]["qb_epa_play"].iloc[0] if (qbform["team"] == home).any() else 0.0
    qA = qbform[qbform["team"] == away]["qb_epa_play"].iloc[0] if (qbform["team"] == away).any() else 0.0

    # EPA contrast → points
    epa_home = rH["epa_off"] - rA["epa_def"]
    epa_away = rA["epa_off"] - rH["epa_def"]
    epa_pts_home = (epa_home / 0.1) * EPA_TO_POINTS_PER_0P1
    epa_pts_away = (epa_away / 0.1) * EPA_TO_POINTS_PER_0P1

    # PF/PA stabilizer near league average
    pfpa_home = 0.5 * (pH["pf_g"] + (league_avg_pf - (pA["pa_g"] - league_avg_pf)))
    pfpa_away = 0.5 * (pA["pf_g"] + (league_avg_pf - (pH["pa_g"] - league_avg_pf)))

    # QB relative adjustment
    qb_pts_home = (qH - qA) / 0.1 * (EPA_TO_POINTS_PER_0P1 / 2.0)
    qb_pts_away = -qb_pts_home

    # Pace (mild)
    pace_factor_pts = PACE_SCALER * ((rH.get("pace",60.0) - 60.0) + (rA.get("pace",60.0) - 60.0))

    base = league_avg_pf
    model_home = base + (W_EPA*epa_pts_home) + (W_PFPA*(pfpa_home - base)) + (W_QB*qb_pts_home) + pace_factor_pts + HOME_FIELD_POINTS
    model_away = base + (W_EPA*epa_pts_away) + (W_PFPA*(pfpa_away - base)) + (W_QB*qb_pts_away) + pace_factor_pts
    return float(max(6.0, model_home)), float(max(6.0, model_away))

def blend_with_market(home, away, model_home, model_away, market_df):
    if market_df is None or market_df.empty:
        return model_home, model_away, None, None, None
    row = market_df[(market_df["home"] == home) & (market_df["away"] == away)]
    if row.empty:
        return model_home, model_away, None, None, None
    try:
        spread_home = float(row["spread_home"].iloc[0])  # negative = home favored
        total = float(row["total"].iloc[0])
    except Exception:
        return model_home, model_away, None, None, None
    market_home = (total + (-spread_home)) / 2.0
    market_away = total - market_home
    home_pts = W_MARKET*market_home + W_MODEL*model_home
    away_pts = W_MARKET*market_away + W_MODEL*model_away
    return float(home_pts), float(away_pts), market_home, market_away, spread_home

def apply_weather_adjustment(home, away, lam_home, lam_away, weather_df):
    if weather_df is None or weather_df.empty:
        return lam_home, lam_away, ""
    row = weather_df[(weather_df["home"] == home) & (weather_df["away"] == away)]
    if row.empty:
        return lam_home, lam_away, ""
    roof = str(row["roof"].iloc[0]).strip().upper() if pd.notna(row["roof"].iloc[0]) else "OUTDOOR"
    wind = float(row["wind_mph"].iloc[0]) if pd.notna(row["wind_mph"].iloc[0]) else 0.0
    precip = str(row["precip"].iloc[0]).strip().lower() if pd.notna(row["precip"].iloc[0]) else ""

    note_bits = []
    adj_home = adj_away = 0.0

    if roof in DOME_TAGS:
        note_bits.append("Dome: no weather adj")
    else:
        windy = max(0.0, wind - WIND_THRESHOLD)
        if windy > 0:
            pen = windy * WIND_PENALTY_PER_MPH
            adj_home -= pen; adj_away -= pen
            note_bits.append(f"Wind {wind:.0f}mph (−{pen:.1f} each)")
        if precip and precip not in ("0","false","no","none"):
            adj_home -= PRECIP_PENALTY; adj_away -= PRECIP_PENALTY
            note_bits.append(f"Precip (−{PRECIP_PENALTY:.1f} each)")

    new_home = max(6.0, lam_home + adj_home)
    new_away = max(6.0, lam_away + adj_away)
    wx_note = "" if not note_bits else " | Weather: " + ", ".join(note_bits)
    return new_home, new_away, wx_note

def simulate_scores(lambda_home: float, lambda_away: float, n_sims: int = 10_000):
    home_scores = np.random.poisson(lam=lambda_home, size=n_sims)
    away_scores = np.random.poisson(lam=lambda_away, size=n_sims)
    home_wins = (home_scores > away_scores)
    away_wins = (away_scores > home_scores)
    ties = ~(home_wins | away_wins)
    coin = np.random.rand(ties.sum()) < 0.5
    home_wins[ties] = coin
    away_wins[ties] = ~coin
    totals = home_scores + away_scores
    return {
        "home_win_pct": float(home_wins.mean()*100.0),
        "away_win_pct": float(away_wins.mean()*100.0),
        "proj_home_mean": float(home_scores.mean()),
        "proj_away_mean": float(away_scores.mean()),
        "proj_home_median": float(np.median(home_scores)),
        "proj_away_median": float(np.median(away_scores)),
        "total_mean": float(totals.mean()),
        "total_median": float(np.median(totals)),
        "total_p10": float(np.percentile(totals, 10)),
        "total_p90": float(np.percentile(totals, 90)),
    }

def edge_text(favored, home, away, lam_home, lam_away, favored_win, spread_home, wx_note):
    label = edge_label(favored_win)
    diff = round(lam_home) - round(lam_away)  # home minus away
    if diff == 0:
        margin_text = "pick’em"
    else:
        margin_text = f"{home} by {abs(diff)}" if favored == home else f"{away} by {abs(-diff)}"
    return f"{label}: {margin_text}{market_note(spread_home)}{wx_note}"

# ================================================================
# --- Simulate each matchup ---
# ================================================================
rows = []
for _, g in week_df.iterrows():
    home = g["home_team"]
    away = g["away_team"]

    # Model points (EPA + PF/PA + QB + Pace + HFA)
    model_home, model_away = expected_points_from_components(home, away)

    # Market blend if file present
    lam_home, lam_away, mkt_home, mkt_away, spread_home = blend_with_market(
        home, away, model_home, model_away, market_df
    )

    # Weather adjustment if file present
    lam_home, lam_away, wx_note = apply_weather_adjustment(home, away, lam_home, lam_away, weather_df)

    # Simulate
    sim = simulate_scores(lam_home, lam_away, n_sims=N_SIMS)
    home_win = round(sim["home_win_pct"], 1)
    away_win = round(sim["away_win_pct"], 1)
    favored, favored_win = (home, home_win) if home_win >= away_win else (away, away_win)

    # Notes (edge strength + margin + market + weather)
    note = edge_text(favored, home, away, lam_home, lam_away, favored_win, spread_home, wx_note)

    # Totals (tie-breaker for ALL games = median total)
    total_mean   = round(sim["total_mean"], 1)
    total_median = int(round(sim["total_median"]))
    total_p10    = int(round(sim["total_p10"]))
    total_p90    = int(round(sim["total_p90"]))

    rows.append({
        "Matchup": f"{away} @ {home}",
        "Favored": favored,
        "Favored Win %": favored_win,
        "Home": home, "Home Win %": home_win,
        "Away": away, "Away Win %": away_win,
        "Proj Score": f"{away} {pretty_score(sim['proj_away_mean'])} — {home} {pretty_score(sim['proj_home_mean'])}",
        "Median Score": f"{away} {pretty_score(sim['proj_away_median'])} — {home} {pretty_score(sim['proj_home_median'])}",
        "λ_home": round(lam_home, 2), "λ_away": round(lam_away, 2),
        "Model_H": round(model_home, 2), "Model_A": round(model_away, 2),
        "Mkt_H": ("" if mkt_home is None else f"{mkt_home:.2f}"),
        "Mkt_A": ("" if mkt_away is None else f"{mkt_away:.2f}"),
        "Total Median (tiebreaker)": total_median,
        "Total Mean": total_mean,
        "Total P10–P90": f"{total_p10}–{total_p90}",
        "Train≤Week": train_weeks,
        "Notes": note
    })

df = pd.DataFrame(rows)

# Sort by confidence
df = df.sort_values(by="Favored Win %", ascending=False, kind="mergesort").reset_index(drop=True)

# Final columns + formatting
cols = [
    "Matchup", "Favored", "Favored Win %",
    "Home", "Home Win %", "Away", "Away Win %",
    "Proj Score", "Median Score",
    "λ_home", "λ_away", "Model_H", "Model_A", "Mkt_H", "Mkt_A",
    "Total Median (tiebreaker)", "Total Mean", "Total P10–P90",
    "Train≤Week", "Notes"
]
df = df.reindex(columns=cols)

fmt = {
    "Favored Win %": "{:.1f}".format,
    "Home Win %": "{:.1f}".format,
    "Away Win %": "{:.1f}".format,
    "λ_home": "{:.2f}".format,
    "λ_away": "{:.2f}".format,
    "Model_H": "{:.2f}".format,
    "Model_A": "{:.2f}".format,
    "Total Mean": "{:.1f}".format,
}

print(f"\n=== NFL Week {TARGET_WEEK} — {SEASON} (Auto CSV + D-model + EWMA + Weather, {N_SIMS} sims) ===")
print(df.to_string(index=False, formatters=fmt))

out_path = f"{file_stem}_D_model_{SEASON}.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved CSV → {out_path}")
