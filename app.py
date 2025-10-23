# app.py
import os, sys, subprocess, pathlib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="NFL Pick’em Simulator", layout="wide")
st.title("NFL Pick’em Simulator")

# Controls
c1, c2, c3 = st.columns(3)
season = c1.number_input("Season", min_value=2020, max_value=2035, value=2025, step=1)
week   = c2.number_input("Week", min_value=1, max_value=22, value=8, step=1)
sims   = c3.number_input("Simulations per game", min_value=1000, max_value=500000, value=10000, step=1000)

st.write("Click **Run** to generate market & weather (if missing) and run the sims.")

# Allow ODDS_API_KEY from Streamlit Secrets (Cloud) or a textbox (local)
if "ODDS_API_KEY" in st.secrets:
    os.environ["ODDS_API_KEY"] = st.secrets["ODDS_API_KEY"]

manual_key = st.text_input("Optional: Override ODDS_API_KEY for this run", type="password")
if manual_key:
    os.environ["ODDS_API_KEY"] = manual_key

run = st.button("Run")

if run:
    file_stem = f"week{week}"
    out_csv = f"{file_stem}_D_model_{season}.csv"

    cmd = [sys.executable, "pickem.py", "--season", str(season), "--week", str(week), "--sims", str(sims)]
    with st.spinner("Running simulations…"):
        proc = subprocess.run(cmd, capture_output=True, text=True)

    # Show logs
    st.subheader("Run Logs")
    st.code(proc.stdout or "", language="bash")
    if proc.returncode != 0:
        st.error("Run failed.")
        if proc.stderr:
            st.code(proc.stderr, language="bash")
    else:
        p = pathlib.Path(out_csv)
        if not p.exists():
            st.warning(f"Expected output not found: {out_csv}")
        else:
            df = pd.read_csv(p)
            st.success("Done!")
            st.download_button("Download CSV", data=df.to_csv(index=False), file_name=out_csv, mime="text/csv")
            st.dataframe(df, use_container_width=True)

st.caption("Hint: Odds are blended from The Odds API if ODDS_API_KEY is set; weather via Open-Meteo.")
