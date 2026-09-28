from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

APP_TITLE = "RS3 Merchant Terminal"
API_BASE = "https://prices.runescape.wiki/api/v2/rs"
DB_PATH = Path(os.getenv("RS3_MERCH_DB", "rs3_merch.db"))
USER_AGENT = os.getenv("RS3_USER_AGENT", "RS3MerchantTerminal/1.0 (personal Streamlit merching tool)")
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}
REQUEST_TIMEOUT = 20

st.set_page_config(page_title=APP_TITLE, page_icon="💰", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root { --gold:#f5c451; --green:#32d583; --red:#f97066; --panel:#121923; }
.stApp { background: radial-gradient(circle at 15% 0%, #172233 0%, #0b1017 45%, #070a0f 100%); }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(27,38,55,.96),rgba(12,18,27,.96)); border:1px solid rgba(245,196,81,.2); padding:16px; border-radius:18px; box-shadow:0 12px 35px rgba(0,0,0,.25); }
[data-testid="stSidebar"] { background:#090e15; border-right:1px solid rgba(255,255,255,.08); }
h1,h2,h3 { letter-spacing:-.025em; }
.gold { color:var(--gold); }
.pill { display:inline-block; padding:5px 10px; border-radius:999px; background:rgba(50,213,131,.12); color:#73e2ad; border:1px solid rgba(50,213,131,.24); font-size:.82rem; }
.small-note { opacity:.72; font-size:.86rem; }
</style>
""", unsafe_allow_html=True)


def db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript("""
    CREATE TABLE IF NOT EXISTS snapshots (
      captured_at INTEGER NOT NULL, item_id INTEGER NOT NULL, high REAL, low REAL,
      high_time INTEGER, low_time INTEGER, high_volume REAL, low_volume REAL,
      PRIMARY KEY (captured_at, item_id)
    );
    CREATE INDEX IF NOT EXISTS idx_snap_item_time ON snapshots(item_id, captured_at);
    CREATE TABLE IF NOT EXISTS portfolio (
      id INTEGER PRIMARY KEY AUTOINCREMENT, item_id INTEGER NOT NULL, item_name TEXT NOT NULL,
      quantity INTEGER NOT NULL, buy_price REAL NOT NULL, opened_at TEXT NOT NULL, notes TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS flips (
      id INTEGER PRIMARY KEY AUTOINCREMENT, item_id INTEGER, item_name TEXT NOT NULL,
      quantity INTEGER NOT NULL, buy_price REAL NOT NULL, sell_price REAL NOT NULL,
      profit REAL NOT NULL, closed_at TEXT NOT NULL, notes TEXT DEFAULT ''
    );
    CREATE TABLE IF NOT EXISTS alerts (
      id INTEGER PRIMARY KEY AUTOINCREMENT, item_id INTEGER NOT NULL, item_name TEXT NOT NULL,
      condition TEXT NOT NULL, target REAL NOT NULL, enabled INTEGER NOT NULL DEFAULT 1
    );
    """)
    con.commit()
    return con


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_mapping() -> pd.DataFrame:
    r = requests.get(f"{API_BASE}/mapping", headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["id", "name", "limit", "members", "value", "examine", "icon"])
    for col in ["id", "limit", "value"]:
        if col not in df: df[col] = None
    for col in ["name", "examine", "icon"]:
        if col not in df: df[col] = ""
    if "members" not in df: df["members"] = False
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    return df.dropna(subset=["id", "name"])


def api_json(route: str, params: dict | None = None) -> dict:
    r = requests.get(f"{API_BASE}/{route}", params=params, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_market() -> pd.DataFrame:
    latest = api_json("latest").get("data", {})
    five = api_json("5m").get("data", {})
    mapping = fetch_mapping().copy()
    if mapping.empty:
        return pd.DataFrame()
    latest_rows, five_rows = [], []
    for key, v in latest.items():
        latest_rows.append({"id": int(key), **v})
    for key, v in five.items():
        five_rows.append({"id": int(key), **v})
    ldf = pd.DataFrame(latest_rows)
    fdf = pd.DataFrame(five_rows)
    if ldf.empty:
        return pd.DataFrame()
    ldf = ldf.rename(columns={"high":"instant_buy", "low":"instant_sell", "highTime":"buy_time", "lowTime":"sell_time"})
    fdf = fdf.rename(columns={"avgHighPrice":"avg_buy_5m", "avgLowPrice":"avg_sell_5m", "highPriceVolume":"buy_volume_5m", "lowPriceVolume":"sell_volume_5m"})
    keep5 = [c for c in ["id","avg_buy_5m","avg_sell_5m","buy_volume_5m","sell_volume_5m"] if c in fdf.columns]
    df = mapping.merge(ldf, on="id", how="inner").merge(fdf[keep5], on="id", how="left")
    for c in ["instant_buy","instant_sell","avg_buy_5m","avg_sell_5m","buy_volume_5m","sell_volume_5m","limit"]:
        if c not in df: df[c] = math.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # The latest high is the latest high-side trade; latest low is the latest low-side trade.
    df["margin"] = df["instant_buy"] - df["instant_sell"]
    df["roi_pct"] = (df["margin"] / df["instant_sell"].replace(0, math.nan)) * 100
    df["volume_5m"] = df[["buy_volume_5m","sell_volume_5m"]].fillna(0).sum(axis=1)
    df["limit_profit"] = df["margin"] * df["limit"]
    recency = pd.concat([df.get("buy_time", pd.Series(index=df.index)), df.get("sell_time", pd.Series(index=df.index))], axis=1).max(axis=1)
    df["age_sec"] = (int(time.time()) - pd.to_numeric(recency, errors="coerce")).clip(lower=0)
    # Ranking is a heuristic, not a promise of fill or profit.
    positive = df["margin"].clip(lower=0)
    liquidity = pd.Series(pd.np.log1p(df["volume_5m"].clip(lower=0)) if hasattr(pd, "np") else __import__("numpy").log1p(df["volume_5m"].clip(lower=0)), index=df.index)
    roi_quality = df["roi_pct"].clip(lower=0, upper=15) / 15
    liq_quality = liquidity / max(float(liquidity.max() or 1), 1)
    freshness = (1 - (df["age_sec"] / 3600).clip(lower=0, upper=1))
    df["opportunity_score"] = (roi_quality * .45 + liq_quality * .40 + freshness * .15) * 100
    df.loc[positive <= 0, "opportunity_score"] = 0
    return df


def save_snapshot(df: pd.DataFrame) -> None:
    if df.empty: return
    captured = int(time.time())
    snap = pd.DataFrame({
        "captured_at": captured, "item_id": df["id"], "high": df["instant_buy"], "low": df["instant_sell"],
        "high_time": df.get("buy_time"), "low_time": df.get("sell_time"),
        "high_volume": df.get("buy_volume_5m"), "low_volume": df.get("sell_volume_5m")
    })
    con = db()
    try:
        rows = [
            tuple(row)
            for row in snap[[
                "captured_at", "item_id", "high", "low",
                "high_time", "low_time", "high_volume", "low_volume"
            ]].itertuples(index=False, name=None)
        ]
        con.executemany(
            """
            INSERT OR REPLACE INTO snapshots (
                captured_at, item_id, high, low, high_time, low_time,
                high_volume, low_volume
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        con.commit()
    finally:
        con.close()


def fmt_gp(x) -> str:
    if x is None or pd.isna(x): return "—"
    x = float(x); a = abs(x)
    for value, suffix in [(1e12,"T"),(1e9,"B"),(1e6,"M"),(1e3,"K")]:
        if a >= value: return f"{x/value:,.2f}{suffix} gp"
    return f"{x:,.0f} gp"


def unix_age(ts) -> str:
    if ts is None or pd.isna(ts): return "unknown"
    seconds = max(0, int(time.time()) - int(ts))
    if seconds < 60: return f"{seconds}s ago"
    if seconds < 3600: return f"{seconds//60}m ago"
    return f"{seconds//3600}h ago"


def item_options(mapping: pd.DataFrame) -> dict[str, int]:
    return dict(zip(mapping.sort_values("name")["name"].astype(str), mapping.sort_values("name")["id"].astype(int)))


def market_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["name","instant_sell","instant_buy","margin","roi_pct","volume_5m","limit","limit_profit","opportunity_score","age_sec"]].copy()
    out.columns = ["Item","Buy for","Sell for","Margin","ROI %","5m Volume","GE Limit","Limit Profit","Score","Age (sec)"]
    return out


def history_chart(item_id: int, hours: int = 24):
    timestep = "5m" if hours <= 24 else "1h"
    points = api_json("timeseries", {"timestep": timestep, "id": item_id}).get("data", [])
    h = pd.DataFrame(points)
    if h.empty: return None
    h["time"] = pd.to_datetime(h["timestamp"], unit="s", utc=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=hours)
    h = h[h["time"] >= cutoff]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["time"], y=h["avgHighPrice"], name="High-side average", line=dict(color="#32d583", width=2)))
    fig.add_trace(go.Scatter(x=h["time"], y=h["avgLowPrice"], name="Low-side average", line=dict(color="#f5c451", width=2)))
    fig.update_layout(height=410, margin=dict(l=10,r=10,t=15,b=10), template="plotly_dark", hovermode="x unified", yaxis_title="GP", legend=dict(orientation="h"))
    return fig


mapping = fetch_mapping()
con = db(); con.close()

with st.sidebar:
    st.markdown("## 💰 RS3 Merchant Terminal")
    st.caption("Real-time opportunity discovery and flip tracking")
    nav = st.radio("Workspace", ["Live Market", "Flip Finder", "Item Analyzer", "Budget Builder", "Portfolio", "Profit Journal", "Alerts", "Settings"], label_visibility="collapsed")
    st.divider()
    auto_refresh = st.toggle("5-second auto-refresh", value=True)
    st.caption("The screen refreshes every 5 seconds. Source trades only change when new market data arrives.")
    if st.button("Refresh now", use_container_width=True): st.rerun()


def render_live_content():
    try:
        market = fetch_market()
        st.session_state["market"] = market
        st.session_state["last_ok"] = datetime.now(timezone.utc)
        save_snapshot(market)
    except Exception as e:
        market = st.session_state.get("market", pd.DataFrame())
        st.warning(f"Live feed error: {e}. Showing the last successful response." if not market.empty else f"Could not load prices: {e}")
    if market.empty: return

    latest_trade = int(pd.concat([market.get("buy_time", pd.Series(dtype=float)), market.get("sell_time", pd.Series(dtype=float))]).max())
    profitable = market[(market["margin"] > 0) & (market["volume_5m"] > 0)]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Items live", f"{len(market):,}")
    c2.metric("Positive spreads", f"{len(profitable):,}")
    c3.metric("Latest trade", unix_age(latest_trade))
    c4.metric("Feed refresh", datetime.now().strftime("%I:%M:%S %p"))

    if nav in ["Live Market", "Flip Finder"]:
        left, mid, right = st.columns([1.6,1,1])
        query = left.text_input("Search items", placeholder="Search by item name...")
        min_roi = mid.number_input("Minimum ROI %", min_value=0.0, value=0.5, step=0.25)
        min_volume = right.number_input("Minimum 5m volume", min_value=0, value=1, step=10)
        f = market[(market["margin"] > 0) & (market["roi_pct"] >= min_roi) & (market["volume_5m"] >= min_volume)].copy()
        if query: f = f[f["name"].str.contains(query, case=False, na=False)]
        if nav == "Flip Finder":
            max_cap = st.number_input("Maximum GP needed for one GE-limit cycle", min_value=0, value=250_000_000, step=1_000_000)
            required = f["instant_sell"] * f["limit"]
            f = f[(required <= max_cap) | f["limit"].isna()]
            f = f.sort_values(["opportunity_score","limit_profit"], ascending=False)
            st.markdown('<span class="pill">Heuristic score balances ROI, recent volume, and freshness</span>', unsafe_allow_html=True)
        else:
            sort_choice = st.selectbox("Sort", ["Opportunity score", "Margin", "ROI", "5m volume", "Limit profit"], label_visibility="collapsed")
            sort_map = {"Opportunity score":"opportunity_score","Margin":"margin","ROI":"roi_pct","5m volume":"volume_5m","Limit profit":"limit_profit"}
            f = f.sort_values(sort_map[sort_choice], ascending=False)
        st.dataframe(market_display(f.head(300)), use_container_width=True, hide_index=True, height=620,
            column_config={"Buy for":st.column_config.NumberColumn(format="localized"),"Sell for":st.column_config.NumberColumn(format="localized"),"Margin":st.column_config.NumberColumn(format="localized"),"ROI %":st.column_config.NumberColumn(format="%.2f%%"),"Score":st.column_config.ProgressColumn(min_value=0,max_value=100,format="%.1f"),"Limit Profit":st.column_config.NumberColumn(format="localized")})
        st.caption("Prices represent recent high-side and low-side trades, not guaranteed offers. Always perform a small margin check before committing GP.")

    elif nav == "Item Analyzer":
        opts = item_options(mapping)
        name = st.selectbox("Item", list(opts), index=None, placeholder="Choose an item")
        if name:
            row = market[market["id"] == opts[name]]
            if row.empty: st.info("No recent real-time trade data is available for this item."); return
            r = row.iloc[0]
            a,b,c,d = st.columns(4)
            a.metric("Suggested buy reference", fmt_gp(r["instant_sell"]))
            b.metric("Suggested sell reference", fmt_gp(r["instant_buy"]))
            c.metric("Observed spread", fmt_gp(r["margin"]), f"{r['roi_pct']:.2f}% ROI")
            d.metric("5-minute volume", f"{r['volume_5m']:,.0f}")
            h = st.segmented_control("History", options=[6,24,168,720], default=24, format_func=lambda x: {6:"6 hours",24:"24 hours",168:"7 days",720:"30 days"}[x])
            fig = history_chart(int(r["id"]), int(h or 24))
            if fig: st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"**Buy-side timestamp:** {unix_age(r.get('sell_time'))} · **Sell-side timestamp:** {unix_age(r.get('buy_time'))} · **GE limit:** {r['limit']:,.0f}" if pd.notna(r['limit']) else "GE limit unavailable")

    elif nav == "Budget Builder":
        budget = st.number_input("Available cash", min_value=1_000, value=100_000_000, step=1_000_000)
        risk = st.select_slider("Risk profile", options=["Conservative","Balanced","Aggressive"], value="Balanced")
        roi_floor = {"Conservative":0.5,"Balanced":1.0,"Aggressive":2.0}[risk]
        volume_floor = {"Conservative":50,"Balanced":10,"Aggressive":1}[risk]
        f = market[(market["margin"] > 0) & (market["roi_pct"] >= roi_floor) & (market["volume_5m"] >= volume_floor) & (market["instant_sell"] > 0)].copy()
        f["Affordable Qty"] = (budget / f["instant_sell"]).fillna(0).astype(int)
        f["Affordable Qty"] = f[["Affordable Qty","limit"]].min(axis=1).fillna(f["Affordable Qty"]).astype(int)
        f["Capital Needed"] = f["Affordable Qty"] * f["instant_sell"]
        f["Projected Spread"] = f["Affordable Qty"] * f["margin"]
        f = f[f["Affordable Qty"] > 0].sort_values(["opportunity_score","Projected Spread"], ascending=False).head(50)
        st.dataframe(f[["name","Affordable Qty","instant_sell","instant_buy","Capital Needed","Projected Spread","roi_pct","volume_5m","opportunity_score"]].rename(columns={"name":"Item","instant_sell":"Buy for","instant_buy":"Sell for","roi_pct":"ROI %","volume_5m":"5m Volume","opportunity_score":"Score"}), use_container_width=True, hide_index=True)
        st.caption("Projected spread assumes every item buys and sells at the displayed references. It excludes price movement and incomplete fills.")

    elif nav == "Portfolio":
        opts=item_options(mapping)
        with st.form("add_position", clear_on_submit=True):
            c1,c2,c3,c4=st.columns([2,1,1,2])
            name=c1.selectbox("Item",list(opts)); qty=c2.number_input("Quantity",1,step=1); price=c3.number_input("Buy price",1,step=1); notes=c4.text_input("Notes")
            if st.form_submit_button("Add position"):
                con=db(); con.execute("INSERT INTO portfolio(item_id,item_name,quantity,buy_price,opened_at,notes) VALUES(?,?,?,?,?,?)",(opts[name],name,qty,price,datetime.now().isoformat(),notes)); con.commit(); con.close(); st.rerun()
        con=db(); p=pd.read_sql_query("SELECT * FROM portfolio ORDER BY id DESC",con); con.close()
        if p.empty: st.info("No open positions yet.")
        else:
            p=p.merge(market[["id","instant_sell","instant_buy"]],left_on="item_id",right_on="id",how="left")
            p["cost"]=p.quantity*p.buy_price; p["market_value"]=p.quantity*p.instant_buy; p["unrealized"]=p.market_value-p.cost
            c1,c2,c3=st.columns(3); c1.metric("Invested",fmt_gp(p.cost.sum())); c2.metric("Reference value",fmt_gp(p.market_value.sum())); c3.metric("Unrealized spread",fmt_gp(p.unrealized.sum()))
            st.dataframe(p[["id_x","item_name","quantity","buy_price","instant_buy","cost","market_value","unrealized","opened_at","notes"]].rename(columns={"id_x":"Position ID","item_name":"Item","quantity":"Qty","buy_price":"Bought at","instant_buy":"Current sell ref","cost":"Cost","market_value":"Reference value","unrealized":"Unrealized","opened_at":"Opened","notes":"Notes"}),use_container_width=True,hide_index=True)
            close_id=st.number_input("Position ID to close/remove",min_value=0,step=1)
            if st.button("Remove position") and close_id:
                con=db(); con.execute("DELETE FROM portfolio WHERE id=?",(close_id,)); con.commit(); con.close(); st.rerun()

    elif nav == "Profit Journal":
        opts=item_options(mapping)
        with st.form("log_flip", clear_on_submit=True):
            c1,c2,c3,c4=st.columns(4); name=c1.selectbox("Item",list(opts)); qty=c2.number_input("Quantity",1,step=1); buy=c3.number_input("Buy price",1,step=1); sell=c4.number_input("Sell price",1,step=1)
            notes=st.text_input("Notes")
            if st.form_submit_button("Log completed flip"):
                profit=(sell-buy)*qty; con=db(); con.execute("INSERT INTO flips(item_id,item_name,quantity,buy_price,sell_price,profit,closed_at,notes) VALUES(?,?,?,?,?,?,?,?)",(opts[name],name,qty,buy,sell,profit,datetime.now().isoformat(),notes)); con.commit(); con.close(); st.rerun()
        con=db(); flips=pd.read_sql_query("SELECT * FROM flips ORDER BY id DESC",con); con.close()
        if flips.empty: st.info("No completed flips logged yet.")
        else:
            a,b,c=st.columns(3); a.metric("Lifetime profit",fmt_gp(flips.profit.sum())); b.metric("Completed flips",f"{len(flips):,}"); c.metric("Positive flips",f"{(flips.profit>0).mean()*100:.1f}%")
            st.dataframe(flips,use_container_width=True,hide_index=True)
            st.download_button("Export journal CSV",flips.to_csv(index=False),"rs3_flip_journal.csv","text/csv")

    elif nav == "Alerts":
        opts=item_options(mapping)
        with st.form("new_alert",clear_on_submit=True):
            c1,c2,c3=st.columns([2,1,1]); name=c1.selectbox("Item",list(opts)); condition=c2.selectbox("Condition",["margin_at_least","roi_at_least","buy_price_at_most","sell_price_at_least"]); target=c3.number_input("Target",min_value=0.0,step=1.0)
            if st.form_submit_button("Create alert"):
                con=db(); con.execute("INSERT INTO alerts(item_id,item_name,condition,target) VALUES(?,?,?,?)",(opts[name],name,condition,target)); con.commit(); con.close(); st.rerun()
        con=db(); alerts=pd.read_sql_query("SELECT * FROM alerts WHERE enabled=1 ORDER BY id DESC",con); con.close()
        if alerts.empty: st.info("No active alerts.")
        else:
            results=[]
            for _,a in alerts.iterrows():
                r=market[market.id==a.item_id]
                if r.empty: continue
                r=r.iloc[0]; values={"margin_at_least":r.margin,"roi_at_least":r.roi_pct,"buy_price_at_most":r.instant_sell,"sell_price_at_least":r.instant_buy}
                hit=(values[a.condition] <= a.target) if a.condition=="buy_price_at_most" else (values[a.condition] >= a.target)
                results.append({"ID":a.id,"Item":a.item_name,"Condition":a.condition,"Target":a.target,"Current":values[a.condition],"Triggered":hit})
                if hit: st.success(f"🔔 {a.item_name}: {a.condition} triggered. Current {values[a.condition]:,.2f}, target {a.target:,.2f}.")
            st.dataframe(pd.DataFrame(results),use_container_width=True,hide_index=True)
            disable=st.number_input("Alert ID to disable",min_value=0,step=1)
            if st.button("Disable alert") and disable:
                con=db(); con.execute("UPDATE alerts SET enabled=0 WHERE id=?",(disable,)); con.commit(); con.close(); st.rerun()

    elif nav == "Settings":
        st.markdown("### Data and behavior")
        st.code(f"API: {API_BASE}\nDatabase: {DB_PATH.resolve()}\nRefresh interval: 5 seconds\nUser-Agent: {USER_AGENT}")
        st.info("Set the RS3_USER_AGENT environment variable to a descriptive identifier with optional contact information before deploying publicly.")
        if st.button("Clear cached item mapping"): st.cache_data.clear(); st.rerun()


if auto_refresh and hasattr(st, "fragment"):
    @st.fragment(run_every=5)
    def live_fragment(): render_live_content()
    live_fragment()
else:
    render_live_content()
    if auto_refresh:
        st.caption("Install a recent Streamlit release to enable the 5-second fragment refresh.")

st.divider()
st.caption("Community market data is informational and may be delayed, sparse, or volatile. This tool cannot guarantee fills or profit and is not affiliated with Jagex.")
