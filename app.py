"""IMF Exchange Rate Explorer.

Compare nominal and real bilateral exchange rates for any two IMF member
countries (or the Euro area, G163) using monthly IMF CPI and exchange-rate data.

Definitions
-----------
IMF exchange rates are local-currency units per USD:  E_i(t) = XDC_USD for country i.

    S(t)    = E2(t) / E1(t)            units of Country 2 currency per 1 unit of Country 1 currency
    RER(t)  = S(t) * P1(t) / P2(t)
    RER*    = mean of RER(t) over valid months in the selected sample
    Index(t)= 100 * RER(t) / RER*

Run with:  streamlit run app.py
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

BASE_URL = "https://api.imf.org/external/sdmx/3.0"
AGENCY = "IMF.STA"
CPI_FLOW = "CPI"
FX_FLOW = "ER"
FIRST_MONTH = pd.Timestamp("1976-01-01")
EURO_AREA = "G163"
# Codes the IMF (or older IMF datasets) may use for the euro area; all are tried for every euro selection.
EURO_CODES = ["G163", "EA", "U2", "EMU"]
CACHE_TTL = 6 * 3600  # seconds

ACCEPT_HEADERS = [
    "application/vnd.sdmx.data+json;version=2.0.0",
    "application/vnd.sdmx.data+json",
    "application/json",
]

# Used only if the live country codelist cannot be retrieved.
FALLBACK_COUNTRIES = {
    "ARG": "Argentina", "AUS": "Australia", "AUT": "Austria", "BEL": "Belgium",
    "BGD": "Bangladesh", "BRA": "Brazil", "CAN": "Canada", "CHE": "Switzerland",
    "CHL": "Chile", "CHN": "China, P.R.: Mainland", "COL": "Colombia",
    "CZE": "Czech Republic", "DEU": "Germany", "DNK": "Denmark",
    "DZA": "Algeria", "EGY": "Egypt", "ESP": "Spain", "FIN": "Finland",
    "FRA": "France", "GBR": "United Kingdom", "GHA": "Ghana", "GRC": "Greece",
    "HKG": "China, P.R.: Hong Kong", "HUN": "Hungary", "IDN": "Indonesia",
    "IND": "India", "IRL": "Ireland", "IRN": "Iran", "ISL": "Iceland",
    "ISR": "Israel", "ITA": "Italy", "JPN": "Japan", "KEN": "Kenya",
    "KOR": "Korea, Republic of", "LKA": "Sri Lanka", "MAR": "Morocco",
    "MEX": "Mexico", "MYS": "Malaysia", "NGA": "Nigeria", "NLD": "Netherlands",
    "NOR": "Norway", "NZL": "New Zealand", "PAK": "Pakistan", "PER": "Peru",
    "PHL": "Philippines", "POL": "Poland", "PRT": "Portugal", "ROU": "Romania",
    "RUS": "Russian Federation", "SAU": "Saudi Arabia", "SGP": "Singapore",
    "SWE": "Sweden", "THA": "Thailand", "TUR": "Türkiye", "TWN": "Taiwan",
    "UKR": "Ukraine", "URY": "Uruguay", "USA": "United States",
    "VEN": "Venezuela", "VNM": "Vietnam", "ZAF": "South Africa",
    EURO_AREA: "Euro area",
}


def _empty_series() -> pd.Series:
    """Empty float Series with a DatetimeIndex (so date comparisons/reindexing work)."""
    return pd.Series(dtype=float, index=pd.DatetimeIndex([]))


class IMFError(RuntimeError):
    """Raised for network / API problems that are not simply 'no data'."""


# --------------------------------------------------------------------------
# IMF API access
# --------------------------------------------------------------------------
def _http_get(url: str) -> requests.Response | None:
    """GET with content-negotiation fallbacks and light retry. None on 404."""
    last_err = "unknown error"
    for accept in ACCEPT_HEADERS:
        for attempt in range(3):
            try:
                resp = requests.get(url, headers={"Accept": accept}, timeout=60)
            except requests.RequestException as exc:
                last_err = str(exc)
                time.sleep(1.5 * (attempt + 1))
                continue
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return None  # SDMX convention: no matching data
            if resp.status_code in (406, 415):
                last_err = f"HTTP {resp.status_code} for Accept '{accept}'"
                break  # try next Accept header
            if resp.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {resp.status_code}"
                time.sleep(2 * (attempt + 1))
                continue
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            break
    raise IMFError(f"IMF API request failed ({last_err}) for {url}")


_TIME_RE = re.compile(r"^(\d{4})-?M?(\d{2})$")


def _parse_month(label: str) -> pd.Timestamp | None:
    m = _TIME_RE.match(str(label).strip())
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    if not 1 <= month <= 12:
        return None
    return pd.Timestamp(year=year, month=month, day=1)


def _label(v: dict) -> str:
    return str(v.get("id") or v.get("value") or v.get("name") or "")


def parse_sdmx_json(payload: dict) -> pd.Series:
    """Turn an SDMX-JSON data message into a monthly float Series.

    Handles both the series layout (observations nested under each series)
    and the flat layout (dimensionAtObservation=AllDimensions).
    """
    root = payload.get("data", payload)
    datasets = root.get("dataSets") or []
    if not datasets:
        return _empty_series()

    structure = root.get("structures", root.get("structure"))
    if isinstance(structure, list):
        structure = structure[0] if structure else {}
    structure = structure or {}
    obs_dims = (structure.get("dimensions") or {}).get("observation", [])
    if not obs_dims:
        return _empty_series()

    ds = datasets[0]
    records: dict[pd.Timestamp, float] = {}

    if ds.get("series"):
        time_dim = next((d for d in obs_dims if d.get("id") == "TIME_PERIOD"), obs_dims[0])
        time_values = time_dim.get("values", [])
        # Key is fully specified, so normally one series; keep the richest one.
        best = max((s.get("observations", {}) for s in ds["series"].values()), key=len)
        for idx, arr in best.items():
            ts = _parse_month(_label(time_values[int(idx)]))
            if ts is None or not arr or arr[0] is None:
                continue
            try:
                records[ts] = float(arr[0])
            except (TypeError, ValueError):
                continue
    elif ds.get("observations"):
        ids = [d.get("id") for d in obs_dims]
        pos = ids.index("TIME_PERIOD") if "TIME_PERIOD" in ids else 0
        time_values = obs_dims[pos].get("values", [])
        for key, arr in ds["observations"].items():
            ts = _parse_month(_label(time_values[int(key.split(":")[pos])]))
            if ts is None or not arr or arr[0] is None:
                continue
            try:
                records[ts] = float(arr[0])
            except (TypeError, ValueError):
                continue

    if not records:
        return _empty_series()
    out = pd.Series(records, dtype=float).sort_index()
    out.index = pd.DatetimeIndex(out.index)
    return out


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def fetch_series(flow: str, key: str) -> pd.Series:
    """Fetch one IMF series (monthly). Empty Series if the IMF has no data."""
    url = f"{BASE_URL}/data/dataflow/{AGENCY}/{flow}/+/{key}"
    resp = _http_get(url)
    if resp is None:
        return _empty_series()
    try:
        series = parse_sdmx_json(resp.json())
    except ValueError as exc:
        raise IMFError(f"Could not parse IMF response for {key}: {exc}") from exc
    if series.empty:
        return _empty_series()
    return series[series.index >= FIRST_MONTH]


@st.cache_data(show_spinner=False, ttl=CACHE_TTL)
def load_country_list() -> tuple[dict[str, str], bool]:
    """Return ({ISO3 or G163: name}, loaded_live)."""
    try:
        resp = _http_get(f"{BASE_URL}/structure/codelist/IMF/CL_COUNTRY/+")
        if resp is None:
            raise IMFError("codelist not found")
        data = resp.json()
        root = data.get("data", data)
        codelists = root.get("codelists") or []
        codes = codelists[0].get("codes", []) if codelists else []
        out: dict[str, str] = {}
        for c in codes:
            cid = str(c.get("id", ""))
            if re.fullmatch(r"[A-Z]{3}", cid) or cid in EURO_CODES:
                name = c.get("name") or (c.get("names") or {}).get("en") or cid
                out[cid] = str(name)
        if len(out) >= 50:
            return dict(sorted(out.items(), key=lambda kv: kv[1])), True
    except Exception:  # noqa: BLE001 - any failure -> fallback list
        pass
    return dict(sorted(FALLBACK_COUNTRIES.items(), key=lambda kv: kv[1])), False


# --------------------------------------------------------------------------
# Calculations
# --------------------------------------------------------------------------
@dataclass
class CountryData:
    iso3: str
    cpi: pd.Series
    fx: pd.Series  # local currency units per USD
    cpi_source: str = ""  # human-readable description of what was actually used
    fx_source: str = ""
    notes: list[str] | None = None


# Alternative IMF codes tried (in order) if the primary key returns nothing.
# (code, first_month_to_keep or None, note shown to the user when used)
# The euro area is published as a harmonised index (HICP) rather than a CPI; other countries may be too.
EURO_CPI_HICP_SUFFIX = "HICP._T.IX.M"  # also used as a fallback for every other country
EURO_CPI_ALTERNATIVES = [
    (c, None, "" if c == EURO_AREA else f"Euro area CPI taken from code {c}.") for c in EURO_CODES
]
EURO_FX_ALTERNATIVES = [
    (c, None, "" if c == EURO_AREA else f"Euro area exchange rate taken from code {c}.") for c in EURO_CODES
] + [
    (
        "DEU",
        pd.Timestamp("1999-01-01"),
        "No Euro-area exchange-rate series found at the IMF; used Germany's EUR-per-USD "
        "rate (identical for all euro members) from January 1999 only.",
    ),
]


def _first_available(flow: str, suffixes: list[str], alternatives, default_code: str):
    """Try each (code, min_month, note) x suffix until one returns data.

    Returns (series, source description, notes). `suffixes` are tried in order for each code,
    so the preferred key wins; a non-first suffix adds an explanatory note.
    """
    for code, min_month, note in alternatives:
        for i, suffix in enumerate(suffixes):
            s = fetch_series(flow, f"{code}.{suffix}")
            if min_month is not None and not s.empty:
                s = s[s.index >= min_month]
            if not s.empty:
                notes = [note] if note else []
                if i > 0:
                    notes.append(f"Used the series {code}.{suffix} (the standard {suffixes[0]} key had no data).")
                return s, f"{flow}: {code}.{suffix}", notes
    tried_codes = "/".join(code for code, _, _ in alternatives)
    return _empty_series(), f"{flow}: no data under codes {tried_codes} with keys {' or '.join(suffixes)}", []


def load_country(iso3: str) -> CountryData:
    cpi_suffix, fx_suffix = "CPI._T.IX.M", "XDC_USD.PA_RT.M"
    if iso3 in EURO_CODES:
        cpi, cpi_src, n1 = _first_available(CPI_FLOW, [cpi_suffix, EURO_CPI_HICP_SUFFIX], EURO_CPI_ALTERNATIVES, iso3)
        fx, fx_src, n2 = _first_available(FX_FLOW, [fx_suffix], EURO_FX_ALTERNATIVES, iso3)
        return CountryData(iso3, cpi, fx, cpi_src, fx_src, n1 + n2)
    # Any other country: standard CPI key first, then the harmonised index (HICP) if CPI is empty.
    own = [(iso3, None, "")]
    cpi, cpi_src, n1 = _first_available(CPI_FLOW, [cpi_suffix, EURO_CPI_HICP_SUFFIX], own, iso3)
    fx, fx_src, n2 = _first_available(FX_FLOW, [fx_suffix], own, iso3)
    return CountryData(iso3, cpi, fx, cpi_src, fx_src, n1 + n2)


def build_frame(c1: CountryData, c2: CountryData) -> pd.DataFrame:
    """Monthly frame of raw inputs for both countries (NaN where missing)."""
    series = [c1.cpi, c2.cpi, c1.fx, c2.fx]
    non_empty = [s for s in series if not s.empty]
    if not non_empty:
        return pd.DataFrame()
    start = max(FIRST_MONTH, min(s.index.min() for s in non_empty))
    end = max(s.index.max() for s in non_empty)
    idx = pd.date_range(start, end, freq="MS")
    df = pd.DataFrame(
        {
            "fx1_per_usd": c1.fx.reindex(idx),
            "fx2_per_usd": c2.fx.reindex(idx),
            "cpi1": c1.cpi.reindex(idx),
            "cpi2": c2.cpi.reindex(idx),
        },
        index=idx,
    )
    # The IMF may not publish a USD/USD series; it is identically 1.
    for col, c in (("fx1_per_usd", c1), ("fx2_per_usd", c2)):
        if c.iso3 == "USA" and c.fx.empty:
            df[col] = 1.0
    return df


def compute(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, float, int]:
    """Slice to [start, end] and compute S, RER, RER*, Real Index.

    Returns (sample_df, rer_star, n_valid_months). rer_star is NaN if no valid month.
    """
    out = df.loc[(df.index >= start) & (df.index <= end)].copy()
    pos = lambda s: s.where(s > 0)  # noqa: E731 - non-positive values are invalid
    out["S"] = pos(out["fx2_per_usd"]) / pos(out["fx1_per_usd"])
    out["RER"] = out["S"] * pos(out["cpi1"]) / pos(out["cpi2"])
    out["RER"] = out["RER"].replace([np.inf, -np.inf], np.nan)
    valid = out["RER"].notna()
    n_valid = int(valid.sum())
    rer_star = float(out.loc[valid, "RER"].mean()) if n_valid else float("nan")
    out["RER_star"] = rer_star
    out["Real_Index"] = 100.0 * out["RER"] / rer_star if n_valid else np.nan
    out["valid_rer_month"] = valid
    return out, rer_star, n_valid


def make_csv(sample: pd.DataFrame, iso1: str, iso2: str) -> bytes:
    out = sample.copy()
    out.index.name = "month"
    out.insert(0, "country_2", iso2)
    out.insert(0, "country_1", iso1)
    out = out.rename(
        columns={
            "fx1_per_usd": f"{iso1}_currency_per_USD",
            "fx2_per_usd": f"{iso2}_currency_per_USD",
            "cpi1": f"CPI_{iso1}",
            "cpi2": f"CPI_{iso2}",
            "S": f"S_{iso2}_per_{iso1}",
            "RER": "RER",
            "RER_star": "RER_star",
            "Real_Index": "Real_Index",
        }
    )
    out.index = out.index.strftime("%Y-%m")
    buf = io.StringIO()
    out.to_csv(buf)
    return buf.getvalue().encode("utf-8")


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------
def make_figure(sample: pd.DataFrame, mode: str, n1: str, n2: str, iso1: str, iso2: str) -> go.Figure:
    custom = np.column_stack([sample["S"], sample["RER"], sample["Real_Index"]])
    s_label = f"S(t) [{iso2} per 1 {iso1}]"
    date_line = "<b>%{x|%b %Y}</b>"
    s_line = f"{s_label}: %{{customdata[0]:,.6g}}"
    rer_line = "RER(t): %{customdata[1]:.6g}"
    idx_line = "Real Index(t): %{customdata[2]:.2f}"

    if mode == "Nominal":
        hover = "<br>".join([date_line, s_line]) + "<extra></extra>"
    elif mode == "Real":
        hover = "<br>".join([date_line, rer_line, idx_line]) + "<extra></extra>"
    else:
        hover = "<br>".join([date_line, s_line, rer_line, idx_line]) + "<extra></extra>"

    both = mode == "Both"
    fig = make_subplots(specs=[[{"secondary_y": both}]])

    if mode in ("Nominal", "Both"):
        fig.add_trace(
            go.Scatter(
                x=sample.index, y=sample["S"], mode="lines", name=f"Nominal S(t)",
                line=dict(color="#1f77b4", width=2), customdata=custom,
                hovertemplate=hover, connectgaps=False,
            ),
            secondary_y=False if both else None,
        )
    if mode in ("Real", "Both"):
        fig.add_trace(
            go.Scatter(
                x=sample.index, y=sample["Real_Index"], mode="lines", name="Real Index(t)",
                line=dict(color="#d62728", width=2), customdata=custom,
                hovertemplate=hover, connectgaps=False,
            ),
            secondary_y=True if both else None,
        )

    if mode == "Nominal":
        fig.update_yaxes(title_text=s_label)
    elif mode == "Real":
        fig.update_yaxes(title_text="Real Index (100 = sample-average RER)")
        fig.add_hline(y=100, line_dash="dot", line_color="gray")
    else:
        fig.update_yaxes(title_text=s_label, secondary_y=False, title_font_color="#1f77b4")
        fig.update_yaxes(
            title_text="Real Index (100 = sample-average RER)", secondary_y=True,
            title_font_color="#d62728", showgrid=False,
        )

    fig.update_layout(
        title=f"{n1} vs {n2}",
        hovermode="closest",
        hoverlabel=dict(namelength=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=70, b=10),
        height=560,
    )
    fig.update_xaxes(title_text="Month", rangeslider=dict(visible=True))
    return fig


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="IMF Exchange Rate Explorer", layout="wide")
    st.title("IMF Exchange Rate Explorer")
    st.caption(
        "Nominal and real bilateral exchange rates from monthly IMF CPI and "
        "exchange-rate data (from January 1976, where available)."
    )

    countries, live = load_country_list()
    if not live:
        st.info(
            "Could not load the IMF country list live; showing a built-in list of major "
            "economies instead."
        )
    codes = list(countries)

    def fmt(code: str) -> str:
        return f"{countries[code]} ({code})"

    with st.sidebar:
        st.header("Inputs")
        iso1 = st.selectbox(
            "Country 1 (base currency)", codes,
            index=codes.index("USA") if "USA" in codes else 0, format_func=fmt,
        )
        iso2 = st.selectbox(
            "Country 2 (quote currency)", codes,
            index=codes.index("GBR") if "GBR" in codes else 1, format_func=fmt,
        )

    if iso1 == iso2:
        st.warning("Please choose two different countries.")
        return

    with st.spinner("Fetching IMF data..."):
        try:
            c1, c2 = load_country(iso1), load_country(iso2)
        except IMFError as exc:
            st.error(str(exc))
            return

    df = build_frame(c1, c2)
    if df.empty:
        st.error("The IMF returned no CPI or exchange-rate data for either country.")
        return

    months = list(df.index)
    with st.sidebar:
        start = st.selectbox("Start month", months, index=0, format_func=lambda t: t.strftime("%Y-%m"))
        end = st.selectbox(
            "End month", months, index=len(months) - 1, format_func=lambda t: t.strftime("%Y-%m")
        )
        mode = st.radio("Exchange rate", ["Real", "Nominal", "Both"], index=2)

    if start > end:
        st.error("Start month must be on or before end month.")
        return

    sample, rer_star, n_valid = compute(df, start, end)

    n1, n2 = countries[iso1], countries[iso2]
    missing = [
        label
        for label, s in (
            (f"CPI for {n1}", c1.cpi),
            (f"CPI for {n2}", c2.cpi),
            (f"exchange rate for {n1}", c1.fx if iso1 != "USA" else pd.Series([1.0])),
            (f"exchange rate for {n2}", c2.fx if iso2 != "USA" else pd.Series([1.0])),
        )
        if s.empty
    ]
    if missing:
        st.warning("No IMF data found for: " + "; ".join(missing) + ".")
    for c in (c1, c2):
        for note in c.notes or []:
            st.info(note)

    if mode in ("Real", "Both") and n_valid == 0:
        st.error(
            "No month in the selected sample has all of CPI and exchange rates for both "
            "countries, so the real exchange rate cannot be computed. Try another sample or pair."
        )
        if mode == "Real":
            return
    if mode in ("Nominal", "Both") and sample["S"].notna().sum() == 0:
        st.error("No nominal exchange-rate observations in the selected sample.")
        if mode == "Nominal":
            return

    cols = st.columns(4)
    cols[0].metric("Sample", f"{start:%Y-%m} → {end:%Y-%m}")
    cols[1].metric("Months in sample", len(sample))
    cols[2].metric("Valid RER months", n_valid)
    cols[3].metric("RER* (sample mean)", f"{rer_star:.6g}" if n_valid else "n/a")

    st.plotly_chart(make_figure(sample, mode, n1, n2, iso1, iso2), use_container_width=True)

    st.download_button(
        "Download CSV (all observations in sample)",
        data=make_csv(sample, iso1, iso2),
        file_name=f"imf_fx_{iso1}_{iso2}_{start:%Y%m}_{end:%Y%m}.csv",
        mime="text/csv",
    )

    with st.expander("Data sources used"):
        for c, name in ((c1, n1), (c2, n2)):
            def span(x: pd.Series) -> str:
                return f"{x.index.min():%Y-%m} to {x.index.max():%Y-%m}, {len(x)} obs" if not x.empty else "none"
            st.markdown(
                f"**{name}** — CPI: `{c.cpi_source}` ({span(c.cpi)}); "
                f"FX: `{c.fx_source}` ({span(c.fx)})"
            )

    with st.expander("Definitions"):
        st.markdown(
            f"- **S(t)** = units of **{iso2}** currency per 1 unit of **{iso1}** currency "
            f"= E₂(t) / E₁(t), where Eᵢ is local currency per USD (IMF `XDC_USD`, period average).\n"
            f"- **RER(t)** = S(t) × P₁(t) / P₂(t), with P the monthly CPI index.\n"
            f"- **RER\\*** = average RER(t) over valid months in the selected sample; "
            f"**Real Index(t)** = 100 × RER(t) / RER\\*.\n"
            f"- A month is *valid* when S, P₁ and P₂ are all available and positive."
        )


if __name__ == "__main__":
    main()
