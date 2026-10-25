IMF Exchange Rate Explorer

A Streamlit app that compares nominal and real bilateral exchange rates for any two IMF countries (or the Euro area, G163) using monthly IMF CPI and exchange-rate data from January 1976, where available.

Quick start (Python 3.12)
bash
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

The app opens at http://localhost:8501.

Features
Two drop-down menus to pick Country 1 and Country 2 (list loaded from the IMF country codelist; a built-in list of major economies is used if that call fails).
Start and end month selectors for the sample period.
Choose Real, Nominal, or Both (Both uses separate left/right y-axes).
Interactive Plotly chart with a range slider. Hover shows:
Real: RER(t) and Real Index(t)
Nominal: S(t)
Both: S(t), RER(t), and Real Index(t)
CSV download of every month in the selected sample with all inputs and calculations.
Data and method

IMF series (SDMX 3.0, base https://api.imf.org/external/sdmx/3.0):

Series	Dataflow	Key
Monthly CPI index	IMF.STA/CPI	{ISO3}.CPI._T.IX.M
Exchange rate, period average, local currency per USD	IMF.STA/ER	{ISO3}.XDC_USD.PA_RT.M

Let E1(t) and E2(t) be local currency per USD for Country 1 and Country 2, and P1(t), P2(t) their CPIs.

Nominal: S(t) = E2(t) / E1(t) = units of Country 2 currency per 1 unit of Country 1 currency. Plotted in actual units, not normalized.
Real: RER(t) = S(t) × P1(t) / P2(t).
Benchmark: RER* = average of RER(t) over valid months within the selected start/end months. Changing the sample changes RER*.
Real Index: 100 × RER(t) / RER*.

A month is valid for RER when E1, E2, P1, and P2 are all present and positive. Months missing any input appear as gaps in the real series (they are not interpolated). The nominal series is plotted wherever E1 and E2 both exist, even if CPI is missing. The CSV includes every month in the sample, with blanks where data is missing and a valid_rer_month flag.

Notes and limitations
Euro area (G163): whichever euro-area code you pick, the app tries G163, EA, U2, then EMU for CPI and for the exchange rate. If the IMF has no euro-area exchange-rate series, it falls back to Germany's EUR-per-USD rate from January 1999 (identical for all euro members) and tells you so. Pre-1999 euro-area real rates are then unavailable. The "Data sources used" panel shows exactly which keys returned data.
IMF coverage varies by country and series; the app warns when a series is missing. Some countries start well after 1976, and some series (e.g. Euro area) begin later.
The United States has no USD/USD series in some releases; the app treats it as 1.0 when absent.
Results are cached for 6 hours per series. Restart the app or clear the Streamlit cache to force a refresh.
The IMF API is public and may rate-limit or change dataflow versions; the app retries and reports HTTP errors in the UI. If the IMF changes its response layout, the parsing lives in parse_sdmx_json() in app.py.
The app was tested offline with synthetic SDMX-JSON payloads (series and flat layouts) and the calculations were verified (mean Real Index over valid months = 100). It was not run against the live IMF API in the environment where it was written, so please check one known pair (e.g. USA/GBR) on first run.
