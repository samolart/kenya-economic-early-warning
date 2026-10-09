# African Economic Stress & Early-Warning System

**A data-driven framework for measuring economic stress, forecasting growth and stress-testing Kenya against external shocks.**

**Live dashboard:** `https://samolart.github.io/kenya-economic-early-warning/`
**[Live Dashboard](https://samolart.github.io/kenya-economic-early-warning/)**

Built with World Bank open data for Kenya and seven African peers (Uganda, Tanzania, Rwanda, Ethiopia, Ghana, Nigeria, South Africa), 2000 to 2025.

## The question

Can annual macroeconomic indicators be combined into an economic stress index, and can that index warn of a growth slump the following year? How exposed is Kenya to oil, inflation, currency and regional-growth shocks?

## Headline findings

| Question | Result |
|---|---|
| Does the index track stress? | Yes, for price and currency stress. Kenya peaks at 51.6 in 2008, 45.1 in 2011 and 42.9 in 2023, and is 21.4 (Low band) in 2025. |
| Does it predict next-year slumps? | **No.** Out-of-sample AUC is 0.55 (0.50 is a coin flip) and the Brier score (0.175) is no better than always guessing the average slump rate (0.172). |
| Best forecaster for Kenya's growth? | Damped-trend exponential smoothing (RMSE 2.29 points), but the gap to a plain average is not statistically significant. Projection: about 5.1% in 2026 and 2027. |
| Which shock matters most? | A 15% shilling depreciation lifts the index from 21.4 to 32.2 (Low to Moderate). The only reliable growth link is to regional growth (coefficient 0.70). No adverse oil effect is detectable. |
| How does Kenya compare? | Second-lowest stress in 2025 and second-lowest average growth (4.4%) of the eight economies. |

The negative result is reported on purpose. Careful out-of-sample validation shows that these indicators do not reliably predict slumps, and the project documents that rather than hiding it.

## What is in this repository

| File | What it is |
|---|---|
| `index.html` | Interactive dashboard (five views: overview, stress monitor, forecast, scenarios, peer comparison). Opens in any browser. |
| `early_warning_pipeline.py` | Full pipeline: download, clean, stress index (PCA), logistic early-warning model, six forecasting models, scenario engine. |
| `steps_and_tools_report.md` | Step-by-step method, tools, decisions and every correction made along the way. |
| `data/` | Clean output tables behind the dashboard (panel, forecast accuracy, rolling forecasts, 2026-27 projection, scenarios). |

## Method in brief

1. **Data:** World Development Indicators through the World Bank API (GDP growth, inflation, unemployment, exchange rate, current account) and the Pink Sheet crude-oil price. Government debt was excluded because only 12% of country-years had values.
2. **Stress index:** indicators winsorised, standardised and signed so higher means more stress, combined with principal component analysis and scaled 0 to 100. Bands: Low below 25, Moderate 25 to 50, High 50 to 75, Severe 75 and above.
3. **Early-warning model:** logistic regression for a slump next year (growth below 2% or a fall of more than 3 points), validated with an expanding window (2008 to 2024) and leave-one-country-out tests.
4. **Forecasting:** naive, expanding mean, ARIMA, damped-trend ETS, VAR and Ridge compared on 21 rolling one-year-ahead forecasts for Kenya.
5. **Scenarios:** country-fixed-effects regressions, then shocks applied to Kenya's 2025 values (oil +30%, inflation +3 points, depreciation +15%, regional growth -2 points).

## Run it yourself

```
pip install pandas numpy scikit-learn statsmodels matplotlib openpyxl requests
python early_warning_pipeline.py --pinksheet pinksheet.xlsx --outdir output
```

Download the annual Pink Sheet Excel file from the World Bank commodity markets page and save it as `pinksheet.xlsx`. The script downloads the World Development Indicators itself. It writes the panel, an Excel workbook, charts and a `results.json` file.

**Reproducibility note:** the published figures use unemployment in levels. The script's current default uses the yearly change in unemployment, a robustness variant described in the steps report. To reproduce the published figures exactly, follow the note at the end of that report.

## Limitations

- Annual data, 26 years and 8 countries: about 36 slump events, so every metric has wide uncertainty.
- The slump label flags chronically slow economies (South Africa) as readily as sudden collapses.
- The index measures price and currency pressure and under-weights output shocks such as 2020.
- Unemployment is an ILO modelled estimate, and some current-account and 2025 exchange-rate values are filled with country medians.
- Scenario results are associations, not causal effects.

## Sources

World Bank, World Development Indicators and Commodity Markets (Pink Sheet). Data retrieved October 2026; the latest year may be revised.

## Author

Samatar Ahmed · samolart25@outlook.com
