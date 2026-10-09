# African Economic Stress & Early-Warning System: Steps, Tools and Code

**Status:** pipeline written and code-tested; **real-data run still to be done on your machine** (see section 1). No result in this document comes from real data.

## 1. What was and was not done

| Item | Status |
|---|---|
| Kenya GDP growth 1961-2025 via the World Bank API (one series, retrieved live) | Done, confirms the API is reachable and 2025 data exist (WDI last updated 2026-07-13) |
| Full panel: 10 indicators x 8 countries | **Not downloaded here**: my sandbox had no internet and my fetch tool refused any URL not already in the chat |
| Pipeline code `early_warning_pipeline.py` | Written |
| Code test | Ran end to end on a **synthetic random fixture** (creates all CSV/Excel/chart/JSON outputs). ARIMA, ETS and VAR could **not** be tested because `statsmodels` was not installable in the sandbox; those three functions are untested |
| Final results report | Pending your run (section 6) |

## 2. Tools

Python 3.10+; `pandas`, `numpy` (data), `requests` (API), `scikit-learn` (PCA, logistic regression, Ridge, metrics), `statsmodels` (ARIMA, Holt-Winters ETS, VAR), `matplotlib` (charts), `openpyxl` (Excel). Install:
`pip install pandas numpy scikit-learn statsmodels matplotlib openpyxl requests`

## 3. How to run

1. Download the Pink Sheet **annual** Excel from the World Bank Commodity Markets page (file named like `CMO-Historical-Data-Annual.xlsx`).
2. `python early_warning_pipeline.py --pinksheet CMO-Historical-Data-Annual.xlsx --outdir output`
3. Check the console for the data-quality line and any `WARNING`. Then send me `output/results.json` (and ideally `economic_forecasting.xlsx`) and I will write the final report from it.

If the Pink Sheet parser fails, the sheet layout changed; the function `load_pinksheet` finds the row containing "Crude oil, average" and reads the year column, so a small edit will fix it.

## 4. Pipeline steps and design decisions

**Step 1, acquisition (`fetch_wdi`).** One API call per indicator for all 8 countries (semicolon-separated ISO3 codes), years 1990-2025, JSON cached under `output/raw/` so reruns are offline. Indicators: GDP growth `NY.GDP.MKTP.KD.ZG`, inflation `FP.CPI.TOTL.ZG`, unemployment `SL.UEM.TOTL.ZS`, exchange rate `PA.NUS.FCRF`, central government debt `GC.DOD.TOTL.GD.ZS`, exports `NE.EXP.GNFS.ZS`, imports `NE.IMP.GNFS.ZS`, investment `NE.GDI.TOTL.ZS`, GDP per capita `NY.GDP.PCAP.KD`, current account `BN.CAB.XOKA.GD.ZS`.

**Step 2, commodities (`load_pinksheet`).** Annual nominal crude-oil average price (and Brent if present); oil growth = year-on-year % change on the global series.

**Step 3, validation and cleaning (`build_panel`).**
- Derived `fx_depr` = % change in LCU per USD (positive = currency weakens).
- Coverage table per country and indicator is saved to the workbook.
- Short interior gaps (up to 2 years) interpolated; remaining gaps filled with the country median, then the pooled median, and every filled cell is flagged (`*_imputed` columns, and shares reported in `results.json`).
- Indicators with pooled coverage below 60% are **dropped from the index automatically** and listed in `index_dropped_low_coverage`. Debt and current-account series are the likeliest candidates; this is a real data limitation, not a bug.

**Step 4, Economic Stress Index (`StressModel`).** Winsorise at the 1st/99th percentile, standardise, flip signs so higher always means more stress (growth and current account are negated), run PCA, take PC1 oriented so its loadings sum positive, and scale to 0-100 by min-max over the pooled panel. Bands: Low <25, Moderate 25-50, High 50-75, Severe >=75. Loadings and explained variance are exported. The stored scaler and loadings are reused to re-score scenarios.

**Step 5, early-warning model.** Target for year *t*: `slump_next` = 1 if growth in *t+1* is below 2% **or** falls more than 3 pp from *t*. Both thresholds are constants at the top of the script and should be varied as a robustness check. Two specifications: (A) stress index only, (B) the individual components plus oil growth, each in a standardised L2 logistic regression. Validation is expanding window (train on years up to T-1, test on T, T = 2008-2024), reporting AUC, Brier score and the Brier score of a prevalence-only baseline. A leave-one-country-out AUC is also reported. This replaces a "recession" label: annual recessions are too rare in these countries for a reliable classifier, as the source notes in your project brief also say.

**Step 6, forecasting (Kenya GDP growth).** Rolling-origin one-step forecasts for 2005-2025 from: naive, expanding mean, ARIMA (grid p 0-2, d 0-1, q 0-2 chosen by AIC), damped-trend ETS, VAR(1) on growth/inflation/oil growth, and a Ridge regression on lags as the machine-learning comparison. Metrics: MAE, RMSE, MAPE (only where |actual| > 0.5 because growth near zero blows MAPE up) and bias. The best RMSE model among those supporting multi-step produces the 2026-2027 forecast. With about 20 test points, differences between models will often be statistically small; the report should say so.

**Step 7, scenario engine (`scenarios`).** Country-fixed-effects OLS on the panel: a reduced form (growth on oil growth and peer growth) and a structural form (adding inflation and depreciation), plus pass-through regressions of inflation and depreciation on oil growth. Shocks are applied to Kenya's latest observation: oil +30%, inflation +3 pp, depreciation +15%, peer growth -2 pp. Each scenario outputs the GDP change, the re-scored stress index and band, and the slump probability. These are **associations from a small annual panel**, not causal estimates; the report must state this.

**Step 8, outputs.** `africa_macro_panel_with_stress.csv`, `economic_forecasting.xlsx` (assumptions-to-results sheets), six PNG charts, and `results.json`.

## 5. Known limitations to disclose in the report

- Annual data, about 26 years x 8 countries, so modest power and overlapping shocks (2008-09, 2020).
- Imputation and the choice of slump thresholds affect results; run sensitivity checks.
- Pooled panel assumes common coefficients across countries; fixed effects only absorb level differences.
- ARIMA/ETS/VAR code paths are untested in my sandbox; if one throws, the script logs it and carries on with the others.
- The SQL database from the original plan was not built. The results are delivered as CSV and Excel files and as an interactive web dashboard (`kenya_stress_dashboard.html`), which draws its charts from the same corrected panel.

## 6. Next step for the final report

Once `results.json` exists, the final report will cover: question and motivation; data and quality; index construction with loadings; early-warning performance versus baseline; Kenya's current risk and peer ranking; forecast accuracy table and 2026-27 outlook; scenario table; limitations; recommendations for policy and for the petroleum-sector angle if you want that tie-in.

## 7. Code

Full source: `early_warning_pipeline.py` (single file, about 500 lines, sections numbered to match this document).


## 8. Addendum: partial real-data run (8 Oct 2026)

You uploaded four cached World Bank files (GDP growth, inflation, unemployment, exchange rate; 8 countries, 1990-2025). I ran the script offline on them. Not available: debt, current account, exports, imports, investment, GDP per capita, oil prices, and `statsmodels` models.

**Script changes made after the first partial run (disclosed because they were made after seeing results):**
1. A download failure for one indicator no longer stops the run; the column is left empty and logged. Your cache stopped after the exchange rate, so the debt download most likely failed.
2. Unemployment is now used as its **yearly change** (`unemp_chg`) instead of its level. In the first run the level gave South Africa (structurally ~30% unemployment) the lowest stress score, which is a measurement problem, not an economic signal. This was done on logic about comparability, but it was prompted by that result, so treat it as a specification choice and test alternatives on the full data.
3. Oil variables, the oil scenario and the VAR are skipped automatically when no oil data are supplied.

**Partial results (4 indicators, no oil, no ARIMA/ETS/VAR): not final.**
- Stress index: PC1 explains 38% of variance; inflation (0.71) and depreciation (0.70) dominate, GDP growth and unemployment change barely load.
- Early-warning model: out-of-sample AUC 0.45-0.48 and Brier scores slightly worse than predicting the average slump rate (18%). On these four indicators the model has **no demonstrated predictive skill**.
- Forecasting (Kenya, 21 rolling one-step forecasts): expanding mean RMSE 2.55, Ridge 2.80, naive 3.03. Differences are small relative to noise.
- Scenarios: a 15% depreciation raises Kenya's index from 12.7 to 25.0 and the slump probability from 20% to 25%; the inflation and peer-slowdown scenarios produce effects that are small or in a counter-intuitive direction, consistent with a weak model.


## 9. Addendum: full run (8 Oct 2026)

You ran the pipeline in Colab with the Pink Sheet file. All models ran, including ARIMA, ETS and VAR. Debt was dropped automatically (12% coverage). The run used the earlier version of the script (unemployment in levels). I reproduced its numbers exactly from `economic_forecasting.xlsx`, then ran a robustness variant (unemployment change) on the same panel. Both are reported in the final report. The current script uses unemployment change by default, so a fresh run will give the robustness numbers; to reproduce the delivered report exactly, change `"unemp_chg": +1` back to `"unemployment": +1` in `STRESS_SIGN` and remove the `unemp_chg` line in `build_panel`.


## 10. Addendum: data correction found while expanding the final report (9 Oct 2026)

- **Bug:** `pct_change()` carried the last exchange rate forward over a missing value, so Ethiopia's and Tanzania's missing 2025 rates were recorded as 0% depreciation (and reported as 0% imputed). **Fix:** `pct_change(fill_method=None)`; the gaps are now filled with the country median and flagged.
- **Effect:** Ethiopia's 2025 index 36.9 to 41.4, Tanzania 25.8 to 28.8; Kenya 21.2 to 21.4; other 2025 values move by up to 0.5 points through the PCA refit. AUC (0.548 and 0.554), conclusions and the forecasting results are unchanged (forecasting uses only Kenya's series).
- **Method used to recompute:** the same pipeline functions were run on the panel in `economic_forecasting.xlsx`, with the two exchange-rate values reset to missing and refilled. The final report uses these corrected figures.
- **Reproducing the report exactly:** the script's default is now unemployment change. To reproduce the primary figures, set `STRESS_SIGN` to use `"unemployment": +1` instead of `"unemp_chg": +1`.
