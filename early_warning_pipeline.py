#!/usr/bin/env python3
"""
African Economic Stress & Early-Warning System  -  end-to-end pipeline
=====================================================================
Kenya (primary) + Uganda, Tanzania, Rwanda, Ethiopia, Ghana, Nigeria, South Africa.

Stages
  1. Download World Development Indicators (WDI) via the World Bank API (cached)
  2. Load World Bank Pink Sheet annual commodity prices (Excel you download once)
  3. Validate, clean, impute (with flags), engineer features
  4. Economic Stress Index (z-scores + PCA, scaled 0-100, four stress bands)
  5. Early-warning logistic model (next-year growth slump), expanding-window validation
  6. Kenya GDP growth forecasting: naive, mean, ARIMA, ETS, VAR, Ridge; rolling-origin accuracy
  7. Scenario engine: oil, inflation, exchange-rate, global-slowdown shocks
  8. Save CSVs, Excel workbook, PNG charts and results.json (inputs for the report)

Usage
  pip install pandas numpy scikit-learn statsmodels matplotlib openpyxl requests
  python early_warning_pipeline.py --pinksheet CMO-Historical-Data-Annual.xlsx --outdir output
  python early_warning_pipeline.py --synthetic --outdir test_output   # code test only, NOT real data
"""
import argparse, json, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
COUNTRIES = {"KEN": "Kenya", "UGA": "Uganda", "TZA": "Tanzania", "RWA": "Rwanda",
             "ETH": "Ethiopia", "GHA": "Ghana", "NGA": "Nigeria", "ZAF": "South Africa"}
CORE = "Kenya"
INDICATORS = {
    "gdp_growth": "NY.GDP.MKTP.KD.ZG", "inflation": "FP.CPI.TOTL.ZG",
    "unemployment": "SL.UEM.TOTL.ZS", "fx_rate": "PA.NUS.FCRF",
    "debt_gdp": "GC.DOD.TOTL.GD.ZS", "exports_gdp": "NE.EXP.GNFS.ZS",
    "imports_gdp": "NE.IMP.GNFS.ZS", "investment_gdp": "NE.GDI.TOTL.ZS",
    "gdp_pc": "NY.GDP.PCAP.KD", "ca_gdp": "BN.CAB.XOKA.GD.ZS",
}
FETCH_FROM, FETCH_TO = 1990, 2025        # forecasting uses the long history
ANALYSIS_FROM = 2000                     # stress index / early-warning panel
# sign: +1 means a HIGHER value is MORE stressful
STRESS_SIGN = {"gdp_growth": -1, "inflation": +1, "unemp_chg": +1,
               "fx_depr": +1, "debt_gdp": +1, "ca_gdp": -1}
MIN_COVERAGE = 0.60                      # drop an indicator from the index below this
SLUMP_LEVEL, SLUMP_DROP = 2.0, 3.0       # slump = next-year growth <2% OR fall >3pp
BANDS = [("Low", 0, 25), ("Moderate", 25, 50), ("High", 50, 75), ("Severe", 75, 101)]
EW_TEST_YEARS = range(2008, 2025)
FC_ORIGINS = range(2004, 2025)           # last training year of each rolling forecast


def log(msg):
    print(f"[pipeline] {msg}", flush=True)


# ----------------------------------------------------------------------------
# 1. DATA ACQUISITION
# ----------------------------------------------------------------------------
def fetch_wdi(outdir: Path, offline: bool = False, cache_dir: Path = None) -> pd.DataFrame:
    import requests
    cache = cache_dir or (outdir / "raw")
    cache.mkdir(parents=True, exist_ok=True)
    codes = ";".join(COUNTRIES)
    frames = []
    for name, code in INDICATORS.items():
        f = next((x for x in (cache / f"wdi_{code}.json", cache / f"wdi_{code.replace('.', '_')}.json") if x.exists()), None)
        if f is None and offline:
            log(f"MISSING (offline) {code} -> {name}: column left empty")
            frames.append(pd.DataFrame({name: np.nan}, index=pd.MultiIndex.from_product([list(COUNTRIES), range(FETCH_FROM, FETCH_TO + 1)], names=["iso3", "year"])))
            continue
        f = f or (cache / f"wdi_{code}.json")
        if f.exists():
            rows = json.loads(f.read_text())
        else:
            rows, page = [], 1
            try:
              while True:
                url = f"https://api.worldbank.org/v2/country/{codes}/indicator/{code}"
                r = requests.get(url, params={"format": "json", "per_page": 20000, "page": page,
                                              "date": f"{FETCH_FROM}:{FETCH_TO}"}, timeout=60)
                r.raise_for_status()
                payload = r.json()
                if len(payload) < 2 or payload[1] is None:
                    raise RuntimeError(f"No data returned: {payload}")
                rows += payload[1]
                if page >= payload[0]["pages"]:
                    break
                page += 1
              f.write_text(json.dumps(rows))
            except Exception as e:
                log(f"DOWNLOAD FAILED for {code} ({name}): {e} -> column left empty, continuing")
                frames.append(pd.DataFrame({name: np.nan}, index=pd.MultiIndex.from_product([list(COUNTRIES), range(FETCH_FROM, FETCH_TO + 1)], names=["iso3", "year"])))
                continue
        df = pd.DataFrame([{"iso3": x["countryiso3code"], "year": int(x["date"]),
                            name: x["value"]} for x in rows])
        frames.append(df.set_index(["iso3", "year"]))
        log(f"WDI {code:<22} -> {name:<14} {df[name].notna().sum():>4} non-null values")
    panel = pd.concat(frames, axis=1).reset_index()
    panel["country"] = panel["iso3"].map(COUNTRIES)
    return panel.dropna(subset=["country"])


def load_pinksheet(path: Path) -> pd.DataFrame:
    """Parse annual crude-oil prices from the World Bank Pink Sheet workbook."""
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    key = [s for s in sheets if "annual" in s.lower() and "nominal" in s.lower()]
    if not key:
        raise ValueError(f"No 'Annual Prices (Nominal)' sheet in {path}; sheets: {list(sheets)}")
    raw = sheets[key[0]]
    hdr = next(i for i in range(len(raw)) if raw.iloc[i].astype(str).str.contains(
        "Crude oil, average", case=False).any())
    names = raw.iloc[hdr].astype(str)
    col_avg = names[names.str.contains("Crude oil, average", case=False)].index[0]
    brent = names[names.str.contains("Crude oil, Brent", case=False)]
    body = raw.iloc[hdr + 1:].copy()
    body["year"] = pd.to_numeric(body.iloc[:, 0], errors="coerce")
    body = body[body["year"].between(1960, 2100)]
    out = pd.DataFrame({"year": body["year"].astype(int),
                        "oil_avg": pd.to_numeric(body[col_avg], errors="coerce")})
    if len(brent):
        out["oil_brent"] = pd.to_numeric(body[brent.index[0]], errors="coerce")
    return out.reset_index(drop=True)


def synthetic_data():
    """Random data to test the code path. NEVER use for results."""
    rng = np.random.default_rng(0)
    yrs = range(FETCH_FROM, FETCH_TO + 1)
    oil = pd.DataFrame({"year": list(yrs), "oil_avg": 40 * np.exp(np.cumsum(rng.normal(0, .2, len(yrs))))})
    rows = []
    for iso, c in COUNTRIES.items():
        g = 4 + np.zeros(len(yrs))
        for t in range(1, len(yrs)):
            g[t] = 0.4 * g[t - 1] + 2.4 + rng.normal(0, 2.2)
        for i, y in enumerate(yrs):
            rows.append(dict(iso3=iso, country=c, year=y, gdp_growth=g[i],
                             inflation=abs(rng.normal(8, 5)), unemployment=rng.uniform(3, 12),
                             fx_rate=50 * (1.04 ** i) * rng.uniform(.95, 1.05),
                             debt_gdp=rng.uniform(30, 80), exports_gdp=rng.uniform(10, 30),
                             imports_gdp=rng.uniform(15, 40), investment_gdp=rng.uniform(15, 30),
                             gdp_pc=1000 * 1.02 ** i, ca_gdp=rng.normal(-5, 3)))
    return pd.DataFrame(rows), oil


# ----------------------------------------------------------------------------
# 2-3. VALIDATION, CLEANING, FEATURES
# ----------------------------------------------------------------------------
def build_panel(wdi: pd.DataFrame, oil: pd.DataFrame):
    p = wdi.merge(oil, on="year", how="left").sort_values(["country", "year"]).reset_index(drop=True)
    g = p.groupby("country")
    p["fx_depr"] = g["fx_rate"].pct_change(fill_method=None) * 100          # + = local currency weakens vs USD
    p["unemp_chg"] = g["unemployment"].diff()                # change in pp: levels differ structurally across countries
    p["oil_growth"] = p["oil_avg"].pct_change() * 100       # years are consecutive per country
    p.loc[p["year"] == p["year"].min(), "oil_growth"] = np.nan
    # oil_growth must be computed on the global series, not per-country rows
    oil = oil.sort_values("year").assign(oil_growth=lambda d: d["oil_avg"].pct_change() * 100)
    p = p.drop(columns=["oil_growth"]).merge(oil[["year", "oil_growth"]], on="year", how="left")
    full = p.copy()
    p = p[p["year"] >= ANALYSIS_FROM].copy()

    cols = list(STRESS_SIGN) + ["oil_growth"]
    cov = p.groupby("country")[cols].apply(lambda d: d.notna().mean()).T
    cov["pooled"] = p[cols].notna().mean()
    # interpolate short interior gaps, then impute with country then pooled median (flagged)
    for c in cols:
        p[c] = p.groupby("country")[c].transform(lambda s: s.interpolate(limit=2, limit_area="inside"))
    flags = {}
    for c in cols:
        miss = p[c].isna()
        flags[c] = miss.copy()
        p[c] = p[c].fillna(p.groupby("country")[c].transform("median")).fillna(p[c].median())
        p[f"{c}_imputed"] = miss.astype(int)
    dq = {"rows": len(p), "countries": p["country"].nunique(),
          "years": [int(p.year.min()), int(p.year.max())],
          "imputed_share": {c: round(float(flags[c].mean()), 3) for c in cols},
          "duplicate_keys": int(p.duplicated(["country", "year"]).sum())}
    return full, p, cov, dq


# ----------------------------------------------------------------------------
# 4. STRESS INDEX
# ----------------------------------------------------------------------------
class StressModel:
    def fit(self, p, cov):
        self.cols = [c for c in STRESS_SIGN if cov.loc[c, "pooled"] >= MIN_COVERAGE]
        self.dropped = [c for c in STRESS_SIGN if c not in self.cols]
        self.use_oil = bool(cov.loc["oil_growth", "pooled"] >= MIN_COVERAGE)
        self.bounds = {c: tuple(np.percentile(p[c], [1, 99])) for c in self.cols + (["oil_growth"] if self.use_oil else [])}
        W = self.winsor(p)
        self.mu, self.sd = W[self.cols].mean(), W[self.cols].std()
        Z = self.z(W)
        pca = PCA().fit(Z)
        load = pca.components_[0]
        self.load = load if load.sum() >= 0 else -load           # orient: high = stress
        self.explained = pca.explained_variance_ratio_
        raw = Z.values @ self.load
        self.rmin, self.rmax = raw.min(), raw.max()
        return self

    def winsor(self, d):
        d = d.copy()
        for c, (lo, hi) in self.bounds.items():
            if c in d:
                d[c] = d[c].clip(lo, hi)
        return d

    def z(self, W):
        sign = pd.Series({c: STRESS_SIGN[c] for c in self.cols})
        return (W[self.cols] - self.mu) / self.sd * sign

    def index(self, d):
        raw = self.z(self.winsor(d)).values @ self.load
        return np.clip(100 * (raw - self.rmin) / (self.rmax - self.rmin), 0, 100)


def band(x):
    return next(n for n, lo, hi in BANDS if lo <= x < hi)


# ----------------------------------------------------------------------------
# 5. EARLY-WARNING MODEL
# ----------------------------------------------------------------------------
def add_target(p):
    p = p.sort_values(["country", "year"]).copy()
    nxt = p.groupby("country")["gdp_growth"].shift(-1)
    p["slump_next"] = np.where(nxt.isna(), np.nan, ((nxt < SLUMP_LEVEL) | (p["gdp_growth"] - nxt > SLUMP_DROP)).astype(float))
    return p


def make_logit():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))


def early_warning(p, sm):
    featB = sm.cols + (["oil_growth"] if sm.use_oil else [])
    p = p.copy()
    p["stress_index"] = sm.index(p)
    W = sm.winsor(p)
    for c in featB:
        p[f"w_{c}"] = W[c]
    specs = {"A_stress_index_only": ["stress_index"], "B_components_plus_oil": [f"w_{c}" for c in featB]}
    lab = p.dropna(subset=["slump_next"])
    oof = {k: [] for k in specs}
    for T in EW_TEST_YEARS:
        tr, te = lab[lab.year <= T - 1], lab[lab.year == T]
        if tr["slump_next"].nunique() < 2 or te.empty:
            continue
        for k, cols in specs.items():
            m = make_logit().fit(tr[cols], tr["slump_next"])
            oof[k].append(te[["country", "year", "slump_next"]].assign(prob=m.predict_proba(te[cols])[:, 1]))
    cv, preds = [], {}
    prev = float(lab["slump_next"].mean())
    for k, parts in oof.items():
        d = pd.concat(parts)
        preds[k] = d
        cv.append({"spec": k, "n_test": len(d), "AUC": roc_auc_score(d.slump_next, d.prob),
                   "Brier": brier_score_loss(d.slump_next, d.prob),
                   "Brier_baseline_prevalence": brier_score_loss(d.slump_next, np.full(len(d), d.slump_next.mean()))})
    cv = pd.DataFrame(cv)
    final = {k: make_logit().fit(lab[c], lab["slump_next"]) for k, c in specs.items()}
    coefs = pd.DataFrame({"feature": specs["B_components_plus_oil"],
                          "std_coef": final["B_components_plus_oil"][-1].coef_[0]})
    coefs["odds_ratio_per_1sd"] = np.exp(coefs["std_coef"])
    p["slump_prob_next"] = final["B_components_plus_oil"].predict_proba(p[specs["B_components_plus_oil"]])[:, 1]
    # leave-one-country-out as a robustness check
    loco = []
    for c in lab["country"].unique():
        tr, te = lab[lab.country != c], lab[lab.country == c]
        m = make_logit().fit(tr[specs["B_components_plus_oil"]], tr["slump_next"])
        pr = m.predict_proba(te[specs["B_components_plus_oil"]])[:, 1]
        loco.append({"held_out": c, "AUC": roc_auc_score(te.slump_next, pr) if te.slump_next.nunique() > 1 else np.nan})
    return p, cv, coefs, pd.DataFrame(loco), final, specs, preds, prev


# ----------------------------------------------------------------------------
# 6. FORECASTING (Kenya GDP growth)
# ----------------------------------------------------------------------------
def _arima(y, h):
    from statsmodels.tsa.arima.model import ARIMA
    best = None
    for pq in [(p, d, q) for p in range(3) for d in range(2) for q in range(3)]:
        try:
            r = ARIMA(y, order=pq, trend="c" if pq[1] == 0 else "n").fit()
            if best is None or r.aic < best.aic:
                best = r
        except Exception:
            continue
    return np.asarray(best.forecast(h))


def _ets(y, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    r = ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="estimated").fit()
    return np.asarray(r.forecast(h))


def _var(df, h):
    from statsmodels.tsa.api import VAR
    if df["oil_growth"].isna().all():
        raise ValueError("no oil data for VAR")
    r = VAR(df.values).fit(1)
    return r.forecast(df.values[-1:], h)[:, 0]


def _ridge(df, h):
    d = df.copy()
    use_oil = d["oil_growth"].notna().any()
    X = pd.DataFrame({"g1": d["gdp_growth"].shift(1), "g2": d["gdp_growth"].shift(2),
                      "infl1": d["inflation"].shift(1)})
    last = {"g1": d["gdp_growth"].iloc[-1], "g2": d["gdp_growth"].iloc[-2], "infl1": d["inflation"].iloc[-1]}
    if use_oil:
        X["oil1"] = d["oil_growth"].shift(1)
        last["oil1"] = d["oil_growth"].iloc[-1]
    feats = list(last)
    X["y"] = d["gdp_growth"]
    X = X.dropna()
    m = Ridge(alpha=5.0).fit(X[feats], X["y"])
    return np.array([m.predict(pd.DataFrame([last]))[0]])


def forecasting(full):
    k = full[full.country == CORE].set_index("year").sort_index()
    k = k[["gdp_growth", "inflation", "oil_growth"]]
    k["inflation"] = k["inflation"].interpolate(limit_area="inside")
    k = k.dropna(subset=["gdp_growth"])
    y = k["gdp_growth"]
    models = {"Naive (last value)": lambda tr, h: np.repeat(tr["gdp_growth"].iloc[-1], h),
              "Mean (expanding)": lambda tr, h: np.repeat(tr["gdp_growth"].mean(), h),
              "ARIMA (AIC grid)": lambda tr, h: _arima(tr["gdp_growth"].values, h),
              "ETS (damped trend)": lambda tr, h: _ets(tr["gdp_growth"].values, h),
              "VAR(1) gdp-infl-oil": lambda tr, h: _var(tr.dropna(), h),
              "Ridge (ML benchmark)": lambda tr, h: _ridge(tr, h)}
    recs, unavailable = [], set()
    for T in FC_ORIGINS:
        if T + 1 not in y.index:
            continue
        tr = k.loc[:T]
        for name, fn in models.items():
            if name in unavailable:
                continue
            try:
                recs.append({"model": name, "origin": T, "year": T + 1, "actual": y[T + 1], "forecast": float(fn(tr, 1)[0])})
            except ImportError:
                unavailable.add(name)
                print(f"WARNING: statsmodels missing, skipped {name}", file=sys.stderr)
            except Exception as e:
                log(f"{name} failed at origin {T}: {e}")
    d = pd.DataFrame(recs)
    d["err"] = d["forecast"] - d["actual"]
    m = d.groupby("model").apply(lambda g: pd.Series({
        "n": len(g), "MAE": g.err.abs().mean(), "RMSE": np.sqrt((g.err ** 2).mean()),
        "MAPE_%(|actual|>0.5)": (g.err.abs() / g.actual.abs())[g.actual.abs() > 0.5].mean() * 100,
        "bias": g.err.mean()})).sort_values("RMSE")
    # 2-year-ahead forecast from the best model that supports multi-step
    multi = [n for n in m.index if n != "Ridge (ML benchmark)"]
    best = multi[0]
    fc = models[best](k, 2)
    fdf = pd.DataFrame({"year": [y.index.max() + 1, y.index.max() + 2], "gdp_growth_forecast": fc, "model": best})
    return d, m, fdf, y


# ----------------------------------------------------------------------------
# 7. SCENARIO ENGINE
# ----------------------------------------------------------------------------
def fe_ols(df, ycol, xcols):
    d = df[["country", ycol] + xcols].dropna()
    dm = d.groupby("country")[[ycol] + xcols].transform(lambda s: s - s.mean())
    X, y = dm[xcols].values, dm[ycol].values
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    dof = len(d) - len(xcols) - d["country"].nunique()
    s2 = ((y - X @ b) ** 2).sum() / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return pd.DataFrame({"coef": b, "se": se, "t": b / se}, index=xcols), len(d)


def scenarios(p, sm, final, specs):
    q = p.copy()
    q["peer_growth"] = [q[(q.year == r.year) & (q.country != r.country)]["gdp_growth"].mean() for r in q.itertuples()]
    q["inflation_w"], q["fx_depr_w"] = sm.winsor(q)["inflation"], sm.winsor(q)["fx_depr"]
    oil_ok = sm.use_oil
    o = ["oil_growth"] if oil_ok else []
    red, n1 = fe_ols(q, "gdp_growth", o + ["peer_growth"])
    stru, n2 = fe_ols(q, "gdp_growth", o + ["inflation_w", "fx_depr_w", "peer_growth"])
    if oil_ok:
        pt_inf, _ = fe_ols(q, "inflation_w", ["oil_growth"])
        pt_fx, _ = fe_ols(q, "fx_depr_w", ["oil_growth"])
    base = q[(q.country == CORE) & (q.year == q.year.max())].iloc[0]
    featB = [c.replace("w_", "") for c in specs["B_components_plus_oil"]]

    def score(chg):
        row = base.copy()
        for c, v in chg.items():
            row[c] = row[c] + v
        dfr = pd.DataFrame([row])
        w = sm.winsor(dfr)
        X = pd.DataFrame([{f"w_{c}": w[c].iloc[0] for c in featB}])
        return float(sm.index(dfr)[0]), float(final["B_components_plus_oil"].predict_proba(X)[:, 1][0])

    b = lambda m, v: float(m.loc[v, "coef"])
    sc = {
        "Baseline (latest year)": {},
        "Inflation +3 pp": {"inflation": 3, "gdp_growth": b(stru, "inflation_w") * 3},
        "KES depreciation +15%": {"fx_depr": 15, "gdp_growth": b(stru, "fx_depr_w") * 15},
        "Global/peer slowdown -2 pp": {"gdp_growth": b(red, "peer_growth") * -2},
    }
    if oil_ok:
        sc["Oil price +30%"] = {"oil_growth": 30, "gdp_growth": b(red, "oil_growth") * 30,
                                "inflation": b(pt_inf, "oil_growth") * 30, "fx_depr": b(pt_fx, "oil_growth") * 30}
    out = []
    s0, p0 = score({})
    for name, chg in sc.items():
        s, pr = score(chg)
        out.append({"scenario": name, "gdp_growth_change_pp": chg.get("gdp_growth", 0.0),
                    "stress_index": s, "stress_band": band(s), "slump_probability": pr,
                    "d_stress": s - s0, "d_prob_pp": (pr - p0) * 100})
    regs = {"reduced_form": red, "structural": stru, "n": (n1, n2)}
    if oil_ok:
        regs.update(pass_through_inflation=pt_inf, pass_through_fx=pt_fx)
    return pd.DataFrame(out), regs, base


# ----------------------------------------------------------------------------
# 8. OUTPUTS
# ----------------------------------------------------------------------------
def charts(out, p, sm, cv_preds, fm, fpreds, y, fdf, scn):
    c = out / "charts"
    c.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for n, d in p.groupby("country"):
        ax.plot(d.year, d.stress_index, lw=2.8 if n == CORE else 1, alpha=1 if n == CORE else .55, label=n)
    for _, lo, hi in BANDS:
        ax.axhline(lo, color="grey", lw=.4, ls=":")
    ax.set(title="Economic Stress Index (0-100)", xlabel="Year", ylabel="Index"); ax.legend(ncol=4, fontsize=7)
    fig.tight_layout(); fig.savefig(c / "01_stress_index_peers.png", dpi=150); plt.close(fig)

    k = p[p.country == CORE]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(k.year, k.stress_index, color="firebrick", lw=2, label="Stress index")
    ax2 = ax.twinx(); ax2.bar(k.year, k.gdp_growth, alpha=.25, label="GDP growth %")
    ax.set(title=f"{CORE}: stress index vs GDP growth", xlabel="Year"); fig.tight_layout()
    fig.savefig(c / "02_kenya_stress_vs_growth.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    from sklearn.metrics import roc_curve
    for kname, d in cv_preds.items():
        f, t, _ = roc_curve(d.slump_next, d.prob); ax.plot(f, t, label=kname)
    ax.plot([0, 1], [0, 1], "k--", lw=.7); ax.set(title="Out-of-sample ROC (expanding window)", xlabel="FPR", ylabel="TPR"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(c / "03_roc.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    fm["RMSE"].sort_values().plot.barh(ax=ax); ax.set(title="Rolling 1-step forecast RMSE (Kenya GDP growth)", xlabel="RMSE (pp)")
    fig.tight_layout(); fig.savefig(c / "04_forecast_rmse.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(y.index, y.values, "k", label="Actual")
    best = fm.index[0]
    d = fpreds[fpreds.model == best]; ax.plot(d.year, d.forecast, "r--", label=f"{best} (1-step)")
    ax.plot(fdf.year, fdf.gdp_growth_forecast, "ro-", label=f"2-year forecast ({fdf.model.iloc[0]})")
    ax.set(title="Kenya GDP growth: actual vs forecasts (%)"); ax.legend(); fig.tight_layout()
    fig.savefig(c / "05_kenya_forecast.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(scn.scenario, scn.slump_probability * 100); ax.set(title=f"{CORE}: slump probability by scenario (%)")
    fig.tight_layout(); fig.savefig(c / "06_scenarios.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinksheet", type=Path, help="World Bank Pink Sheet annual Excel file")
    ap.add_argument("--outdir", type=Path, default=Path("output"))
    ap.add_argument("--offline", action="store_true", help="use only cached WDI json files, no downloads")
    ap.add_argument("--cache", type=Path, help="folder with wdi_*.json files")
    ap.add_argument("--synthetic", action="store_true", help="TEST ONLY: random data, results meaningless")
    a = ap.parse_args()
    out = a.outdir; out.mkdir(parents=True, exist_ok=True)

    if a.synthetic:
        log("*** SYNTHETIC MODE: results are meaningless ***")
        wdi, oil = synthetic_data()
    else:
        if not a.pinksheet and not a.offline:
            sys.exit("Provide --pinksheet <Pink Sheet annual .xlsx> (download once from the World Bank commodity-markets page).")
        wdi = fetch_wdi(out, a.offline, a.cache)
        oil = load_pinksheet(a.pinksheet) if a.pinksheet else pd.DataFrame(
            {"year": range(1960, FETCH_TO + 1), "oil_avg": np.nan})
        if not a.pinksheet:
            log("NO OIL DATA: oil variables and the oil scenario are skipped")

    full, p, cov, dq = build_panel(wdi, oil)
    log(f"Panel: {dq}")
    sm = StressModel().fit(p, cov)
    p["stress_index"] = sm.index(p)
    p["stress_band"] = p["stress_index"].map(band)
    p = add_target(p)
    p, cv, coefs, loco, final, specs, cv_preds, prev = early_warning(p, sm)
    fpreds, fm, fdf, y = forecasting(full)
    scn, regs, base = scenarios(p, sm, final, specs)

    p.to_csv(out / "africa_macro_panel_with_stress.csv", index=False)
    load = pd.DataFrame({"indicator": sm.cols, "pc1_loading": sm.load})
    comp = pd.DataFrame({"component": range(1, len(sm.explained) + 1), "explained_var": sm.explained})
    latest = p[p.year == p.year.max()][["country", "year", "stress_index", "stress_band", "slump_prob_next"]]
    with pd.ExcelWriter(out / "economic_forecasting.xlsx") as xw:
        p.to_excel(xw, sheet_name="panel_stress", index=False); cov.to_excel(xw, sheet_name="coverage")
        load.to_excel(xw, sheet_name="pca_loadings", index=False); comp.to_excel(xw, sheet_name="pca_variance", index=False)
        cv.to_excel(xw, sheet_name="ew_cv", index=False); coefs.to_excel(xw, sheet_name="ew_coefs", index=False)
        loco.to_excel(xw, sheet_name="ew_leave_country_out", index=False); latest.to_excel(xw, sheet_name="latest_risk", index=False)
        fm.to_excel(xw, sheet_name="forecast_accuracy"); fpreds.to_excel(xw, sheet_name="forecast_rolling", index=False)
        fdf.to_excel(xw, sheet_name="forecast_2yr", index=False); scn.to_excel(xw, sheet_name="scenarios", index=False)
        for k, v in regs.items():
            if isinstance(v, pd.DataFrame):
                v.to_excel(xw, sheet_name=f"reg_{k}"[:31])
    charts(out, p, sm, cv_preds, fm, fpreds, y, fdf, scn)

    res = {"SYNTHETIC_TEST_ONLY": bool(a.synthetic), "data_quality": dq,
           "index_columns": sm.cols, "index_dropped_low_coverage": sm.dropped,
           "pc1_explained_variance": float(sm.explained[0]), "pc1_loadings": dict(zip(sm.cols, map(float, sm.load))),
           "slump_prevalence": prev, "early_warning_cv": cv.to_dict("records"),
           "early_warning_coefs": coefs.to_dict("records"), "leave_country_out": loco.to_dict("records"),
           "latest_risk": latest.to_dict("records"), "forecast_accuracy": fm.reset_index().to_dict("records"),
           "forecast_2yr": fdf.to_dict("records"), "scenarios": scn.to_dict("records"),
           "regressions": {k: v.round(4).to_dict() for k, v in regs.items() if isinstance(v, pd.DataFrame)},
           "kenya_baseline": {c: float(base[c]) for c in sm.cols + (["oil_growth"] if sm.use_oil else []) + ["stress_index"]},
           "oil_used": bool(sm.use_oil)}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    log(f"Done. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
