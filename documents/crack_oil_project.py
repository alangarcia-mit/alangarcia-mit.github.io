import pandas as pd
import numpy as np
import yfinance as yf
import requests
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import arviz as az
import pymc as pm

"""quick fix for error"""
import pytensor
pytensor.config.cxx = ""
import matplotlib
matplotlib.use("Agg") 

def main():
    START, END = "2010-01-01", "2025-12-31"
    EIA_KEY = os.environ.get("EIA_API_KEY")

    #download futures data
    tickers = {"CL=F": "wti", "RB=F": "rbob", "HO=F": "heating_oil"}
    #print(list(tickers))
    futures = yf.download(list(tickers), start=START, end=END)["Close"]
    futures = futures.rename(columns=tickers)
    #print(futures)

    #eia fundamentals
    EIA_series = {"PET.WCESTUS1.W": "crude_stocks", #US crude stocks ex-SPR, thous bbl
                "PET.WGTSTUS1.W": "gasoline_stocks", #US total gasoline stock
                "PET.WDISTUS1.W": "distillate_stocks",  #US distillate stock
                "PET.WPULEUS3.W": "refinery_util"} #% utlization of operable capacity


    def fetch_eia_data(series_id):
        url = f"https://api.eia.gov/v2/seriesid/{series_id}"
        req = requests.get(url, params={"api_key":EIA_KEY})
        payload = req.json()
        if "response" not in payload:
            raise RuntimeError(f"{series_id} failed: {payload}")
        rows = payload["response"]["data"]
        ser = pd.Series({pd.to_datetime(d["period"]) : float(d["value"]) for d in rows})
        return ser.sort_index()

    eia_data = pd.DataFrame({name : fetch_eia_data(sid) for sid, name in EIA_series.items()})

    #dollar index
    dollar_index = web.DataReader("DTWEXBGS","fred", START, END)
    dollar_index.columns = ["dollar_index"]


    #data alignment
    futures_f = futures.resample("W-FRI").last()
    eia_data_f = eia_data.resample("W-FRI").last()
    dollar_index_f = dollar_index.resample("W-FRI").last()

    df = futures_f.join([eia_data_f, dollar_index_f], how="inner")

    df["crack_321"] = (42*(2*df.rbob + df.heating_oil) - 3*df.wti)/3

    print(df.crack_321.describe())
    df.to_csv("crack_data_weekly.csv")

    df = pd.read_csv("crack_data_weekly.csv", index_col=0, parse_dates=True)
    df["year"] = df.index.year
    df["woy"]  = df.index.isocalendar().week.astype(int).clip(upper=52)
    df = df.sort_index()

    #just the crack history
    df.crack_321.plot(figsize=(12, 4), title="3-2-1 Crack Spread ($/bbl)")
    plt.axhline(df.crack_321.median(), ls="--", c="grey")
    plt.savefig("crack_history.png", dpi=150, bbox_inches="tight")
    plt.close()

    #trailing 5-year seasonal norms
    dev_map = {
            "gasoline_stocks": "gas_dev",
            "distillate_stocks": "dist_dev"}

    for src, new in dev_map.items():
        norm = (df.groupby("woy")[src]
                .transform(lambda s: s.shift(1).rolling(5, min_periods=5).mean()))
        df[new] = df[src] - norm
        df[new + "_norm"] = norm          # keep for sanity checks

    # --- 3. seasonality ---
    df["sin_yr"] = np.sin(2 * np.pi * df.woy / 52)
    df["cos_yr"] = np.cos(2 * np.pi * df.woy / 52)

    #era-colored scatters, better than previous
    def era(y):
        if y <= 2019: return "2015-2019 shale surplus"
        if y <= 2021: return "2020-2021 COVID"
        return "2022-2026 post-invasion"

    df["era"] = df.year.map(era)
    colors = {"2015-2019 shale surplus": "#1f77b4",
            "2020-2021 COVID":         "#d62728",
            "2022-2026 post-invasion": "#2ca02c"}

    plot_vars = [ "gas_dev", "dist_dev", "refinery_util"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, p in zip(axes, plot_vars):
        for name, c in colors.items():
            sub = df[df.era == name]
            ax.scatter(sub[p], sub.crack_321, s=16, alpha=0.65,
                    c=c, label=name, edgecolors="none")
        if p.endswith("_dev"):
            ax.axvline(0, ls="--", c="grey", lw=0.8)
        ax.set_xlabel(p)
    axes[0].set_ylabel("3-2-1 crack")
    axes[0].legend(fontsize=8, framealpha=0.9)
    plt.savefig("scatters_by_era.png", dpi=150, bbox_inches="tight")
    plt.close()

    #hinges on the deviations only
    for p in ["gas_dev", "dist_dev"]:
        df[f"{p}_deficit"] = np.maximum(0, -df[p])
        df[f"{p}_surplus"] = np.maximum(0,  df[p])

    df["post2022"] = (df.index >= "2022-03-01").astype(int)
    #diagnostics
    preds = ["gas_dev_deficit",  "gas_dev_surplus",
            "dist_dev_deficit", "dist_dev_surplus",
            "refinery_util", "dollar_index", "post2022"]

    print(df[preds].corr().round(2))

    plot_acf(df.crack_321.dropna(), lags=52)
    plt.savefig("acf.png", dpi=150, bbox_inches="tight")
    plt.close()

    #save
    model_cols = ["crack_321"] + preds + ["sin_yr", "cos_yr"]
    out = df[model_cols].dropna()
    out.to_csv("crack_model_data.csv")
    print(out.shape, out.index.min().date(), out.index.max().date())





    df = pd.read_csv("crack_model_data.csv", index_col=0, parse_dates=True)

    features = ["gas_dev_deficit",  "gas_dev_surplus",
                "dist_dev_deficit", "dist_dev_surplus",
                "refinery_util", "post2022",
                "sin_yr", "cos_yr", "dollar_index"]

    cont = ["gas_dev_deficit", "gas_dev_surplus",
            "dist_dev_deficit", "dist_dev_surplus",
            "refinery_util", "dollar_index"]



    X = df[features].copy()
    X[cont] = (X[cont] - X[cont].mean()) / X[cont].std()
    y = df.crack_321.values

    #model
    with pm.Model(coords={"feature": features}) as model:
        intercept = pm.Normal("intercept", mu=25, sigma=15)
        beta = pm.Normal("beta", mu=0, sigma=5, dims="feature")
        sigma = pm.HalfNormal("sigma", 10)
        nu = pm.Gamma("nu", alpha=2, beta=0.1)

        mu = intercept + pm.math.dot(X.values, beta)
        pm.StudentT("y", nu=nu, mu=mu, sigma=sigma, observed=y)

        idata = pm.sample(1000, tune=1000, chains=4,nuts_sampler="numpyro", random_seed=42)

    #check
    print(az.summary(idata, var_names=["intercept", "beta", "sigma", "nu"], round_to=2))
    print("divergences:", int(idata.sample_stats.diverging.sum()))


    #forest plot
    az.plot_forest(idata, var_names=["beta"], combined=True, hdi_prob=0.9)
    plt.axvline(0, ls="--", c="grey")
    plt.savefig("coefficients.png", dpi=150, bbox_inches="tight")
    plt.close()

    #posterior predictive
    with model:
        pm.sample_posterior_predictive(idata, extend_inferencedata=True, random_seed=42)

    az.plot_ppc(idata, num_pp_samples=100)
    plt.savefig("ppc_density.png", dpi=150, bbox_inches="tight")
    plt.close()

    ppc = az.extract(idata, group="posterior_predictive", var_names="y").values
    lo, hi = np.percentile(ppc, [5, 95], axis=1)
    pred = ppc.mean(axis=1)
    
    #fitted vs actual
    plt.figure(figsize=(13, 5))
    plt.fill_between(df.index, lo, hi, alpha=0.3, label="90% predictive")
    plt.plot(df.index, pred, lw=1.2, label="posterior mean")
    plt.plot(df.index, y, lw=1.0, c="k", alpha=0.7, label="actual")
    plt.legend(); plt.ylabel("3-2-1 crack ($/bbl)")
    plt.savefig("fit_vs_actual.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("90% coverage:", ((y >= lo) & (y <= hi)).mean().round(3))

    #residual autocorrelation
    resid = y - pred
    plot_acf(resid, lags=52)
    plt.savefig("resid_acf.png", dpi=150, bbox_inches="tight")
    plt.close()

    az.plot_pair(idata, var_names=["beta"],
             coords={"feature": ["post2022", "dollar_index"]},
             kind="scatter", marginals=True)
    plt.savefig("ridge_post2022_dollar.png", dpi=150, bbox_inches="tight")
    plt.close()

    b = az.extract(idata, var_names="beta")
    print("posterior corr:", np.corrcoef(b.sel(feature="post2022").values, b.sel(feature="dollar_index").values)[0, 1].round(3))
if __name__ == "__main__":
    main()
