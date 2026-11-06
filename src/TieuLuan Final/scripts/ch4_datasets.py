"""Chương 4: xuất ch4_datasets.json và ba hình khảo sát dữ liệu (chuỗi, phân bố, ACF)."""
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from tlf.ch4_data import POWER_CSV, POWER_META, PRICE_CSV, TICKERS, WEATHER_CSV, frame_of
from tlf.common import save_json
from tlf.plot_style import C, W, save

TICKER_COLOR = {"FPT": C["blue"], "VCB": C["green"], "HPG": C["orange"], "VIC": C["purple"]}
BINS = 40
SAMPLE_ROWS = 4


def _hist(values):
    v, e = np.histogram(values, bins=BINS)
    return {"labels": [round(float(x), 4) for x in (e[:-1] + e[1:]) / 2], "values": v.tolist()}


def _sample(df, cols, ndigits=2):
    picks = df.iloc[np.linspace(0, len(df) - 1, SAMPLE_ROWS).astype(int)]
    rows = []
    for ts, r in picks.iterrows():
        rows.append({"time": str(ts), **{c: round(float(r[c]), ndigits) for c in cols}})
    return rows


def vn30_meta():
    raw = pd.read_csv(PRICE_CSV, parse_dates=["Date"]).set_index("Date")
    rets = np.concatenate([np.diff(np.log(g.Close.to_numpy())) for _, g in raw.groupby("Ticker")])
    per = {t: int((raw.Ticker == t).sum()) for t in TICKERS}
    return {"id": "vn30", "name": "VN30: FPT, VCB, HPG, VIC", "source": "Yahoo Finance (yfinance)",
            "url": "https://finance.yahoo.com/quote/FPT.VN/history/",
            "task": "Dự báo giá đóng cửa 1 đến 10 phiên", "n_samples": len(raw), "n_features": 8,
            "size_mb": PRICE_CSV.stat().st_size / 2**20, "license": "Dữ liệu công khai, chỉ dùng học thuật",
            "note": "Giá đã điều chỉnh cổ tức và chia tách; phiên khối lượng 0 bị loại.", "per_ticker": per,
            "date_start": str(raw.index.min().date()), "date_end": str(raw.index.max().date()),
            "sample_rows": _sample(raw[(raw.Ticker == "FPT") & (raw.Volume > 0)], ["Open", "High", "Low", "Close", "Volume"], 0),
            "distribution": _hist(rets[np.abs(rets) < 0.08])}


def weather_meta():
    raw = pd.read_csv(WEATHER_CSV, parse_dates=["Time"]).set_index("Time")
    return {"id": "weather", "name": "Thời tiết Hà Nội theo giờ", "source": "Open-Meteo Archive API (ERA5)",
            "url": "https://open-meteo.com/en/docs/historical-weather-api", "task": "Dự báo nhiệt độ 1 đến 24 giờ",
            "n_samples": len(raw), "n_features": 10, "size_mb": WEATHER_CSV.stat().st_size / 2**20,
            "license": "CC BY 4.0", "note": "Tái phân tích tại lưới gần Hà Nội, không phải trạm đo.",
            "date_start": str(raw.index.min().date()), "date_end": str(raw.index.max().date()),
            "sample_rows": _sample(raw, list(raw.columns)[:5], 1), "distribution": _hist(raw.temperature_2m),
            "missing": int(raw.isna().sum().sum()), "temp_min": float(raw.temperature_2m.min()),
            "temp_max": float(raw.temperature_2m.max())}


def power_meta():
    raw = pd.read_csv(POWER_CSV, parse_dates=["Time"]).set_index("Time")
    meta = json.loads(POWER_META.read_text(encoding="utf-8"))
    return {"id": "power", "name": "Individual Household Electric Power Consumption", "source": "UCI id 235",
            "url": "https://archive.ics.uci.edu/dataset/235", "task": "Dự báo công suất 1 đến 24 giờ",
            "n_samples": len(raw), "n_features": 10, "size_mb": POWER_CSV.stat().st_size / 2**20, "license": "CC BY 4.0",
            "note": "Gộp từ đo theo phút; bản theo phút không giữ lại trên đĩa.", "n_minutes": meta["n_minutes"],
            "missing_pct": 100 * meta["n_missing_minutes"] / meta["n_minutes"], "raw_mb": meta["txt_mb"],
            "missing_hours": int(raw.Global_active_power.isna().sum()),
            "date_start": str(raw.index.min().date()), "date_end": str(raw.index.max().date()),
            "sample_rows": _sample(raw.dropna(), ["Global_active_power", "Global_reactive_power", "Voltage",
                                                  "Sub_metering_3"]),
            "distribution": _hist(raw.Global_active_power.dropna())}


def _acf_axes(ax, lags, sets):
    for label, v, color in sets:
        ax.plot(lags, acf(v, nlags=lags[-1], fft=True)[1:], color=color, label=label, lw=1.8)
    ax.axhline(0, color=C["gray"], lw=1)
    ax.set_xlabel("độ trễ")
    ax.set_title("ACF", fontsize=12)


def fig_vn30():
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.7), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    all_ret = []
    for t in TICKERS:
        c = frame_of(t)["target"]
        ax[0].plot(c.index, c / c.iloc[0], color=TICKER_COLOR[t], label=t, lw=1.2)
        all_ret.append(np.diff(np.log(c.to_numpy())))
    ax[0].set_yscale("log")
    ax[0].set_title("Giá chuẩn hóa", fontsize=12)
    ax[0].legend(ncols=2, fontsize=10, loc="upper left", columnspacing=0.8, handlelength=1)
    ax[0].tick_params(axis="x", labelsize=10, rotation=30)
    r = np.concatenate(all_ret)
    ax[1].hist(100 * r, bins=BINS, range=(-8, 8), color=C["blue"])
    ax[1].set_title("Lợi suất ngày", fontsize=12)
    ax[1].set_xlabel("%")
    fpt = np.diff(np.log(frame_of("FPT")["target"].to_numpy()))
    _acf_axes(ax[2], np.arange(1, 31), (("lợi suất", fpt, C["blue"]), ("|lợi suất|", np.abs(fpt), C["orange"])))
    ax[2].legend(fontsize=10)
    fig.tight_layout()
    save(fig, "ch4_eda_vn30")


def fig_hourly(name, series, unit, season_label, color):
    fig, ax = plt.subplots(1, 3, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    daily = series.resample("D").mean()
    ax[0].plot(daily.index, daily, color=color, lw=0.8)
    ax[0].set_title(f"Trung bình ngày ({unit})", fontsize=12)
    span = daily.index[-1].year - daily.index[0].year
    ax[0].xaxis.set_major_locator(mdates.YearLocator(max(1, span // 5)))
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax[0].tick_params(axis="x", labelsize=10, rotation=30)
    ax[1].hist(series, bins=BINS, color=color)
    ax[1].set_title("Phân bố theo giờ", fontsize=12)
    ax[1].set_xlabel(unit)
    _acf_axes(ax[2], np.arange(1, 169), ((season_label, series.to_numpy(), color),))
    ax[2].set_xticks([24, 72, 120, 168])
    ax[2].tick_params(axis="x", labelsize=10)
    fig.tight_layout()
    save(fig, name)


def main():
    save_json("ch4_datasets", {"datasets": [vn30_meta(), weather_meta(), power_meta()]})
    fig_vn30()
    fig_hourly("ch4_eda_weather", frame_of("weather")["target"], "°C", "nhiệt độ", C["orange"])
    fig_hourly("ch4_eda_power", frame_of("power")["target"], "kW", "công suất", C["teal"])


if __name__ == "__main__":
    main()
