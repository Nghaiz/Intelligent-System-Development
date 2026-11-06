"""Chương 4: tải ba nguồn dữ liệu chuỗi thời gian vào data/ch4/ (bỏ qua tệp đã có).

    vn30_prices.csv         giá ngày FPT, VCB, HPG, VIC từ Yahoo Finance, đã điều chỉnh cổ tức và chia tách
    hanoi_hourly.csv        thời tiết Hà Nội theo giờ từ Open-Meteo Archive (tái phân tích ERA5)
    household_power_hourly.csv  công suất hộ gia đình UCI id 235, gộp từ phút thành giờ
"""
import datetime as dt
import io
import json
import zipfile

import pandas as pd
import requests

from tlf.ch4_data import OUT_DIR, POWER_CSV, POWER_META, PRICE_CSV, TICKERS, WEATHER_CSV
from tlf.mem import require_disk, slot

START_STOCK = "2012-01-01"
START_WEATHER = "2016-01-01"
ERA5_LAG_DAYS = 7
HANOI = {"latitude": 21.0285, "longitude": 105.8542, "timezone": "Asia/Bangkok"}
WEATHER_VARS = ("temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
                "precipitation", "cloud_cover")
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
UCI_ZIP = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
TIMEOUT = 300
CHUNK = 300_000


def download_prices():
    import yfinance as yf
    frames = []
    for tk in TICKERS:
        raw = yf.Ticker(f"{tk}.VN").history(start=START_STOCK, auto_adjust=True, actions=False)
        if raw.empty:
            raise RuntimeError(f"Yahoo Finance không trả dữ liệu cho {tk}.VN")
        d = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
        d.insert(0, "Ticker", tk)
        frames.append(d.rename_axis("Date").reset_index())
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(PRICE_CSV, index=False)
    return out


def download_weather():
    end = (dt.date.today() - dt.timedelta(days=ERA5_LAG_DAYS)).isoformat()
    params = {**HANOI, "start_date": START_WEATHER, "end_date": end, "hourly": ",".join(WEATHER_VARS)}
    r = requests.get(OPEN_METEO, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    out = pd.DataFrame(r.json()["hourly"]).rename(columns={"time": "Time"})
    out.to_csv(WEATHER_CSV, index=False)
    return out


def _hourly_power(handle):
    """Đọc từng khúc để không giữ 2 triệu dòng trong RAM: cộng dồn tổng và số phút đo được của mỗi giờ."""
    sums, counts, rows, missing = [], [], 0, 0
    for c in pd.read_csv(handle, sep=";", na_values="?", low_memory=False, chunksize=CHUNK):
        stamp = pd.to_datetime(c.pop("Date") + " " + c.pop("Time"), format="%d/%m/%Y %H:%M:%S")
        c = c.astype("float32").set_index(stamp.dt.floor("h"))
        rows += len(c)
        missing += int(c.Global_active_power.isna().sum())
        sums.append(c.groupby(level=0).sum(min_count=1))
        counts.append(c.groupby(level=0).count())
    total = pd.concat(sums).groupby(level=0).sum(min_count=1)
    return total / pd.concat(counts).groupby(level=0).sum(), rows, missing


def download_power():
    """Chỉ giữ bản theo giờ; tệp theo phút (127 MB khi giải nén) không lưu lại đĩa."""
    r = requests.get(UCI_ZIP, timeout=TIMEOUT)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".txt"))
        out, rows, missing = _hourly_power(z.open(name))
    meta = {"n_minutes": rows, "n_missing_minutes": missing, "zip_mb": len(r.content) / 2**20,
            "columns": 9, "txt_mb": z.getinfo(name).file_size / 2**20}
    POWER_META.write_text(json.dumps(meta), encoding="utf-8")
    out.rename_axis("Time").to_csv(POWER_CSV)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    require_disk(1, "tải dữ liệu ch4")
    for path, fn in ((PRICE_CSV, download_prices), (WEATHER_CSV, download_weather), (POWER_CSV, download_power)):
        if path.exists():
            print("đã có", path.name, flush=True)
            continue
        with slot(1, f"tải {path.name}"):
            df = fn()
        print(path.name, df.shape, f"{path.stat().st_size / 2**20:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
