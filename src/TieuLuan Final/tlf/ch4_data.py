"""Chương 4: nạp ba nguồn chuỗi thời gian, dựng đặc trưng, chia theo thời gian và tạo cửa sổ trượt.

Sáu chuỗi: FPT, VCB, HPG, VIC (giá đóng cửa theo phiên), weather (nhiệt độ Hà Nội theo giờ), power (công suất
hộ gia đình theo giờ). Mô hình dự báo phần thay đổi so với điểm cuối cửa sổ, nên đầu ra 0 chính là Persistence.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .common import data_dir

OUT_DIR = data_dir("ch4")
PRICE_CSV = OUT_DIR / "vn30_prices.csv"
WEATHER_CSV = OUT_DIR / "hanoi_hourly.csv"
POWER_CSV = OUT_DIR / "household_power_hourly.csv"
POWER_META = OUT_DIR / "household_power_meta.json"
TICKERS = ("FPT", "VCB", "HPG", "VIC")
SPLIT = (0.70, 0.15)                 # tỉ lệ train và validation, phần còn lại là test
ROLL = 20                            # cửa sổ của chỉ báo trượt trên chuỗi giá
RSI_DAYS = 14
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
DAYS_PER_YEAR = 365.25
INTERP_LIMIT_HOURS = 6


@dataclass(frozen=True)
class Spec:
    """Cấu hình một chuỗi: cửa sổ quá khứ, tầm dự báo, biến đổi mục tiêu và đơn vị."""
    id: str
    name: str
    unit: str
    window: int
    horizon: int
    log_target: bool
    freq: str
    season: int = 0                  # chu kỳ của baseline mùa vụ; 0 nghĩa là không dùng


SPECS = {**{t: Spec(t, f"Cổ phiếu {t}", "VNĐ", 60, 10, True, "B") for t in TICKERS},
         "weather": Spec("weather", "Nhiệt độ Hà Nội", "°C", 72, 24, False, "h", HOURS_PER_DAY),
         "power": Spec("power", "Công suất hộ gia đình", "kW", 72, 24, False, "h", HOURS_PER_DAY)}


def _cyclic(frame, phase, name):
    """Mã hóa pha chu kỳ thành sin, cos để 23 giờ và 0 giờ nằm gần nhau."""
    frame[f"{name}_sin"] = np.sin(2 * np.pi * phase)
    frame[f"{name}_cos"] = np.cos(2 * np.pi * phase)


def _rsi(close):
    delta = close.diff()
    up = delta.clip(lower=0).rolling(RSI_DAYS).mean()
    down = (-delta.clip(upper=0)).rolling(RSI_DAYS).mean()
    return up / (up + down + 1e-12)


def stock_frame(ticker):
    """Tám đặc trưng không phụ thuộc mức giá; mọi chỉ báo trượt chỉ nhìn về quá khứ."""
    raw = pd.read_csv(PRICE_CSV, parse_dates=["Date"])
    d = raw[(raw.Ticker == ticker) & (raw.Volume > 0)].set_index("Date").sort_index()
    ret = np.log(d.Close).diff()
    logv = np.log1p(d.Volume)
    x = pd.DataFrame(index=d.index)
    x["target"] = d.Close
    x["ret"] = ret
    x["gap"] = np.log(d.Open / d.Close.shift())
    x["body"] = np.log(d.Close / d.Open)
    x["range"] = np.log(d.High / d.Low)
    x["vol_rel"] = logv - logv.rolling(ROLL).mean()
    x["sma_gap"] = np.log(d.Close / d.Close.rolling(ROLL).mean())
    x["rsi"] = _rsi(d.Close)
    x["volat"] = ret.rolling(ROLL).std()
    return x.dropna()


def weather_frame():
    d = pd.read_csv(WEATHER_CSV, parse_dates=["Time"]).set_index("Time").sort_index()
    d = d.interpolate(limit=INTERP_LIMIT_HOURS)
    x = pd.DataFrame({"target": d.temperature_2m, "humidity": d.relative_humidity_2m,
                      "pressure": d.surface_pressure, "wind": d.wind_speed_10m,
                      "rain": np.log1p(d.precipitation), "cloud": d.cloud_cover}, index=d.index)
    _cyclic(x, x.index.hour / HOURS_PER_DAY, "hour")
    _cyclic(x, x.index.dayofyear / DAYS_PER_YEAR, "doy")
    return x.dropna()


def power_frame():
    """Lỗ ngắn nội suy tuyến tính, đợt mất tín hiệu dài lấy giá trị cùng giờ tuần trước."""
    h = pd.read_csv(POWER_CSV, parse_dates=["Time"]).set_index("Time").interpolate(limit=INTERP_LIMIT_HOURS)
    h = h.fillna(h.shift(HOURS_PER_DAY * DAYS_PER_WEEK)).ffill()
    x = pd.DataFrame({"target": h.Global_active_power, "reactive": h.Global_reactive_power,
                      "voltage": h.Voltage, "kitchen": h.Sub_metering_1, "laundry": h.Sub_metering_2,
                      "climate": h.Sub_metering_3}, index=h.index)
    _cyclic(x, x.index.hour / HOURS_PER_DAY, "hour")
    _cyclic(x, x.index.dayofweek / DAYS_PER_WEEK, "dow")
    return x.dropna()


def frame_of(series):
    if series in TICKERS:
        return stock_frame(series)
    return {"weather": weather_frame, "power": power_frame}[series]()


def split_points(n):
    """Hai ranh giới theo hàng: [0, a) train, [a, b) validation, [b, n) test."""
    a = int(n * SPLIT[0])
    return a, a + int(n * SPLIT[1])


class Prepared:
    """Chuỗi đã chuẩn hóa. Mọi thống kê (mean, std, sd_h) chỉ khớp trên các hàng thuộc tập train."""

    def __init__(self, series, frame=None):
        self.spec = SPECS[series]
        df = frame_of(series) if frame is None else frame
        self.index = df.index
        self.level = df["target"].to_numpy(np.float64)
        self.y = np.log(self.level) if self.spec.log_target else self.level.copy()
        self.columns = [c for c in df.columns if c != "target" or not self.spec.log_target]
        raw = df[self.columns].to_numpy(np.float64)
        self.a, self.b = split_points(len(df))
        self.mean, self.std = raw[:self.a].mean(0), raw[:self.a].std(0) + 1e-9
        self.x = ((raw - self.mean) / self.std).astype(np.float32)
        self.sd_h = self._horizon_scale()

    def last_index(self, part):
        """Chỉ số t của điểm cuối cửa sổ. Nhãn y[t+1..t+H] của train không vượt ranh giới a, val không vượt b."""
        w, h, n = self.spec.window, self.spec.horizon, len(self.y)
        lo, hi = {"train": (w - 1, self.a - h), "val": (self.a - 1, self.b - h), "test": (self.b - 1, n - h)}[part]
        return np.arange(lo, hi)

    def _delta(self, ts):
        return np.stack([self.y[ts + k] - self.y[ts] for k in range(1, self.spec.horizon + 1)], 1)

    def _horizon_scale(self):
        return self._delta(self.last_index("train")).std(0) + 1e-9

    def targets(self, ts):
        """Nhãn đã chuẩn hóa, cỡ (số mẫu, H): (y[t+k] − y[t]) / sd_k."""
        return (self._delta(ts) / self.sd_h).astype(np.float32)

    def windows(self, ts):
        """Cửa sổ thật, cỡ (số mẫu, T, D). Chỉ dùng khi cần mảng NumPy; PyTorch cắt ngay trên GPU."""
        offs = np.arange(-self.spec.window + 1, 1)
        return self.x[ts[:, None] + offs]

    def to_level(self, scaled, ts):
        """Đổi đầu ra chuẩn hóa về đơn vị gốc: y[t] + đầu ra × sd_k, rồi exp nếu mục tiêu là log giá."""
        z = self.y[ts, None] + np.asarray(scaled, np.float64) * self.sd_h
        return np.exp(z) if self.spec.log_target else z

    def truth(self, ts):
        return np.stack([self.level[ts + k] for k in range(1, self.spec.horizon + 1)], 1)

    def persistence(self, ts):
        return np.repeat(self.level[ts, None], self.spec.horizon, 1)

    def seasonal(self, ts):
        """Baseline mùa vụ: giá trị cùng giờ của ngày gần nhất đã quan sát; None nếu chuỗi không có mùa."""
        s = self.spec.season
        if not s:
            return None
        rows = [self.level[ts + k - s * int(np.ceil(k / s))] for k in range(1, self.spec.horizon + 1)]
        return np.stack(rows, 1)
