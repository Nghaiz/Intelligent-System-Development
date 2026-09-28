"""Hạng mục 2 (Ch.2): giá cổ phiếu AMZN, bài hồi quy dự báo giá đóng cửa phiên kế tiếp.

Hai điểm khác báo cáo mẫu: (1) mọi scaler fit trên train; (2) không dự báo mức giá (giá test 2019-2024
vượt xa giá lớn nhất của train) mà dự báo log-return r_{t+1} = ln(P_{t+1}/P_t), rồi quy đổi
P̂_{t+1} = P_t · exp(r̂) để báo sai số bằng USD.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, pacf

from .export import DATA

CSV = DATA / "amzn" / "AMZN.csv"
FEATURES = ["Close", "Open", "High", "Low", "Volume", "Return", "MA20", "RSI14"]
PRICE_COLS = ["Close", "Open", "High", "Low", "MA20"]   # chia cho Close ngày cuối cửa sổ
WINDOW = 30
FRACTIONS = (0.70, 0.15, 0.15)


def download_amzn(start="1997-05-15", end="2024-09-25"):
    """Tải một lần từ Yahoo Finance, lưu CSV để các lần chạy sau tái lập đúng dữ liệu."""
    import yfinance as yf
    df = yf.download("AMZN", start=start, end=end, auto_adjust=False, progress=False)
    df.columns = [c[0] for c in df.columns]
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    df.index.name = "Date"
    df.to_csv(CSV, float_format="%.6f")
    return df


def rsi(close, n=14):
    """RSI của Wilder: 100 − 100/(1 + TB tăng / TB giảm), trung bình mũ hệ số 1/n."""
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def add_features(df):
    """Đặc trưng tài chính; các dòng đầu thiếu MA/RSI bị bỏ."""
    df = df.copy()
    df["Return"] = np.log(df.Close / df.Close.shift(1))
    df["MA20"] = df.Close.rolling(20).mean()
    df["MA50"] = df.Close.rolling(50).mean()
    df["Volatility20"] = df.Return.rolling(20).std() * np.sqrt(252)
    df["RSI14"] = rsi(df.Close)
    return df.dropna()


def load_amzn():
    df = pd.read_csv(CSV, parse_dates=["Date"], index_col="Date")
    return add_features(df)


def adf(series):
    """Kiểm định Augmented Dickey-Fuller: p < 0,05 thì bác bỏ giả thuyết có nghiệm đơn vị (chuỗi dừng)."""
    stat, p, lags, nobs, crit, _ = adfuller(series.dropna(), autolag="AIC")
    return {"stat": stat, "p": p, "lags": lags, "nobs": nobs, "crit5": crit["5%"]}


def acf_pacf(series, nlags=40):
    s = series.dropna()
    return acf(s, nlags=nlags), pacf(s, nlags=nlags), 1.96 / np.sqrt(len(s))


def _feature_values(df):
    vals = df[FEATURES].to_numpy(np.float64).copy()
    vals[:, FEATURES.index("Volume")] = np.log1p(vals[:, FEATURES.index("Volume")])
    return vals


def _relative_prices(w, close_t):
    """Chia các cột giá của cửa sổ cho Close ngày cuối: mô hình thấy giá tương đối, không thấy mức giá."""
    w = w.copy()
    pcols = [FEATURES.index(c) for c in PRICE_COLS]
    w[:, pcols] = w[:, pcols] / close_t
    return w


def make_windows(df, window=WINDOW):
    """Cửa sổ trượt: X[i] = 30 phiên kết thúc ở t, nhãn = log-return của phiên t+1.

    Giá trong cửa sổ chia cho Close ngày t (bất biến theo mức giá), Volume lấy log.
    Trả về X thô (chưa z-score), nhãn, giá P_t, giá P_{t+1} và ngày của P_{t+1}.
    """
    vals = _feature_values(df)
    close = df.Close.to_numpy(np.float64)
    X, y, p_now, p_next, dates = [], [], [], [], []
    for t in range(window - 1, len(df) - 1):
        X.append(_relative_prices(vals[t - window + 1:t + 1], close[t]))
        y.append(np.log(close[t + 1] / close[t]))
        p_now.append(close[t])
        p_next.append(close[t + 1])
        dates.append(df.index[t + 1])
    return np.stack(X), np.array(y), np.array(p_now), np.array(p_next), pd.DatetimeIndex(dates)


def latest_window(df, window=WINDOW):
    """Cửa sổ 30 phiên mới nhất (chưa có nhãn) để dự báo phiên kế tiếp; cùng biến đổi như make_windows."""
    close = float(df.Close.iloc[-1])
    return _relative_prices(_feature_values(df)[-window:], close), close, df.index[-1]


def time_split(n, fractions=FRACTIONS):
    """Chia theo thời gian, không xáo trộn: train là quá khứ, test là giai đoạn cuối."""
    a = int(n * fractions[0])
    b = a + int(n * fractions[1])
    return {"train": np.arange(0, a), "val": np.arange(a, b), "test": np.arange(b, n)}


class WindowScaler:
    """z-score từng đặc trưng cho X và z-score cho nhãn, cả hai chỉ fit trên cửa sổ train."""

    def fit(self, X, y):
        flat = X.reshape(-1, X.shape[-1])
        self.mean_, self.std_ = flat.mean(axis=0), flat.std(axis=0) + 1e-8
        self.y_mean_, self.y_std_ = float(y.mean()), float(y.std())
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def transform_y(self, y):
        return ((y - self.y_mean_) / self.y_std_).astype(np.float32)

    def inverse_y(self, z):
        return np.asarray(z, dtype=np.float64) * self.y_std_ + self.y_mean_

    def state(self):
        return {"mean": self.mean_.tolist(), "std": self.std_.tolist(),
                "y_mean": self.y_mean_, "y_std": self.y_std_}


def prepare_stock(df=None):
    """Toàn bộ tiền xử lý AMZN. Trả về dict X/y từng phần (đã chuẩn hoá) + giá để quy đổi USD."""
    df = load_amzn() if df is None else df
    X, y, p_now, p_next, dates = make_windows(df)
    split = time_split(len(X))
    tr = split["train"]
    scaler = WindowScaler().fit(X[tr], y[tr])
    return {"X": {k: scaler.transform(X[i]) for k, i in split.items()},
            "y": {k: scaler.transform_y(y[i]) for k, i in split.items()},
            "ret": {k: y[i] for k, i in split.items()},
            "p_now": {k: p_now[i] for k, i in split.items()},
            "p_next": {k: p_next[i] for k, i in split.items()},
            "dates": {k: dates[i] for k, i in split.items()},
            "scaler": scaler, "split": split, "df": df, "features": FEATURES}


def to_price(scaler, z_pred, p_now):
    """Đầu ra mô hình (return đã z-score) -> giá dự báo USD: P̂_{t+1} = P_t · exp(r̂)."""
    return p_now * np.exp(scaler.inverse_y(z_pred))
