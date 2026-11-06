"""Thị trường thương mại điện tử hai cửa hàng của Laboratory 02 (nguồn duy nhất của mô hình).

Mỗi bước, cửa hàng i chọn a_i ∈ {0: giữ giá, 1: giảm giá}. Xác suất chuyển đổi của một khách
p_i = clip(0,30 + 0,25 d + 0,15 a_i − 0,10 a_−i; 0,05; 0,95); số đơn S_i ~ Binomial(N_d, p_i);
phần thưởng r_i = S_i [P (1 − δ a_i) − C].
"""
from dataclasses import dataclass, replace

import numpy as np

KEEP, DISCOUNT = 0, 1


@dataclass(frozen=True)
class MarketConfig:
    """Tham số mô phỏng; mặc định là cấu hình tham chiếu của đề bài."""
    base_price: float = 10.0
    unit_cost: float = 5.0
    discount_rate: float = 0.20
    customers_low: int = 10
    customers_high: int = 20
    base_conversion: float = 0.30
    demand_effect: float = 0.25
    own_discount_effect: float = 0.15
    rival_discount_effect: float = 0.10
    p_min: float = 0.05
    p_max: float = 0.95
    steps: int = 20
    demand_persist: float = 0.5   # P(d_{t+1} = d_t); 0,5 = nhu cầu độc lập như đề bài
    reward_kind: str = "profit"   # "profit" (đề bài) hoặc "revenue" (biến thể câu 5, 9)

    def with_(self, **kw):
        return replace(self, **kw)


def conversion_probability(cfg, demand, own, rival):
    """p_i(t) = clip(0,30 + 0,25 d + 0,15 a_i − 0,10 a_−i, 0,05, 0,95)."""
    p = (cfg.base_conversion + cfg.demand_effect * demand
         + cfg.own_discount_effect * own - cfg.rival_discount_effect * rival)
    return min(max(p, cfg.p_min), cfg.p_max)


def customers(cfg, demand):
    """Số khách tiềm năng N_t: 10 khi nhu cầu thấp, 20 khi nhu cầu cao."""
    return cfg.customers_high if demand else cfg.customers_low


def unit_reward(cfg, action):
    """Phần thưởng trên mỗi đơn: biên lợi nhuận P(1−δa) − C, hoặc doanh thu P(1−δa)."""
    price = cfg.base_price * (1.0 - cfg.discount_rate * action)
    return price - cfg.unit_cost if cfg.reward_kind == "profit" else price


def expected_reward(cfg, demand, own, rival):
    """E[r_i | d, a_i, a_−i] = N_d · p_i · đơn giá thưởng (dùng cho bảng phân tích và đánh giá chính xác)."""
    return customers(cfg, demand) * conversion_probability(cfg, demand, own, rival) * unit_reward(cfg, own)


class ECommerceMarket:
    """Môi trường hai cửa hàng. Trạng thái chung: (d_t, a_1(t−1), a_2(t−1)).

    Số ngẫu nhiên chung (common random numbers): nhu cầu và doanh số của từng cửa hàng có bộ sinh riêng,
    tách từ một seed. Hai lượt đánh giá cùng seed thấy cùng chuỗi nhu cầu; cửa hàng nào chọn cùng hành động
    thì nhận đúng cùng doanh số. Chênh lệch giữa hai chính sách vì vậy không do may rủi của mẫu.
    """

    def __init__(self, cfg=MarketConfig(), seed=0):
        self.cfg = cfg
        ss = np.random.SeedSequence(seed)
        self.rng_demand, *self.rng_sales = (np.random.default_rng(s) for s in ss.spawn(3))
        self.reset()

    def reset(self):
        """Đầu episode: rút nhu cầu ngẫu nhiên đều, hành động trước của cả hai đặt bằng 0."""
        self.t = 0
        self.demand = int(self.rng_demand.integers(2))
        self.prev = (KEEP, KEEP)
        return self.joint_state()

    def joint_state(self):
        return self.demand, self.prev[0], self.prev[1]

    def _next_demand(self):
        if self.rng_demand.random() < self.cfg.demand_persist:
            return self.demand
        return 1 - self.demand

    def step(self, actions):
        """Thực hiện cặp hành động đồng thời; trả (phần thưởng, doanh số, kết thúc episode)."""
        a1, a2 = actions
        n = customers(self.cfg, self.demand)
        sales, rewards = [], []
        for i, (own, rival) in enumerate(((a1, a2), (a2, a1))):
            p = conversion_probability(self.cfg, self.demand, own, rival)
            s = int(self.rng_sales[i].binomial(n, p))
            sales.append(s)
            rewards.append(s * unit_reward(self.cfg, own))
        self.prev = (a1, a2)
        self.demand = self._next_demand()
        self.t += 1
        return rewards, sales, self.t >= self.cfg.steps
