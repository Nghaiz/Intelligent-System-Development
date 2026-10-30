# ag2/market.py
def expected_reward(cfg, demand, own, rival):
    """E[r_i | d, a_i, a_−i] = N_d · p_i · đơn giá thưởng (dùng cho bảng phân tích và đánh giá chính xác)."""
    return customers(cfg, demand) * conversion_probability(cfg, demand, own, rival) * unit_reward(cfg, own)
