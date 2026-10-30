# ag2/market.py
def conversion_probability(cfg, demand, own, rival):
    """p_i(t) = clip(0,30 + 0,25 d + 0,15 a_i − 0,10 a_−i, 0,05, 0,95)."""
    p = (cfg.base_conversion + cfg.demand_effect * demand
         + cfg.own_discount_effect * own - cfg.rival_discount_effect * rival)
    return min(max(p, cfg.p_min), cfg.p_max)
