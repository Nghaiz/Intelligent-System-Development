# ag2/agents.py
def act(self, obs, explore=False):
    demand, rival_prev = obs[0], obs[1]
    if demand == 0:
        return KEEP
    return DISCOUNT if rival_prev == DISCOUNT else KEEP
