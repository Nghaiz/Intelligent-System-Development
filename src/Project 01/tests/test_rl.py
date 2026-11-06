"""RL: hợp đồng đặc trưng, MDP nhỏ trên dữ liệu giả, che hành động, baseline, policy thật nếu đã huấn luyện."""
import pytest
import torch

from dnx.rl import baselines as B
from dnx.rl import dqn
from dnx.rl.env import ScreenEnv, log_score, make_state, valid_actions
from dnx.rl.features import ALIASES, BY_KEY, COSTS, FEATURES, KEYS, N, STOP, age_bin, answer_value
from dnx.rl.predictor import DIR, MaskedMLP


def toy(n=64):
    torch.manual_seed(0)
    return MaskedMLP().eval(), torch.randn(n, N)


def test_feature_table():
    assert N == 21 and len(set(KEYS)) == 21
    for f in FEATURES:
        assert f.question_vi.endswith("?") and len(f.options) >= 2 and f.cost > 0


def test_age_bin_matches_cdc():
    assert (age_bin(20), age_bin(25), age_bin(52), age_bin(79), age_bin(90)) == (1, 2, 7, 12, 13)


def test_answer_value_maps_profile_keys():
    assert answer_value("smoking", "current") == (BY_KEY["smoker"], 1.0)
    assert answer_value("smoking", "unknown") is None
    assert answer_value("sex", "female") == (BY_KEY["sex"], 0.0)
    assert answer_value("age", 52) == (BY_KEY["age"], 7.0)
    assert answer_value("hypertension", 1) == (BY_KEY[ALIASES["hypertension"]], 1.0)
    assert answer_value("hba1c", 7) is None and answer_value("bmi", None) is None


def test_state_dim_and_valid_actions():
    m = torch.zeros(2, N)
    m[0, 3] = 1
    s = make_state(torch.ones(2, N), m, torch.tensor([0.1, 0.9]))
    assert s.shape == (2, dqn.STATE_DIM)
    v = valid_actions(m)
    assert v.shape == (2, N + 1) and not v[0, 3] and v[:, STOP].all()


def test_log_score_prefers_true_label():
    assert log_score(torch.tensor(1.0), torch.tensor(0.9)) > log_score(torch.tensor(1.0), torch.tensor(0.1))


def test_env_stop_gives_zero_reward_and_terminates():
    model, x = toy()
    env = ScreenEnv(model, x, 4, lam=0.01)
    r, done = env.step(torch.full((4,), STOP))
    assert (r == 0).all() and done.all()


def test_env_charges_question_cost():
    model, x = toy()
    a, b = ScreenEnv(model, x, 1, lam=0.0), ScreenEnv(model, x, 1, lam=1.0)
    b.idx[:] = a.idx
    ra = a.step(torch.tensor([0]))[0]
    rb = b.step(torch.tensor([0]))[0]
    assert (ra - rb).item() == pytest.approx(COSTS[0] * 100, rel=1e-3)


def test_env_ends_after_all_questions():
    model, x = toy()
    env = ScreenEnv(model, x, 1, lam=0.0)
    done = None
    for k in range(N):
        _, done = env.step(torch.tensor([k]))
    assert done.all()


def test_masked_q_never_repeats_a_question():
    s = torch.zeros(1, dqn.STATE_DIM)
    s[0, N] = 1          # đặc trưng 0 đã hỏi
    q = torch.zeros(1, N + 1)
    q[0, 0] = 100.0
    assert dqn.masked_q(q, s).argmax().item() != 0


def test_replay_wraps_around():
    r = dqn.Replay(5, "cpu")
    s = torch.zeros(3, dqn.STATE_DIM)
    for _ in range(2):
        r.add(s, torch.zeros(3, dtype=torch.long), torch.ones(3), s, torch.zeros(3, dtype=torch.bool))
    assert r.full and r.i == 1


def test_greedy_episode_stops_when_stop_is_best():
    class Q(torch.nn.Module):
        def forward(self, s):
            q = torch.zeros(len(s), N + 1)
            q[:, STOP] = 1
            return q
    model, x = toy(8)
    _, _, mask = dqn.greedy_episode(Q(), model, x)
    assert mask.sum() == 0


def test_short_training_runs_and_reports_history():
    model, x = toy(128)
    cfg = {**dqn.CFG, "steps": 30, "n_env": 8, "replay": 500, "batch": 16, "warmup": 5, "eval_every": 10, "eval_rows": 16}
    best, hist = dqn.train(model, x, x[:16], torch.zeros(16), 0.01, 0, cfg)
    assert best is not None and len(hist) == 3 and {"objective", "questions", "cost"} <= set(hist[0])


def test_trace_asks_each_feature_exactly_once():
    model, x = toy(8)
    for sel in (B.random_order(0), B.fixed_order(list(range(N)))):
        _, order = B.trace(sel, model, x)
        assert order.shape == (N, 8) and all(len(set(order[:, i].tolist())) == N for i in range(8))


def test_fixed_order_follows_rank():
    model, x = toy(4)
    rank = list(reversed(range(N)))
    _, order = B.trace(B.fixed_order(rank), model, x)
    assert order[:, 0].tolist() == rank


def test_greedy_info_gains_nonpositive_for_asked():
    model, x = toy(32)
    g = B.GreedyInfo(x, None, (x[:, 0] > 0).float())
    mask = torch.zeros_like(x)
    mask[:, 2] = 1
    gains = g.gains(model, x, mask, model.prob(x, mask))
    assert (gains[:, 2] < -1e8).all() and (gains[:, 0] > -1e8).all()


def test_tabular_q_index_in_range():
    t = B.TabularQ(list(range(N)), "cpu")
    i = t.index(torch.ones(3, N), torch.tensor([1e-5, 0.5, 0.999]))
    assert ((i >= 0) & (i < t.q.shape[0])).all()


@pytest.mark.skipif(not (DIR / "policy.json").exists(), reason="chưa chọn chính sách")
def test_real_policy_contract():
    import dnx.rl.policy as P
    r = P.next_question({})
    assert set(r) == {"stop", "feature", "question_vi", "options", "prob", "q_values"}
    assert not r["stop"] and r["feature"] in KEYS and r["options"][0]["label"]
    r2 = P.next_question({r["feature"]: r["options"][0]["value"], "age": 60, "sex": "male"})
    assert r2["feature"] != r["feature"] and r2["feature"] not in ("age", "sex")
    assert 0 < r2["prob"] < 1
