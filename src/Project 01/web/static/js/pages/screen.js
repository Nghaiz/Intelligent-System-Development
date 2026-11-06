// Sàng lọc thích ứng: mỗi lượt chính sách RL chọn câu hỏi kế tiếp, thước xác suất nhảy theo đáp án.
import { $, api, ApiError, esc, guarded, num, pct, pendingHtml, signed } from "../lib.js";

export const lead = "Chính sách học tăng cường chọn câu hỏi đáng hỏi nhất ở mỗi bước và tự dừng khi đã đủ thông tin, thay vì hỏi đủ 21 câu.";
const FULL = 21;
const sid = () => crypto.randomUUID().replaceAll("-", "");

// Lựa chọn có thể là [giá trị, nhãn], {value, label} hoặc một giá trị trần
const opt = (o) => (Array.isArray(o) ? { value: o[0], label: o[1] } : (o && typeof o === "object" ? o : { value: o, label: String(o) }));

export async function mount(el) {
  el.innerHTML = `<div class="two"><div class="stack">
      <section class="lead"><h2>Nguy cơ ước tính</h2><div class="gauge" id="gauge"><i class="fill" style="width:0"></i></div>
        <div class="gauge-scale"><span>0%</span><span>50%</span><span>100%</span></div><p id="prob" class="risk-num" style="margin-top:8px">—</p></section>
      <section class="lead question" id="ask"></section>
      <section class="lead"><header><h2>Đã hỏi</h2><span class="note" id="count">0 trên ${FULL} câu</span></header><div><div class="of21" id="of21">${"<i></i>".repeat(FULL)}</div><ol class="asked" id="asked" style="margin-top:12px"></ol></div></section>
    </div><div class="stack">
      <section class="lead"><h2>Giá trị Q của từng hành động</h2><div id="qv"><p class="note">Chưa có bước nào.</p></div></section>
      <section class="lead"><h2>So với hỏi đủ 21 câu</h2><div id="cmp"></div></section>
    </div></div>`;
  let answers = {}; let history = []; let last = null; let session = sid();
  const ask = $("#ask", el);

  const paint = (r) => {
    $(".fill", el).style.width = `${(r.prob ?? 0) * 100}%`;
    $("#prob", el).textContent = pct(r.prob);
    const q = Object.entries(r.q_values || {}).sort((a, b) => b[1] - a[1]).slice(0, 9);
    if (q.length) {
      const hi = Math.max(...q.map(([, v]) => Math.abs(v)), 1e-9);
      const name = (k) => r.labels?.[k] || k;
      $("#qv", el).innerHTML = `<div class="shap">${q.map(([k, v]) => `<div class="shap-row"><span class="name" title="${esc(name(k))}">${esc(name(k))}</span><span class="shap-track"><i class="shap-bar ${v < 0 ? "neg" : ""}" style="left:${v < 0 ? 50 - Math.abs(v) / hi * 50 : 50}%;width:${Math.abs(v) / hi * 50}%"></i></span><span class="val">${num(v, 2)}</span></div>`).join("")}</div>`;
    }
    $("#count", el).textContent = `${history.length} trên ${FULL} câu`;
    [...$("#of21", el).children].forEach((c, i) => c.classList.toggle("on", i < history.length));
    $("#asked", el).innerHTML = history.map((h, i) => {
      const d = i ? h.prob - history[i - 1].prob : null;
      return `<li><i>${i + 1}</i><span>${esc(h.q)}: <b>${esc(h.label)}</b></span><b class="${d > 0 ? "up" : "down"}">${d === null ? "" : `${signed(d)} điểm`}</b><b>${pct(h.prob, 0)}</b></li>`;
    }).join("");
  };
  const renderAsk = (r) => {
    if (r.stop) {
      ask.innerHTML = `<h2>Đủ thông tin sau ${history.length} câu</h2><p>Nguy cơ ước tính ${pct(r.prob)}. Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ.</p>
        <div class="row" style="margin-top:12px"><button class="btn" id="again">Sàng lọc lại</button></div>`;
      $("#again", el).addEventListener("click", restart);
      return;
    }
    ask.innerHTML = `<h2>${esc(r.question_vi)}</h2><div class="opts">${(r.options || []).map((o, i) => `<button data-i="${i}">${esc(opt(o).label)}</button>`).join("")}</div>`;
  };
  const next = async () => {
    const r = await api("/screen/next", { json: { answers, session_id: session } });
    last = r;
    if (history.length) history.at(-1).prob = r.prob;
    paint(r); renderAsk(r);
  };
  const restart = () => { answers = {}; history = []; session = sid(); next().catch(show); };
  const show = (e) => { if (!(e instanceof ApiError)) throw e; ask.innerHTML = pendingHtml(e); };
  ask.addEventListener("click", (e) => {
    const b = e.target.closest("[data-i]");
    if (!b) return;
    const o = opt(last.options[Number(b.dataset.i)]);
    answers[last.feature] = o.value;
    history.push({ q: last.labels?.[last.feature] || last.feature, label: o.label, prob: last.prob });
    next().catch(show);
  });
  const stopCmp = guarded($("#cmp", el), async () => {
    const m = await api("/metrics/rl_compare");
    $("#cmp", el).innerHTML = compareHtml(m);
  }, 15000);
  const stop = guarded(ask, next);
  return () => { stop(); stopCmp(); };
}

const POLICY_VI = { fixed_importance: "Thứ tự cố định", random: "Ngẫu nhiên", greedy_info: "Tham lam thông tin", greedy_info_per_cost: "Tham lam theo chi phí" };

// So DQN (trung bình các seed của λ có nhiều seed nhất) với các chính sách nền ở cùng số câu và với hỏi đủ 21 câu; số lấy từ rl_compare.
function compareHtml(m) {
  const runs = Object.entries(m.dqn_adaptive || {}).sort((a, b) => b[1].length - a[1].length)[0];
  if (!runs) return '<p class="note">Tệp số liệu chưa có kết quả DQN.</p>';
  const [lam, seeds] = runs;
  const mean = (k) => seeds.reduce((s, r) => s + r[k], 0) / seeds.length;
  const k = Math.round(mean("questions"));
  const at = (name, q) => m.policies[name].curve.find((c) => c.k === q);
  const rows = [[`Double DQN (${seeds.length} seed)`, mean("questions"), mean("cost"), mean("auc"), mean("f1")],
    ...Object.keys(POLICY_VI).map((p) => { const c = at(p, k); return [POLICY_VI[p], c.k, c.cost, c.auc, c.f1]; }),
    ["Hỏi đủ 21 câu", 21, at("fixed_importance", FULL).cost, at("fixed_importance", FULL).auc, at("fixed_importance", FULL).f1]];
  return `<div class="tbl-wrap"><table class="tbl"><thead><tr>${["Chính sách", "Số câu", "Chi phí", "AUC", "F1"].map((c) => `<th>${c}</th>`).join("")}</tr></thead>
    <tbody>${rows.map(([n, ...v]) => `<tr><td>${esc(n)}</td>${v.map((x, i) => `<td class="n">${num(x, i < 2 ? 1 : 3)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>
    <p class="note" style="margin-top:8px">Tập kiểm tra ${num(m.n_test)} mẫu, λ = ${esc(lam.replace(".", ","))}.</p>`;
}
