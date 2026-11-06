// Phòng thí nghiệm nguy cơ: thanh trượt cập nhật tức thì, SHAP và "song sinh số" so hồ sơ hiện tại với mục tiêu.
import { $, $$, api, ApiError, debounce, esc, num, pct, pendingHtml, rulesHtml, shapHtml, signed } from "../lib.js";

export const lead = "Kéo thanh trượt để xem nguy cơ thay đổi ngay lập tức. Song sinh số là chính hồ sơ này sau khi đạt mục tiêu về lối sống và xét nghiệm.";

const NUM = {
  full: [["age", "Tuổi", 18, 90, 1, 52], ["bmi", "BMI", 15, 50, .1, 29.4], ["hba1c", "HbA1c (%)", 4, 13, .1, 6.1], ["glucose", "Đường huyết (mg/dL)", 70, 300, 1, 128]],
  screening: [["age", "Tuổi", 18, 90, 1, 52], ["bmi", "BMI", 15, 50, .1, 29.4], ["gen_hlth", "Sức khỏe tự đánh giá (1 tốt, 5 kém)", 1, 5, 1, 3]],
};
const FLAGS = {
  full: [["hypertension", "Tăng huyết áp"], ["heart_disease", "Bệnh tim"]],
  screening: [["high_bp", "Huyết áp cao"], ["high_chol", "Cholesterol cao"], ["phys_activity", "Có vận động"]],
};
const TARGETS = { full: ["bmi", "hba1c", "glucose"], screening: ["bmi", "gen_hlth"] };
const SMOKING = [["never", "Chưa hút"], ["former", "Đã bỏ"], ["current", "Đang hút"], ["unknown", "Không rõ"]];

const slider = (k, label, lo, hi, step, v, attr, cls = "") => `<label class="slider"><span>${esc(label)}</span><output>${num(v, 1)}</output><input class="${cls}" type="range" ${attr}="${k}" min="${lo}" max="${hi}" step="${step}" value="${v}"></label>`;

export async function mount(el) {
  el.innerHTML = `<div class="two"><div class="stack">
      <div class="twin"><div><h3>Hồ sơ hiện tại</h3><div class="big" id="now">—</div><p class="note" id="meta">&nbsp;</p></div>
        <div><h3>Song sinh số đạt mục tiêu</h3><div class="big" id="then">—</div><p class="delta" id="delta">&nbsp;</p></div></div>
      <div id="alert"></div>
      <section class="lead"><h2>Vì sao mô hình cho kết quả này</h2><div id="shap"><p class="note">Đang tính...</p></div>
        <p class="note" style="margin-top:8px">Thanh san hô kéo nguy cơ lên, thanh ngọc kéo xuống.</p></section>
      <section class="lead" id="rules-box" hidden><h2>Luật đã kích hoạt</h2><div id="rules"></div></section>
    </div>
    <div class="stack">
      <section class="lead"><header><h2>Hồ sơ</h2><div class="seg" id="mode"><button data-m="full" aria-pressed="true">Có xét nghiệm</button><button data-m="screening" aria-pressed="false">Chỉ sàng lọc</button></div></header><div id="form"></div></section>
      <section class="lead"><h2>Mục tiêu</h2><div id="goals"></div></section>
    </div></div>`;
  let mode = "full"; let profile = {}; let goals = {}; let seq = 0;

  const seed = () => {
    profile = { sex: "female" };
    NUM[mode].forEach(([k, , , , , v]) => { profile[k] = v; });
    FLAGS[mode].forEach(([k]) => { profile[k] = 0; });
    if (mode === "full") profile.smoking = "never";
    goals = Object.fromEntries(TARGETS[mode].map((k) => [k, profile[k]]));
  };
  const form = () => {
    $("#form", el).innerHTML = NUM[mode].map(([k, l, a, b, s]) => slider(k, l, a, b, s, profile[k], "data-k")).join("")
      + `<div class="row" style="margin-top:8px"><div class="seg" id="sex">${[["female", "Nữ"], ["male", "Nam"]].map(([v, t]) => `<button data-v="${v}" aria-pressed="${profile.sex === v}">${t}</button>`).join("")}</div>
         ${FLAGS[mode].map(([k, l]) => `<button class="chip" data-f="${k}" aria-pressed="${!!profile[k]}">${esc(l)}</button>`).join("")}</div>
      ${mode === "full" ? `<div class="seg" id="smoke" style="margin-top:8px">${SMOKING.map(([v, t]) => `<button data-v="${v}" aria-pressed="${profile.smoking === v}">${t}</button>`).join("")}</div>` : ""}`;
    $("#goals", el).innerHTML = NUM[mode].filter(([k]) => TARGETS[mode].includes(k)).map(([k, l, a, b, s]) => slider(k, l, a, b, s, goals[k], "data-g", "target")).join("");
  };
  const changes = () => Object.fromEntries(Object.entries(goals).filter(([k, v]) => v !== profile[k]));

  const compute = async () => {
    const mine = ++seq;
    try {
      const [p, w] = await Promise.all([api("/predict", { json: { profile, mode } }), api("/whatif", { json: { profile, changes: changes() } })]);
      if (mine !== seq) return;
      $("#alert", el).innerHTML = "";
      $("#now", el).textContent = pct(p.prob);
      $("#then", el).textContent = pct(w.after);
      $("#delta", el).textContent = `${signed(w.delta)} điểm phần trăm`;
      $("#meta", el).textContent = `${p.model}, ngưỡng ${pct(p.threshold)}${p.imputed?.length ? `, điền thiếu ${p.imputed.length} cột` : ""}`;
      $("#shap", el).innerHTML = shapHtml(p.shap, 10);
      $("#rules-box", el).hidden = !p.rules?.length;
      $("#rules", el).innerHTML = rulesHtml(p.rules);
    } catch (e) {
      if (!(e instanceof ApiError)) throw e;
      if (mine === seq) { $("#alert", el).innerHTML = pendingHtml(e); $("#shap", el).innerHTML = ""; }
    }
  };
  const later = debounce(compute, 120);

  el.addEventListener("input", (e) => {
    const i = e.target;
    if (i.dataset.k) { profile[i.dataset.k] = Number(i.value); } else if (i.dataset.g) { goals[i.dataset.g] = Number(i.value); } else return;
    i.previousElementSibling.textContent = num(Number(i.value), 1);
    later();
  });
  el.addEventListener("click", (e) => {
    const m = e.target.closest("[data-m]");
    if (m) {
      mode = m.dataset.m;
      $$("#mode button", el).forEach((b) => b.setAttribute("aria-pressed", b === m));
      seed(); form(); compute(); return;
    }
    const f = e.target.closest("[data-f]");
    if (f) { profile[f.dataset.f] = profile[f.dataset.f] ? 0 : 1; f.setAttribute("aria-pressed", !!profile[f.dataset.f]); compute(); return; }
    const v = e.target.closest("[data-v]");
    if (v) {
      const key = v.parentElement.id === "sex" ? "sex" : "smoking";
      profile[key] = v.dataset.v;
      $$("button", v.parentElement).forEach((b) => b.setAttribute("aria-pressed", b === v));
      compute();
    }
  });
  seed(); form(); await compute();
  const retry = setInterval(() => { if ($("#alert .pending", el)) compute(); }, 8000);
  return () => { seq += 1; clearInterval(retry); };
}
