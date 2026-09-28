// Giao diện hai phân hệ: gọi REST API của app.py, vẽ bằng Chart.js, nền tối/sáng theo data-theme.
const $ = id => document.getElementById(id);
const FW_LABEL = {torch: "PyTorch", keras: "Keras"};
const RISK_CLASS = {"cao": "cao", "trung bình": "tb", "thấp": "thap"};
const fmt = (v, d = 2) => Number(v).toLocaleString("vi-VN", {minimumFractionDigits: d, maximumFractionDigits: d});
const state = {meta: null, stock: {fw: "torch", model: null, n: 100, hist: null, pred: null},
               churn: {fw: "torch", model: null, source: "churn", seq: null}};
const charts = {};

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(t.timer);
  t.timer = setTimeout(() => t.classList.remove("show"), 3500);
}

// Chạy fn trong lúc nút hiện vòng quay; lỗi API hiện thành thông báo nổi thay vì im lặng.
async function busy(btn, fn) {
  btn.classList.add("busy");
  try { return await fn(); } catch (e) { toast("Lỗi: " + e.message); } finally { btn.classList.remove("busy"); }
}

// Số chạy từ giá trị cũ tới giá trị mới trong 0,7 s.
function countUp(el, to, digits, suffix = "", prefix = "") {
  const from = parseFloat(el.dataset.v || 0), t0 = performance.now();
  el.dataset.v = to;
  const step = now => {
    const k = Math.min(1, (now - t0) / 700), e = 1 - Math.pow(1 - k, 3);
    el.textContent = prefix + fmt(from + (to - from) * e, digits) + suffix;
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function css(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function chartDefaults() {
  Chart.defaults.font.family = "'Be Vietnam Pro', system-ui, sans-serif";
  Chart.defaults.color = css("--muted");
  Chart.defaults.borderColor = css("--line");
}

function segmented(group, attr, onPick) {
  group.querySelectorAll("button").forEach(b => b.onclick = () => {
    group.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
    onPick(b.dataset[attr]);
  });
}

// Danh sách bốn kiến trúc dạng thẻ: số tham số, chỉ số test, nhãn "tốt nhất trên val".
function renderModels(box, dataset, fw, current, onPick) {
  const {models, cards, best} = state.meta;
  const bestCell = best[`${dataset}_${fw}`];
  box.innerHTML = state.meta.cells.map(cell => [cell, models[cell]]).map(([cell, label]) => {
    const c = cards[`${dataset}_${fw}_${cell}`];
    const metrics = dataset === "amzn"
      ? `RMSE ${fmt(c.rmse, 3)} · đúng hướng ${fmt(100 * c.dir_acc, 1)}%`
      : `F1 ${fmt(c.f1, 3)} · ROC-AUC ${fmt(c.roc_auc, 3)}`;
    return `<button class="model ${cell === current ? "on" : ""}" data-cell="${cell}">
      <span class="name">${label}${cell === bestCell ? '<span class="tag">tốt nhất val</span>' : ""}</span>
      <span class="params">${c.params.toLocaleString("vi-VN")} tham số</span>
      <span class="metrics">${metrics} · ${fmt(c.latency_ms, 2)} ms</span></button>`;
  }).join("");
  box.querySelectorAll(".model").forEach(b => b.onclick = () => {
    box.querySelectorAll(".model").forEach(x => x.classList.toggle("on", x === b));
    onPick(b.dataset.cell);
  });
}

// Bảng so sánh chỉ đúng với đầu vào lúc bấm; đổi framework hay chuỗi thì trả về dòng gợi ý ban đầu.
function resetCompare(id) {
  const box = $(id);
  box.dataset.hint ??= box.innerHTML;
  box.innerHTML = box.dataset.hint;
}

// ================================================================ phân hệ 1: giá AMZN
async function loadHistory() {
  const s = state.stock;
  s.hist = await api(`/api/stock/history?n=${s.n}`);
  s.pred = null;
  drawStock();
  const L = s.hist.last, cls = L.change_pct >= 0 ? "up" : "down";
  $("s-stats").innerHTML = `
    <div class="stat"><span>Phiên cuối</span><b>${s.hist.dates.at(-1)}</b></div>
    <div class="stat"><span>Đóng cửa</span><b class="${cls}">$${fmt(L.close)} (${L.change_pct >= 0 ? "+" : ""}${fmt(L.change_pct)}%)</b></div>
    <div class="stat"><span>Cao / Thấp</span><b>${fmt(L.high)} / ${fmt(L.low)}</b></div>
    <div class="stat"><span>Khối lượng</span><b>${(L.volume / 1e6).toLocaleString("vi-VN", {maximumFractionDigits: 1})} triệu</b></div>`;
}

function drawStock() {
  const {hist, pred} = state.stock, n = hist.close.length;
  const labels = pred ? [...hist.dates, "Phiên kế tiếp"] : hist.dates;
  const ctx = $("s-chart").getContext("2d"), fill = ctx.createLinearGradient(0, 0, 0, 330);
  fill.addColorStop(0, "rgba(56,189,248,.30)");
  fill.addColorStop(1, "rgba(56,189,248,0)");
  const datasets = [
    {label: "Giá đóng cửa (USD)", data: hist.close, borderColor: css("--blue"), backgroundColor: fill, fill: true,
     borderWidth: 2.2, pointRadius: 0, tension: .25},
    {label: "Đường MA-20", data: hist.ma20, borderColor: css("--amber"), borderDash: [6, 4], borderWidth: 1.6,
     pointRadius: 0, tension: .25}];
  if (pred) {
    datasets.push({label: `Dự báo ${pred.model_label}`, data: [...Array(n - 1).fill(null), pred.last_close, pred.pred_close],
                   borderColor: css("--red"), backgroundColor: css("--red"), borderDash: [3, 3], borderWidth: 2,
                   pointRadius: [...Array(n).fill(0), 6], pointHoverRadius: 8});
  }
  charts.stock?.destroy();
  charts.stock = new Chart(ctx, {type: "line", data: {labels, datasets}, options: {
    maintainAspectRatio: false, interaction: {mode: "index", intersect: false},
    plugins: {legend: {labels: {usePointStyle: true, boxWidth: 8, boxHeight: 8}},
              tooltip: {callbacks: {label: c => c.raw == null ? "" : `${c.dataset.label}: $${fmt(c.raw)}`}}},
    scales: {x: {ticks: {maxTicksLimit: 8, maxRotation: 0}, grid: {display: false}},
             y: {ticks: {callback: v => "$" + v}}}}});
  charts.rsi?.destroy();
  charts.rsi = new Chart($("s-rsi"), {type: "line", data: {labels: hist.dates, datasets: [
    {data: hist.rsi14, borderColor: css("--violet"), borderWidth: 1.6, pointRadius: 0, tension: .25},
    {data: hist.dates.map(() => 70), borderColor: "rgba(244,63,94,.6)", borderDash: [4, 4], borderWidth: 1, pointRadius: 0},
    {data: hist.dates.map(() => 30), borderColor: "rgba(16,185,129,.6)", borderDash: [4, 4], borderWidth: 1, pointRadius: 0}]},
    options: {maintainAspectRatio: false, plugins: {legend: {display: false}, tooltip: {enabled: false}},
              scales: {x: {display: false}, y: {min: 0, max: 100, ticks: {stepSize: 50}}}}});
}

function stockResult(r) {
  const up = r.change_pct >= 0, rsiPos = Math.max(0, Math.min(100, r.rsi14));
  return `<div class="res-top">Kết quả dự báo · ${r.model_label} · ${FW_LABEL[r.framework]}
      <span class="pill ${up ? "up" : "down"}">${up ? "▲ TĂNG" : "▼ GIẢM"}</span></div>
    <div class="big" id="s-big">$0</div>
    <div class="res-sub">Biến động ${up ? "+" : ""}${fmt(r.pred_close - r.last_close)} USD
      (<b class="${up ? "up" : "down"}">${up ? "+" : ""}${fmt(r.change_pct, 3)}%</b>) so với phiên ${r.last_date}</div>
    <div class="res-grid">
      <div><span>Giá phiên gần nhất</span><b>$${fmt(r.last_close)}</b></div>
      <div><span>Độ trễ suy luận</span><b>${fmt(r.latency_ms, 2)} ms</b></div>
    </div>
    <div class="res-grid" style="grid-template-columns:1fr">
      <div><span>RSI-14: <b>${fmt(r.rsi14)}</b> · vùng ${r.rsi_zone}</span>
        <div class="gauge"><i style="left:0%" data-left="${rsiPos}"></i></div>
        <div class="gauge-l"><span>0 · quá bán</span><span>30</span><span>70</span><span>quá mua · 100</span></div></div>
    </div>`;
}

async function initStock() {
  const s = state.stock;
  s.model = state.meta.best[`amzn_${s.fw}`];
  const pickModels = () => renderModels($("s-models"), "amzn", s.fw, s.model, cell => { s.model = cell; });
  pickModels();
  segmented($("s-fw"), "fw", fw => { s.fw = fw; s.model = state.meta.best[`amzn_${fw}`]; pickModels(); resetCompare("s-compare-out"); });
  segmented($("s-range"), "n", n => { s.n = Number(n); loadHistory().catch(e => toast(e.message)); });
  await loadHistory();
  $("s-run").onclick = () => busy($("s-run"), async () => {
    const r = await api("/api/stock/predict", {framework: s.fw, model: s.model});
    s.pred = r;
    $("s-out").innerHTML = stockResult(r);
    countUp($("s-big"), r.pred_close, 2, "", "$");
    requestAnimationFrame(() => $("s-out").querySelectorAll(".gauge i").forEach(i => { i.style.left = i.dataset.left + "%"; }));
    drawStock();
  });
  $("s-compare").onclick = () => busy($("s-compare"), async () => {
    const cells = state.meta.cells;
    const rows = [];
    for (const c of cells) rows.push(await api("/api/stock/predict", {framework: s.fw, model: c}));   // tuần tự: độ trễ không bị nhiễu
    const naive = state.meta.naive_rmse, cards = state.meta.cards;
    const maxRmse = Math.max(naive, ...cells.map(c => cards[`amzn_${s.fw}_${c}`].rmse));
    $("s-compare-out").innerHTML = `<table><thead><tr><th>Kiến trúc</th><th>Giá dự báo</th><th>Thay đổi</th>
      <th>Độ trễ</th><th>RMSE test (vạch = mốc ngây thơ ${fmt(naive, 3)})</th></tr></thead><tbody>` +
      rows.map((r, i) => {
        const c = cards[`amzn_${s.fw}_${r.model}`], up = r.change_pct >= 0;
        return `<tr style="animation-delay:${i * 70}ms"><td><b>${r.model_label}</b></td>
          <td class="num">$${fmt(r.pred_close)}</td>
          <td class="num ${up ? "up" : "down"}">${up ? "+" : ""}${fmt(r.change_pct, 3)}%</td>
          <td class="num">${fmt(r.latency_ms, 2)} ms</td>
          <td><div class="mini"><div class="bar"><span style="width:${100 * c.rmse / maxRmse}%;background:${css("--indigo")}"></span>
            <i style="left:${100 * naive / maxRmse}%"></i></div><span class="num">${fmt(c.rmse, 3)}</span></div></td></tr>`;
      }).join("") + `</tbody></table>`;
  });
}

// ================================================================ phân hệ 2: churn KKBox
const F = name => state.meta.features.indexOf(name);

// Chuỗi 31 × 8 giả định từ bốn thanh trượt, cùng dạng log1p với dữ liệu huấn luyện (ngày không nghe để 0).
function simulatedSequence() {
  const minutes = +$("r-min").value, songs = +$("r-song").value, perWeek = +$("r-week").value, stop = +$("r-stop").value;
  const seq = [];
  for (let d = 1; d <= 31; d++) {
    const row = Array(8).fill(0);
    if ((d - 1) % 7 < perWeek && d < stop && minutes > 0) {
      const jitter = 0.75 + 0.5 * Math.abs(Math.sin(d * 12.9898) * 43758.5453 % 1);
      const plays = songs * 1.3 * jitter;
      row[F("num_25")] = Math.log1p(Math.round(plays * 0.15));
      row[F("num_50")] = Math.log1p(Math.round(plays * 0.05));
      row[F("num_75")] = Math.log1p(Math.round(plays * 0.04));
      row[F("num_985")] = Math.log1p(Math.round(plays * 0.04));
      row[F("num_100")] = Math.log1p(Math.round(plays * 0.72));
      row[F("num_unq")] = Math.log1p(Math.round(songs * jitter));
      row[F("total_secs")] = Math.log1p(minutes * 60 * jitter);
      row[F("active")] = 1;
    }
    seq.push(row);
  }
  return seq;
}

function currentSequence() {
  const c = state.churn;
  return c.source === "sim" ? simulatedSequence() : state.meta.presets[c.source].sequence;
}

function drawChurn(seq) {
  const minutes = seq.map(d => Math.expm1(d[F("total_secs")]) / 60);
  const songs = seq.map(d => Math.expm1(d[F("num_unq")]));
  const active = seq.filter(d => d[F("active")] > 0).length, hours = minutes.reduce((a, b) => a + b, 0) / 60;
  const trend = seq.slice(-7).reduce((a, d) => a + d[F("total_secs")], 0) / 7 - seq.slice(0, 7).reduce((a, d) => a + d[F("total_secs")], 0) / 7;
  const lastDay = seq.map(d => d[F("active")]).lastIndexOf(1) + 1;
  $("c-stats").innerHTML = `
    <div class="stat"><span>Ngày có nghe</span><b>${active} / 31</b></div>
    <div class="stat"><span>Tổng giờ nghe</span><b>${fmt(hours, 1)} giờ</b></div>
    <div class="stat"><span>Ngày nghe cuối</span><b>${lastDay ? "ngày " + lastDay : "không có"}</b></div>
    <div class="stat"><span>Xu hướng nghe</span><b class="${trend >= 0 ? "up" : "down"}">${trend >= 0 ? "+" : ""}${fmt(trend, 2)}</b></div>`;
  const data = {labels: seq.map((_, i) => `Ngày ${i + 1}`), datasets: [
    {type: "bar", label: "Phút nghe", data: minutes, yAxisID: "y", borderRadius: 4, maxBarThickness: 16,
     backgroundColor: minutes.map(m => m > 0 ? "rgba(16,185,129,.75)" : "rgba(148,163,184,.25)")},
    {type: "line", label: "Số bài khác nhau", data: songs, yAxisID: "y1", borderColor: css("--blue"),
     backgroundColor: css("--blue"), borderWidth: 2, pointRadius: 2.5, tension: .3, borderDash: [5, 3]}]};
  if (charts.churn) {
    charts.churn.data = data;
    charts.churn.update();
    return;
  }
  charts.churn = new Chart($("c-chart"), {data, options: {
    maintainAspectRatio: false, interaction: {mode: "index", intersect: false},
    plugins: {legend: {labels: {usePointStyle: true, boxWidth: 8, boxHeight: 8}},
              tooltip: {callbacks: {label: c => `${c.dataset.label}: ${fmt(c.raw, c.datasetIndex ? 0 : 1)}`}}},
    scales: {x: {grid: {display: false}, ticks: {maxTicksLimit: 11, maxRotation: 0, callback: (v, i) => i + 1}},
             y: {beginAtZero: true, title: {display: true, text: "phút / ngày"}},
             y1: {beginAtZero: true, position: "right", grid: {display: false}, title: {display: true, text: "bài / ngày"}}}}});
}

function churnResult(r) {
  const cls = RISK_CLASS[r.risk];
  return `<div class="res-top">Xác suất rời bỏ · ${r.model_label} · ${FW_LABEL[r.framework]}
      <span class="pill ${cls}">Mức: ${r.risk.toUpperCase()}</span></div>
    <div class="big ${cls}" id="c-big">0%</div>
    <div class="res-sub">khả năng không gia hạn gói sau tháng 3/2017</div>
    <div class="bar"><span style="background:${css(cls === "cao" ? "--red" : cls === "tb" ? "--amber" : "--green")}"
      data-w="${100 * r.prob}"></span><i style="left:${100 * r.threshold}%" title="ngưỡng"></i></div>
    <div class="gauge-l"><span>0%</span><span>ngưỡng val ${fmt(100 * r.threshold, 1)}%</span><span>100%</span></div>
    <div class="advice"><svg viewBox="0 0 24 24"><path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9V16h7v-2.1A6 6 0 0 0 12 3z"/></svg>
      <div><b>Khuyến nghị can thiệp:</b> ${r.advice}</div></div>
    <div class="res-grid">
      <div><span>Xu hướng nghe</span><b class="${r.trend >= 0 ? "up" : "down"}">${r.trend >= 0 ? "+" : ""}${fmt(r.trend, 3)}</b></div>
      <div><span>Độ trễ suy luận</span><b>${fmt(r.latency_ms, 2)} ms</b></div>
    </div>`;
}

async function runChurn() {
  const c = state.churn;
  const body = c.source === "sim" ? {sequence: simulatedSequence()} : {preset: c.source};
  const r = await api("/api/churn/predict", {...body, framework: c.fw, model: c.model});
  $("c-out").innerHTML = churnResult(r);
  countUp($("c-big"), 100 * r.prob, 1, "%");
  requestAnimationFrame(() => $("c-out").querySelectorAll(".bar span").forEach(s => { s.style.width = s.dataset.w + "%"; }));
}

async function initChurn() {
  const c = state.churn, meta = state.meta;
  c.model = meta.best[`kkbox_${c.fw}`];
  const pickModels = () => renderModels($("c-models"), "kkbox", c.fw, c.model, cell => { c.model = cell; });
  pickModels();
  segmented($("c-fw"), "fw", fw => { c.fw = fw; c.model = meta.best[`kkbox_${fw}`]; pickModels(); resetCompare("c-compare-out"); });
  const sources = {...Object.fromEntries(Object.entries(meta.presets).map(([k, v]) => [k, v.label])), sim: "Tự mô phỏng hành vi"};
  $("c-presets").innerHTML = Object.entries(sources).map(([k, v]) =>
    `<button class="chip chip-${k} ${k === c.source ? "on" : ""}" data-src="${k}">${v}</button>`).join("");
  $("c-presets").querySelectorAll(".chip").forEach(b => b.onclick = () => {
    $("c-presets").querySelectorAll(".chip").forEach(x => x.classList.toggle("on", x === b));
    c.source = b.dataset.src;
    resetCompare("c-compare-out");
    $("c-sim").hidden = c.source !== "sim";
    $("c-source").textContent = c.source === "sim" ? "Chuỗi giả định từ thanh trượt" : "Người dùng thật trong tập test";
    drawChurn(currentSequence());
  });
  let timer;
  const outputs = {min: v => v + " phút", song: v => v + " bài", week: v => v + " ngày",
                   stop: v => v >= 32 ? "không ngừng" : "ngày " + v};
  Object.entries(outputs).forEach(([k, show]) => {
    const input = $("r-" + k);
    $("o-" + k).textContent = show(+input.value);
    input.oninput = () => {
      $("o-" + k).textContent = show(+input.value);
      drawChurn(simulatedSequence());
      resetCompare("c-compare-out");
      clearTimeout(timer);
      timer = setTimeout(() => busy($("c-run"), runChurn), 300);    // dự báo lại khi người dùng dừng kéo
    };
  });
  drawChurn(currentSequence());
  $("c-run").onclick = () => busy($("c-run"), runChurn);
  $("c-compare").onclick = () => busy($("c-compare"), async () => {
    const body = c.source === "sim" ? {sequence: simulatedSequence()} : {preset: c.source};
    const rows = [];
    for (const m of meta.cells) rows.push(await api("/api/churn/predict", {...body, framework: c.fw, model: m}));
    $("c-compare-out").innerHTML = `<table><thead><tr><th>Kiến trúc</th><th>Xác suất rời bỏ (vạch = ngưỡng)</th>
      <th>Mức rủi ro</th><th>F1 test</th><th>Độ trễ</th></tr></thead><tbody>` +
      rows.map((r, i) => {
        const cls = RISK_CLASS[r.risk], card = meta.cards[`kkbox_${c.fw}_${r.model}`];
        const color = css(cls === "cao" ? "--red" : cls === "tb" ? "--amber" : "--green");
        return `<tr style="animation-delay:${i * 70}ms"><td><b>${r.model_label}</b></td>
          <td><div class="mini"><div class="bar"><span style="width:${100 * r.prob}%;background:${color}"></span>
            <i style="left:${100 * r.threshold}%"></i></div><span class="num">${fmt(100 * r.prob, 1)}%</span></div></td>
          <td><span class="pill ${cls}">${r.risk}</span></td>
          <td class="num">${fmt(card.f1, 3)}</td><td class="num">${fmt(r.latency_ms, 2)} ms</td></tr>`;
      }).join("") + `</tbody></table>`;
  });
}

// ================================================================ khung chung
function initShell() {
  try { const saved = localStorage.getItem("a06-theme"); if (saved) document.documentElement.dataset.theme = saved; } catch (e) { /* trình duyệt chặn storage */ }
  $("theme-toggle").onclick = () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("a06-theme", next); } catch (e) { /* bỏ qua */ }
    chartDefaults();
    if (state.stock.hist) drawStock();
    if (charts.churn) { charts.churn.destroy(); charts.churn = null; drawChurn(currentSequence()); }
  };
  document.querySelectorAll(".tab-btn").forEach(b => b.onclick = () => {
    document.querySelectorAll(".tab-btn").forEach(x => { x.classList.toggle("active", x === b); x.setAttribute("aria-selected", x === b); });
    document.querySelectorAll(".panel").forEach(p => p.classList.toggle("active", p.id === b.dataset.tab));
    Object.values(charts).forEach(ch => ch?.resize());
  });
}

async function main() {
  initShell();
  chartDefaults();
  state.meta = await api("/api/meta");
  const st = state.meta.stats, lat = Object.entries(state.meta.cards).filter(([k]) => k.includes("_torch_")).map(([, v]) => v.latency_ms).sort((a, b) => a - b);
  document.querySelector('[data-kpi="amzn"]').textContent = st.amzn.n.toLocaleString("vi-VN");
  document.querySelector('[data-kpi="kkbox"]').textContent = st.kkbox.n_sample.toLocaleString("vi-VN");
  document.querySelector('[data-kpi="lat"]').textContent = fmt((lat[3] + lat[4]) / 2, 2) + " ms";
  await Promise.all([initStock(), initChurn()]);
}

main().catch(e => toast("Không tải được dữ liệu: " + e.message));
