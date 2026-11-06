// Tổng quan: "Vòng tiến trình đồng tâm", mỗi vòng là một chương, độ dài cung theo số lượt chạy thật.
import { $, api, axis, chart, countUp, css, empty, palette } from "../lib.js";

const TITLES = { 2: "Học máy", 3: "CNN", 4: "RNN", 5: "KG, RAG", 6: "Đa tác tử", 7: "GNN", 8: "Tác tử RL" };

function rings(canvas, chapters) {
  const dpr = devicePixelRatio || 1, S = canvas.clientWidth;
  canvas.width = canvas.height = S * dpr;
  const g = canvas.getContext("2d"); g.scale(dpr, dpr);
  const cols = palette(), max = Math.max(...chapters.map((c) => Math.log1p(c.runs)), 1), t0 = performance.now();
  const step = S / 2 / (chapters.length + 2);
  const draw = (t) => {
    const k = Math.min(1, (t - t0) / 1100);
    g.clearRect(0, 0, S, S);
    chapters.forEach((c, i) => {
      const r = step * (i + 2);
      g.lineWidth = step * .62; g.lineCap = "round";
      g.strokeStyle = css("--panel2"); g.beginPath(); g.arc(S / 2, S / 2, r, 0, 7); g.stroke();
      g.strokeStyle = cols[i % 5]; g.beginPath();
      g.arc(S / 2, S / 2, r, -Math.PI / 2, -Math.PI / 2 + 6.28 * k * Math.max(.02, Math.log1p(c.runs) / max)); g.stroke();
      g.fillStyle = css("--fg"); g.font = "12px BVP"; g.textAlign = "right";
      g.fillText(`Ch${c.chapter}`, S / 2 - 6, S / 2 - r + 4);
    });
    if (k < 1) requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
}

export default async function home(view) {
  const o = await api("/overview");
  view.innerHTML = `<section class="hero reveal"><div>
      <h1>Một phòng thí nghiệm, tám chương, mọi con số đều đo thật.</h1>
      <p class="lead">Đồng Tâm AI Lab gom kết quả của cả tiểu luận vào một nơi: đồ thị tri thức Neo4j, trợ lý GraphRAG, đấu trường tác tử, mô hình chạy trực tiếp và nhật ký huấn luyện.</p>
      <div class="row"><a href="/graph" data-nav><button class="primary">Mở đồ thị tri thức</button></a><a href="/tracker" data-nav><button>Xem thực nghiệm</button></a></div>
    </div><canvas id="rings" role="img" aria-label="Số lượt chạy theo chương"></canvas></section>
    <section class="kpis" id="kpis"></section>
    <section class="two"><div class="card"><h3>Lượt chạy theo chương</h3><div class="chart" id="c-runs"></div></div>
    <div class="card"><h3>Tiến độ từng chương</h3><div class="tbl" id="t-ch"></div></div></section>`;
  const cols = palette();
  const kpis = [["runs", "lượt huấn luyện và đánh giá"], ["models", "mô hình đã so sánh"], ["datasets", "bộ dữ liệu"], ["samples", "mẫu dữ liệu"]];
  $("#kpis").innerHTML = kpis.map(([k, t], i) => `<div class="kpi" style="--c1:${cols[i]}"><b data-to="${o[k]}">0</b><span>${t}</span></div>`).join("")
    + (o.graph ? `<div class="kpi" style="--c1:${cols[4]}"><b data-to="${o.graph.nodes}">0</b><span>nút trong Neo4j</span></div>` : "");
  view.querySelectorAll("[data-to]").forEach((el) => countUp(el, +el.dataset.to));
  if (!o.chapters.length) { $("#c-runs").innerHTML = empty("Chưa có lượt chạy nào."); return; }
  rings($("#rings"), o.chapters);
  chart($("#c-runs"), { grid: { left: 44, right: 12, top: 12, bottom: 28 },
    xAxis: axis({ type: "category", data: o.chapters.map((c) => TITLES[c.chapter] ?? `Ch${c.chapter}`) }), yAxis: axis({ type: "value" }),
    series: [{ type: "bar", data: o.chapters.map((c, i) => ({ value: c.runs, itemStyle: { color: cols[i % 5], borderRadius: [5, 5, 0, 0] } })) }] });
  $("#t-ch").innerHTML = `<table><tr><th>Chương</th><th>Lượt chạy</th><th>Dataset</th><th>Mô hình</th></tr>${o.chapters.map((c) =>
    `<tr><td>${c.chapter}. ${TITLES[c.chapter] ?? ""}</td><td class="num">${c.runs}</td><td class="num">${c.datasets}</td><td class="num">${c.models}</td></tr>`).join("")}</table>`;
}
