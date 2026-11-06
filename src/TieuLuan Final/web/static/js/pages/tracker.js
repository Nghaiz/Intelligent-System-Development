// Experiment Tracker: bảng so sánh có lọc và sắp xếp, biểu đồ đánh đổi chất lượng và thời gian trên dữ liệu thật.
import { $, api, axis, chart, empty, esc, fmt } from "../lib.js";

export default async function tracker(view) {
  const [{ tables }, { rows: runs }] = await Promise.all([api("/compare"), api("/runs")]);
  view.innerHTML = `<h2>Experiment Tracker</h2><p class="lead">${runs.length} lượt chạy đã ghi. Chọn chương và dataset để so sánh.</p>
    <div class="row"><select id="ch">${tables.map((t, i) => `<option value="${i}">Chương ${t.chapter}</option>`).join("")}</select><select id="ds"></select><input id="f" placeholder="Lọc mô hình"></div>
    <div class="two"><div class="card"><h3>Xếp hạng</h3><div class="chart tall" id="bar"></div></div><div class="card"><h3>Chất lượng và thời gian</h3><div class="chart tall" id="sc"></div></div></div>
    <div class="card" style="margin-top:1rem"><div class="tbl" id="tb"></div></div>`;
  if (!tables.length) { $("#tb").innerHTML = empty("Chưa có bảng so sánh nào."); return; }
  let sortKey = null, dir = -1;
  const cur = () => tables[+$("#ch").value];
  const setDs = () => { const ds = [...new Set(cur().rows.map((r) => r.dataset))]; $("#ds").innerHTML = ds.map((d) => `<option>${esc(d)}</option>`).join(""); };
  const draw = () => {
    const t = cur(), k = $("#f").value.toLowerCase();
    const rows = t.rows.filter((r) => r.dataset === $("#ds").value && String(r.model).toLowerCase().includes(k));
    const keys = [...new Set(rows.flatMap(Object.keys))].filter((c) => c !== "dataset");
    const m = t.primary_by_dataset?.[$("#ds").value] ?? t.primary ?? keys.find((c) => typeof rows[0]?.[c] === "number");
    const by = sortKey ?? m, sorted = [...rows].sort((a, b) => dir * ((a[by] ?? -Infinity) > (b[by] ?? -Infinity) ? 1 : -1));
    $("#tb").innerHTML = `<table><tr>${keys.map((c) => `<th data-k="${esc(c)}">${esc(c)}${c === by ? (dir < 0 ? " ▾" : " ▴") : ""}</th>`).join("")}</tr>${sorted.map((r) => `<tr>${keys.map((c) => `<td class="${typeof r[c] === "number" ? "num" : ""}">${fmt(r[c], 4)}</td>`).join("")}</tr>`).join("")}</table>`;
    const lab = (r) => `${r.model}${r.framework ? " · " + r.framework : ""}`, top = [...rows].sort((a, b) => (/rmse|mae|mape|regret|loss/i.test(m) ? 1 : -1) * ((a[m] ?? 0) - (b[m] ?? 0))).slice(0, 14).reverse();
    if (!top.length) return;
    chart($("#bar"), { grid: { left: 170, right: 24, top: 8, bottom: 28 }, xAxis: axis({ type: "value", name: m, scale: true }), yAxis: axis({ type: "category", data: top.map(lab) }),
      series: [{ type: "bar", data: top.map((r) => r[m]), itemStyle: { borderRadius: [0, 5, 5, 0] } }] });
    chart($("#sc"), { grid: { left: 58, right: 20, top: 16, bottom: 44 }, tooltip: { formatter: (p) => `${esc(p.data[2])}<br>${m}: ${fmt(p.data[1], 4)}<br>${fmt(p.data[0], 2)} giây` },
      xAxis: axis({ type: "log", name: "giây", nameLocation: "middle", nameGap: 26 }), yAxis: axis({ type: "value", name: m, scale: true }),
      series: [{ type: "scatter", symbolSize: 13, data: rows.filter((r) => r.seconds > 0 && typeof r[m] === "number").map((r) => [r.seconds, r[m], lab(r)]) }] });
  };
  const again = () => { [$("#bar"), $("#sc")].forEach((el) => echarts.getInstanceByDom(el)?.dispose()); draw(); };
  $("#ch").onchange = () => { setDs(); sortKey = null; again(); };
  $("#ds").onchange = $("#f").oninput = again;
  $("#tb").onclick = (e) => { const k = e.target.dataset?.k; if (k) { dir = sortKey === k ? -dir : -1; sortKey = k; again(); } };
  setDs(); draw();
}
