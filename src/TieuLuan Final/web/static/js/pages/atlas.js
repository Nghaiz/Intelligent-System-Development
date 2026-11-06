// Data Atlas: mỗi dataset một thẻ gồm nguồn, kích cỡ, phân bố và vài mẫu đọc từ chN_datasets.json.
import { $, api, chart, empty, esc, fmt } from "../lib.js";

function sample(rows) {
  if (!rows?.length) return "";
  if (typeof rows[0] !== "object") return `<div class="ds-sample">${rows.slice(0, 4).map((r) => `<div>${esc(r)}</div>`).join("")}</div>`;
  const keys = Object.keys(rows[0]).slice(0, 6);
  return `<div class="ds-sample tbl"><table><tr>${keys.map((k) => `<th>${esc(k)}</th>`).join("")}</tr>${rows.slice(0, 4).map((r) => `<tr>${keys.map((k) => `<td>${esc(String(r[k] ?? "").slice(0, 48))}</td>`).join("")}</tr>`).join("")}</table></div>`;
}

export default async function atlas(view) {
  const { datasets } = await api("/datasets");
  view.innerHTML = `<h2>Data Atlas</h2><p class="lead">${datasets.length} bộ dữ liệu từ nhiều nguồn khác nhau, đúng số liệu ghi trong báo cáo.</p>
    <div class="row"><select id="ch"><option value="">Mọi chương</option>${[...new Set(datasets.map((d) => d.chapter))].map((c) => `<option>${c}</option>`).join("")}</select></div><div class="grid" id="cards"></div>`;
  const draw = () => {
    const ch = $("#ch").value, list = datasets.filter((d) => !ch || d.chapter === +ch);
    $("#cards").innerHTML = list.length ? list.map((d, i) => `<article class="card reveal"><span class="chip">Ch${d.chapter}</span> <span class="chip">${esc(d.task ?? "")}</span>
      <h3 style="margin-top:.4em">${esc(d.name)}</h3><p>${fmt(d.n_samples)} mẫu · ${fmt(d.n_features)} đặc trưng · ${fmt(d.size_mb, 1)} MB</p>
      ${d.url ? `<a href="${esc(d.url)}" rel="noopener" target="_blank">${esc(d.source ?? "Nguồn")}</a>` : esc(d.source ?? "")}
      ${d.distribution ? `<div class="chart" style="height:170px" id="d-${i}"></div>` : ""}${sample(d.sample_rows)}</article>`).join("") : empty("Chưa có dataset nào.");
    list.forEach((d, i) => {
      if (!d.distribution) return;
      const { labels, values } = d.distribution;
      chart($(`#d-${i}`), { series: [{ type: "pie", radius: ["42%", "72%"], data: labels.slice(0, 10).map((l, j) => ({ name: String(l), value: values[j] })), label: { show: labels.length <= 6, color: "inherit", fontSize: 11 } }] });
    });
  };
  $("#ch").onchange = draw; draw();
}
