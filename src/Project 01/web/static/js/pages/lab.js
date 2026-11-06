// Thí nghiệm: bảng và biểu đồ từ outputs/runs.jsonl, xem nguyên văn từng tệp outputs/metrics.
import { $, $$, api, ApiError, chart, esc, guarded, num, pendingHtml } from "../lib.js";

export const lead = "Mọi lượt huấn luyện và đánh giá đã ghi. Chọn một chỉ số để so các mô hình, hoặc mở từng tệp số liệu.";
const GROUPS = { data: "Dữ liệu", ml: "Mô hình bảng", dl: "Học sâu", retina: "Võng mạc", kb: "Kho tri thức", kg: "Đồ thị", rag: "Truy xuất", rl: "Học tăng cường", agent: "Tác tử", web: "Web" };
const MAX_BARS = 24;

const cell = (v) => (typeof v === "number" ? `<td class="n">${num(v, 4)}</td>` : `<td title="${esc(v)}">${esc(typeof v === "object" ? JSON.stringify(v) : v)}</td>`);

// Bảng hóa JSON theo hình dạng thật của tệp, không giả định tên khóa
function metricHtml(d) {
  const asRows = Array.isArray(d) ? d : (Object.values(d).every((v) => v && typeof v === "object" && !Array.isArray(v)) ? Object.entries(d).map(([k, v]) => ({ khóa: k, ...v })) : null);
  if (asRows?.length && typeof asRows[0] === "object") {
    const cols = [...new Set(asRows.slice(0, 40).flatMap(Object.keys))].slice(0, 14);
    return `<div class="tbl-wrap"><table class="tbl"><thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead><tbody>${asRows.slice(0, 200).map((r) => `<tr>${cols.map((c) => cell(r[c])).join("")}</tr>`).join("")}</tbody></table></div>`;
  }
  return `<pre class="cypher" style="max-height:420px">${esc(JSON.stringify(d, null, 1))}</pre>`;
}

export async function mount(el) {
  el.innerHTML = `<div class="stack">
    <section class="lead"><header><h2>So sánh mô hình</h2><div class="row"><label class="field">Chỉ số<select id="metric"></select></label><label class="field">Tập dữ liệu<select id="ds"></select></label></div></header><div id="cmp"><div class="chart" id="bars"></div></div></section>
    <section class="lead"><h2>Các lượt chạy</h2><div id="runs"></div></section>
    <section class="lead"><h2>Tệp số liệu</h2><div><div class="row" id="files"></div><div id="view-metric" style="margin-top:12px"></div></div></section></div>`;
  const ch = chart($("#bars", el), (rows, axis) => ({
    grid: { left: 180, right: 56, top: 8, bottom: 24 },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
    xAxis: { type: "value", ...axis, axisLine: { show: false } },
    yAxis: { type: "category", data: rows.map((r) => r.name), ...axis, inverse: true, splitLine: { show: false } },
    series: [{ type: "bar", data: rows.map((r) => r.value), barMaxWidth: 16, itemStyle: { borderRadius: [0, 4, 4, 0] }, label: { show: true, position: "right", formatter: (p) => num(p.value, 3), color: axis.axisLabel.color } }],
  }));
  let rows = [];
  const draw = () => {
    const m = $("#metric", el).value; const ds = $("#ds", el).value;
    const best = new Map();
    rows.filter((r) => (!ds || r.dataset === ds) && typeof r.metrics?.[m] === "number").forEach((r) => {
      const key = `${r.model} · ${r.dataset}`;
      best.set(key, r.metrics[m]); // lượt mới nhất của mỗi mô hình
    });
    const list = [...best].map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value).slice(0, MAX_BARS);
    if (!list.length) return;
    $("#bars", el).style.height = `${Math.max(180, list.length * 26 + 40)}px`;
    ch.update(list);
  };
  const stop = guarded($("#runs", el), async () => {
    rows = (await api("/runs")).rows;
    if (!rows.length) {
      $("#cmp", el).innerHTML = pendingHtml({ status: 503, message: "Chưa có lượt chạy nào trong outputs/runs.jsonl." });
      throw new ApiError("Các làn huấn luyện chưa ghi lượt chạy.", 503);
    }
    const metrics = [...new Set(rows.flatMap((r) => Object.keys(r.metrics || {})))].sort();
    $("#metric", el).innerHTML = metrics.map((m) => `<option ${/auc/i.test(m) ? "selected" : ""}>${esc(m)}</option>`).join("");
    $("#ds", el).innerHTML = `<option value="">Tất cả</option>${[...new Set(rows.map((r) => r.dataset))].map((d) => `<option>${esc(d)}</option>`).join("")}`;
    $("#runs", el).innerHTML = `<div class="tbl-wrap"><table class="tbl"><thead><tr>${["Thời điểm", "Nhóm", "Tập dữ liệu", "Mô hình", "Khung", "Thiết bị", "Giây"].map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows.slice(-300).reverse().map((r) => `<tr>${[r.ts, r.chapter, r.dataset, r.model, r.framework, r.device, r.seconds].map(cell).join("")}</tr>`).join("")}</tbody></table></div>`;
    draw();
  }, 15000);
  const showFile = async (name, btn) => {
    $$("#files button", el).forEach((b) => b.setAttribute("aria-pressed", b === btn));
    const box = $("#view-metric", el);
    try { box.innerHTML = metricHtml(await api(`/metrics/${encodeURIComponent(name)}`)); } catch (e) { if (!(e instanceof ApiError)) throw e; box.innerHTML = pendingHtml(e); }
  };
  const { names } = await api("/metrics");
  const label = (n) => `${GROUPS[n.split("_")[0]] ? `${GROUPS[n.split("_")[0]]}: ` : ""}${n.split("_").slice(1).join(" ") || n}`;
  $("#files", el).innerHTML = names.length ? names.map((n) => `<button class="chip" data-n="${esc(n)}" aria-pressed="false">${esc(label(n))}</button>`).join("") : '<p class="note">Đang chờ các làn ghi số liệu.</p>';
  $("#files", el).addEventListener("click", (e) => { const b = e.target.closest("[data-n]"); if (b) showFile(b.dataset.n, b); });
  $("#metric", el).addEventListener("change", draw);
  $("#ds", el).addEventListener("change", draw);
  return () => { stop(); ch.dispose(); };
}
