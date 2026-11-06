// Dữ liệu: duyệt bảng SQLite, phân trang, thống kê cột bằng SQL, biểu đồ phân bố ECharts.
import { $, $$, api, ApiError, chart, esc, guarded, num, pct, pendingHtml } from "../lib.js";

export const lead = "Năm bộ dữ liệu và các bảng vận hành trong SQLite, đọc qua kết nối chỉ đọc. Bấm tên cột để xem phân bố.";
const LIMIT = 50;

export async function mount(el) {
  el.innerHTML = `<div id="root"></div>`;
  const root = $("#root", el); let ch = null; let state = null;

  const drawTable = (p) => {
    const cols = p.columns.map((c) => c.name);
    $("#tbl", el).innerHTML = `<table class="tbl"><thead><tr>${cols.map((c) => `<th data-col="${esc(c)}" aria-sort="${c === state.col ? "ascending" : "none"}">${esc(c)}</th>`).join("")}</tr></thead><tbody>${p.rows.map((r) => `<tr>${cols.map((c) => `<td class="${typeof r[c] === "number" ? "n" : ""}" title="${esc(r[c])}">${typeof r[c] === "number" ? esc(String(r[c]).replace(".", ",")) : esc(r[c])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
    const from = p.total ? p.offset + 1 : 0;
    $("#pos", el).textContent = `${num(from)} đến ${num(Math.min(p.offset + p.limit, p.total))} trên ${num(p.total)} dòng`;
    $("#prev", el).disabled = p.offset === 0;
    $("#next", el).disabled = p.offset + p.limit >= p.total;
  };
  const drawColumn = async () => {
    const { col, table } = state;
    $("#col-title", el).textContent = col || "";
    $("#col-meta", el).textContent = "Đang tính thống kê...";
    const { column: s } = await api(`/db/${table}/stats${col ? `?col=${encodeURIComponent(col)}` : ""}`);
    if (table !== state.table || (col && col !== state.col)) return;
    state.col = s.name;
    $("#col-title", el).textContent = s.name;
    $$("th[data-col]", el).forEach((th) => th.setAttribute("aria-sort", th.dataset.col === s.name ? "ascending" : "none"));
    $("#col-meta", el).textContent = `${s.type || "không kiểu"} · ${num(s.distinct)} giá trị khác nhau · thiếu ${pct(s.null_pct)}${s.kind === "num" ? ` · nhỏ nhất ${num(s.min, 2)} · lớn nhất ${num(s.max, 2)} · trung bình ${num(s.mean, 2)}` : ""}`;
    ch.update(s);
  };
  const buildChart = (s, axis) => {
    const hist = s.kind === "num" && s.hist;
    const cats = hist ? s.hist.map((b) => num(b.x0, 1)) : (s.top || []).map((t) => t.value);
    const vals = hist ? s.hist.map((b) => b.n) : (s.top || []).map((t) => t.n);
    return {
      grid: { left: 62, right: 16, top: 12, bottom: 34 },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      xAxis: { type: "category", data: cats, ...axis, splitLine: { show: false } },
      yAxis: { type: "value", ...axis, axisLine: { show: false }, axisLabel: { ...axis.axisLabel, formatter: (v) => num(v) } },
      series: [{ type: "bar", data: vals, barCategoryGap: 2, itemStyle: { borderRadius: [4, 4, 0, 0] } }],
    };
  };
  const loadPage = async () => { drawTable(await api(`/db/${state.table}?limit=${LIMIT}&offset=${state.offset}`)); };
  const fail = (e) => { if (!(e instanceof ApiError)) throw e; $("#tbl", el).innerHTML = pendingHtml(e); };
  const open = (table) => {
    state = { table, offset: 0, col: null };
    $$("#tabs button", el).forEach((b) => b.setAttribute("aria-pressed", b.dataset.t === table));
    Promise.all([loadPage(), drawColumn()]).catch(fail);
  };
  const start = async () => {
    const tables = await api("/db");
    if (!tables.length) throw new ApiError("Đang nạp các bảng dữ liệu.", 503);
    root.innerHTML = `<div class="stack"><div class="row" id="tabs">${tables.map((t) => `<button class="chip" data-t="${esc(t.name)}" aria-pressed="false">${esc(t.name)} <span class="mono">${num(t.rows)}</span></button>`).join("")}</div>
      <section class="lead"><header><h2>Phân bố cột <span id="col-title"></span></h2></header><div><div class="chart" id="hist"></div><p class="note" id="col-meta"></p></div></section>
      <section class="lead"><header><h2>Bản ghi</h2><div class="row"><span class="note" id="pos"></span><button class="btn-ghost" id="prev">Trước</button><button class="btn-ghost" id="next">Sau</button></div></header><div class="tbl-wrap" id="tbl"></div></section></div>`;
    ch = chart($("#hist", el), buildChart);
    open(tables.find((t) => t.group === "data")?.name || tables[0].name);
  };
  el.addEventListener("click", (e) => {
    const t = e.target.closest("[data-t]");
    if (t) { open(t.dataset.t); return; }
    const th = e.target.closest("th[data-col]");
    if (th) { state.col = th.dataset.col; drawColumn().catch(fail); return; }
    if (e.target.id === "prev" || e.target.id === "next") {
      state.offset = Math.max(0, state.offset + (e.target.id === "next" ? LIMIT : -LIMIT));
      loadPage().catch(fail);
    }
  });
  const stop = guarded(root, start);
  return () => { stop(); ch?.dispose(); };
}
