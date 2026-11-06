// Tiện ích chung: gọi API, thoát HTML, số kiểu Việt, trạng thái đang chuẩn bị, biểu đồ theo màu chủ đề.
export const $ = (s, r = document) => r.querySelector(s);
export const $$ = (s, r = document) => [...r.querySelectorAll(s)];
export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
export const num = (v, d = 0) => (typeof v === "number" && Number.isFinite(v)
  ? v.toLocaleString("vi-VN", { maximumFractionDigits: d, minimumFractionDigits: 0 }) : "—");
export const pct = (v, d = 1) => (typeof v === "number" ? `${(v * 100).toLocaleString("vi-VN", { maximumFractionDigits: d, minimumFractionDigits: d })}%` : "—");
export const signed = (v, d = 1) => (typeof v === "number" ? `${v >= 0 ? "+" : "−"}${(Math.abs(v) * 100).toLocaleString("vi-VN", { maximumFractionDigits: d, minimumFractionDigits: d })}` : "—");
export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : null);
export const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

export async function api(path, opts = {}) {
  const init = { ...opts };
  if (opts.json !== undefined) {
    init.method = init.method || "POST";
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.json);
    delete init.json;
  }
  let res;
  try {
    res = await fetch(`/api${path}`, init);
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError("Không kết nối được máy chủ DiaNexus.", 0);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new ApiError(body.error || `HTTP ${res.status}`, res.status);
  return body;
}

export function pendingHtml(err) {
  const preparing = err.status === 503;
  return `<div class="pending ${preparing ? "" : "err"}" role="status"><b>${preparing ? "Đang chuẩn bị" : "Có lỗi"}</b><span>${esc(err.message)}</span></div>`;
}

// Chạy fn; thành phần chưa sẵn sàng (503) thì báo ngay trên el và tự thử lại, không dựng số giả.
export function guarded(el, fn, retryMs = 8000) {
  let stopped = false; let timer = null;
  const run = async () => {
    if (stopped) return;
    try {
      await fn();
    } catch (e) {
      if (stopped) return;
      if (!(e instanceof ApiError)) throw e;
      el.innerHTML = pendingHtml(e);
      if (e.status === 503 || e.status === 0) timer = setTimeout(run, retryMs);
    }
  };
  run();
  return () => { stopped = true; clearTimeout(timer); };
}

export function countUp(el, to, fmt = (v) => num(v)) {
  if (typeof to !== "number") return;
  const from = Number(el.dataset.v || 0); el.dataset.v = to;
  if (reduced() || from === to) { el.textContent = fmt(to); return; }
  const t0 = performance.now(); const dur = 650;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur); const e = 1 - (1 - k) ** 3;
    el.textContent = fmt(Number.isInteger(to) ? Math.round(from + (to - from) * e) : from + (to - from) * e);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

export const seriesColors = () => [1, 2, 3, 4, 5, 6].map((i) => css(`--chart-${i}`));

// Biểu đồ ECharts theo màu hiện tại; đổi nền hoặc đổi cỡ thì tự vẽ lại.
export function chart(el, build) {
  let inst = null; let last = null;
  const axis = () => ({
    axisLine: { lineStyle: { color: css("--line-2") } }, axisTick: { show: false },
    axisLabel: { color: css("--ink-2"), fontFamily: "Plex Sans", fontSize: 12 },
    splitLine: { lineStyle: { color: css("--line") } },
  });
  const draw = () => {
    if (!last) return;
    inst?.dispose();
    inst = echarts.init(el, null, { renderer: "svg" });
    const opt = build(last, axis());
    inst.setOption({
      animationDuration: reduced() ? 0 : 400, color: seriesColors(), textStyle: { fontFamily: "Plex Sans", color: css("--ink") },
      ...opt,
      tooltip: { backgroundColor: css("--surface-2"), borderColor: css("--line-2"), textStyle: { color: css("--ink"), fontSize: 13 }, ...opt.tooltip },
    });
  };
  const onTheme = () => draw();
  const ro = new ResizeObserver(() => inst?.resize());
  ro.observe(el); addEventListener("dnx-theme", onTheme);
  return {
    update(data) { last = data; draw(); },
    dispose() { ro.disconnect(); removeEventListener("dnx-theme", onTheme); inst?.dispose(); },
  };
}

// Thanh SHAP hai phía quanh 0; chiều dài chuẩn theo đóng góp lớn nhất của chính danh sách.
export function shapHtml(rows, limit = 10) {
  const list = (rows || []).slice(0, limit);
  const top = Math.max(...list.map((r) => Math.abs(r.contribution)), 1e-9);
  return `<div class="shap">${list.map((r) => {
    const w = (Math.abs(r.contribution) / top) * 50; const neg = r.contribution < 0;
    const label = r.label_vi || r.feature; const value = r.value ?? "—";
    return `<div class="shap-row" title="${esc(label)} = ${esc(value)}"><span class="name">${esc(label)}</span>
      <span class="shap-track"><i class="shap-bar ${neg ? "neg" : ""}" style="left:${neg ? 50 - w : 50}%;width:${w}%"></i></span>
      <span class="val">${r.contribution >= 0 ? "+" : "−"}${Math.abs(r.contribution).toFixed(2).replace(".", ",")}</span></div>`;
  }).join("")}</div>`;
}

export const rulesHtml = (rules) => (rules || []).map((r) => {
  const tag = { urgent: "Khẩn", warn: "Lưu ý", info: "Thông tin" }[r.severity] || r.severity;
  return `<div class="rule" data-sev="${esc(r.severity)}"><b>${esc(tag)}</b><span>${esc(r.message)}${r.source ? ` <small class="note">(${esc(r.source)})</small>` : ""}</span></div>`;
}).join("");
