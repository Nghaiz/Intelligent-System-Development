// Tiện ích chung: gọi API, thoát HTML, vẽ biểu đồ theo màu chủ đề, luồng SSE.
export const $ = (s, r = document) => r.querySelector(s);
export const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const fmt = (v, d = 3) => (typeof v === "number"
  ? v.toLocaleString("vi-VN", { maximumFractionDigits: Number.isInteger(v) ? 0 : d }) : esc(v ?? "—"));

export async function api(path, opts) {
  const r = await fetch("/api" + path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail ?? `Lỗi HTTP ${r.status}`);
  return body;
}
export const post = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
export const palette = () => ["--a", "--b", "--c", "--d", "--ok", "--dim"].map(css);

const charts = new Set();
export function chart(el, option) {
  const c = echarts.init(el, null, { renderer: "canvas" });
  c.setOption({ color: palette(), textStyle: { fontFamily: "BVP, sans-serif", color: css("--fg") }, animationDuration: 700,
    tooltip: { trigger: "item", confine: true }, ...option });
  charts.add(c);
  return c;
}
export function disposeCharts() { charts.forEach((c) => c.dispose()); charts.clear(); }
addEventListener("resize", () => charts.forEach((c) => c.resize()));
export const axis = (extra = {}) => ({ axisLine: { lineStyle: { color: css("--line") } }, splitLine: { lineStyle: { color: css("--line"), opacity: .5 } },
  axisLabel: { color: css("--dim") }, nameTextStyle: { color: css("--dim") }, ...extra });

// Mở EventSource, trả hàm đóng. Lỗi kết nối đóng luôn, không tự thử lại vì mỗi lần chạy tốn LLM.
export function sse(url, event, onData, onEnd) {
  const s = new EventSource(url);
  s.addEventListener(event, (e) => onData(JSON.parse(e.data)));
  s.onerror = () => { s.close(); onEnd?.(); };
  return () => s.close();
}

export function countUp(el, to) {
  const t0 = performance.now();
  const step = (t) => {
    const k = Math.min(1, (t - t0) / 900);
    el.textContent = Math.round(to * (1 - (1 - k) ** 3)).toLocaleString("vi-VN");
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
export const empty = (msg) => `<div class="empty">${esc(msg)}</div>`;
export const errBox = (msg) => `<div class="err-box">${esc(msg)}</div>`;
