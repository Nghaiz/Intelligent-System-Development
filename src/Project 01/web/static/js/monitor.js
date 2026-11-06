// Thanh sinh hiệu: đọc /api/status và vẽ một đường theo dõi thật. Mỗi nhịp là một lần máy chủ trả lời,
// biên độ theo tải GPU; mất kết nối thì đường phẳng màu san hô.
import { api, css, esc, num, reduced } from "./lib.js";

const POLL_MS = 3000;
const BEAT = [0, .04, .08, 0, -.1, .95, -.28, .04, .06, .12, .1, .03, 0]; // hình dạng phức bộ QRS rút gọn
export let lastStatus = null;

function cells(s) {
  const gpu = s.gpu.ok ? `${s.gpu.util_pct}% · ${(s.gpu.mem_used_mb / 1024).toFixed(1).replace(".", ",")} GB` : "không đọc được";
  const rows = [
    ["Neo4j", s.neo4j.ok ? `${num(s.neo4j.nodes)} nút` : "chưa chạy", s.neo4j.ok, ""],
    ["Ollama", s.ollama.ok ? (s.ollama.loaded?.length ? "đang nạp LLM" : "sẵn sàng") : "chưa chạy", s.ollama.ok, "mid"],
    ["GPU", gpu, s.gpu.ok, ""],
    ["Commit trống", `${num(s.commit_free_gb, 1)} GB`, s.commit_free_gb > 6, "opt"],
    ["SQLite", s.db.ok ? `${num(s.db.rows)} dòng` : "chưa có", s.db.ok, "opt"],
  ];
  return rows.map(([k, v, ok, cls]) => `<div class="cell ${cls}" data-state="${ok ? "ok" : "bad"}"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("");
}

export function startMonitor() {
  const box = document.getElementById("vitals-cells");
  const cv = document.getElementById("trace");
  const ctx = cv.getContext("2d");
  let x = 0; let y0 = 0; let beat = null; let alive = true; let amp = .3; let colors = {};

  const size = () => {
    const r = window.devicePixelRatio || 1;
    cv.width = cv.clientWidth * r; cv.height = cv.clientHeight * r;
    ctx.setTransform(r, 0, 0, r, 0, 0); x = 0;
    colors = { ok: css("--teal"), bad: css("--coral"), bg: css("--surface") };
    ctx.fillStyle = colors.bg; ctx.fillRect(0, 0, cv.clientWidth, cv.clientHeight);
    const mid = cv.clientHeight * .62; // đường nền phẳng ngay từ đầu, nhịp thật quét đè lên
    ctx.strokeStyle = colors.ok; ctx.globalAlpha = .35; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(cv.clientWidth, mid); ctx.stroke(); ctx.globalAlpha = 1;
  };
  const frame = () => {
    const h = cv.clientHeight; const w = cv.clientWidth; const mid = h * .62;
    let v = 0;
    if (beat !== null) { v = BEAT[beat] ?? 0; beat = beat + 1 >= BEAT.length ? null : beat + 1; }
    const y = mid - v * amp * h * .55;
    ctx.fillStyle = colors.bg; ctx.fillRect(x, 0, 14, h); // vạch xóa phía trước, như máy theo dõi thật
    ctx.strokeStyle = alive ? colors.ok : colors.bad; ctx.lineWidth = 1.6;
    ctx.beginPath(); ctx.moveTo(x - 2, y0 || mid); ctx.lineTo(x, y); ctx.stroke();
    y0 = y; x = x >= w ? 0 : x + 2;
  };
  let raf = 0; let last = 0;
  const loop = (t) => {
    raf = requestAnimationFrame(loop);
    if (document.hidden || t - last < 33) return;
    last = t; frame();
  };

  const poll = async () => {
    try {
      const s = await api("/status");
      lastStatus = s; alive = true; amp = .3 + .7 * ((s.gpu.util_pct || 0) / 100); beat = 0;
      box.innerHTML = cells(s);
      if (reduced()) { for (let i = 0; i < 14; i += 1) frame(); }
      dispatchEvent(new CustomEvent("dnx-status", { detail: s }));
    } catch {
      alive = false; beat = null;
      box.innerHTML = `<div class="cell" data-state="bad"><span>Máy chủ</span><b>mất kết nối</b></div>`;
      if (reduced()) frame();
    }
  };
  size(); addEventListener("resize", size); addEventListener("dnx-theme", size);
  if (!reduced()) raf = requestAnimationFrame(loop);
  poll(); setInterval(poll, POLL_MS);
  return () => cancelAnimationFrame(raf);
}
