// Tổng quan: số sống từ /api/status, sơ đồ hai nhánh DB và KB hội tụ vào Agent.
import { api, countUp, esc, num, reduced } from "../lib.js";
import { go } from "../app.js";
import { lastStatus } from "../monitor.js";

export const lead = "Hai nhánh dữ liệu bảng và tri thức y khoa hội tụ vào hệ đa tác tử. Đường sáng là phần đang chạy thật, đường đứt là phần đang chuẩn bị.";

// [id, x, y, tiêu đề, trang đích, hàm phụ đề, hàm sẵn sàng]
const W = 158; const H = 54;
const NODES = [
  ["sql", 30, 36, "SQLite 5 bộ dữ liệu", "data", (s) => (s.db.ok ? `${num(s.db.rows)} dòng` : "đang chuẩn bị"), (s) => s.db.ok],
  ["ml", 250, 36, "Mô hình nguy cơ", "risk", () => "XGBoost, SHAP", (s) => s.modules["dnx.ml.serve"].ready],
  ["rl", 250, 106, "Sàng lọc RL", "screen", () => "Double DQN", (s) => s.modules["dnx.rl.policy"].ready],
  ["img", 30, 176, "Ảnh đáy mắt", "retina", () => "5 cấp bệnh", (s) => s.modules["dnx.ml.serve_retina"].ready],
  ["ret", 250, 176, "CNN võng mạc", "retina", () => "Grad-CAM", (s) => s.modules["dnx.ml.serve_retina"].ready],
  ["docs", 30, 276, "Tài liệu, PubMed", "search", (s) => (s.chunks ? `${num(s.chunks)} đoạn` : "đang chuẩn bị"), (s) => !!s.chunks],
  ["kg", 30, 346, "Đồ thị Neo4j", "graph", (s) => (s.neo4j.ok ? `${num(s.neo4j.nodes)} nút` : "chưa chạy"), (s) => s.neo4j.ok && s.neo4j.nodes > 0],
  ["rag", 250, 311, "HybridRAG", "search", () => "BM25, e5, đồ thị", (s) => s.modules["dnx.kb.search"].ready],
  ["rule", 500, 300, "RuleGuard", "consult", () => "luật có nguồn", (s) => s.modules["dnx.rules"].ready],
  ["hub", 500, 170, "8 tác tử", "consult", () => "có phản biện", (s) => s.modules["dnx.agents.graph"].ready],
  ["ans", 730, 170, "Câu trả lời", "consult", () => "trích dẫn [n]", (s) => s.modules["dnx.agents.graph"].ready],
];
const WIRES = [["sql", "ml"], ["sql", "rl"], ["img", "ret"], ["ml", "hub"], ["rl", "hub"], ["ret", "hub"], ["docs", "rag"], ["kg", "rag"], ["rag", "hub"], ["rule", "hub"], ["hub", "ans"]];
const byId = Object.fromEntries(NODES.map((n) => [n[0], n]));

function wirePath(a, b) {
  const [, ax, ay] = byId[a]; const [, bx, by] = byId[b];
  const x0 = ax + W; const y0 = ay + H / 2; const x1 = bx; const y1 = by + H / 2;
  if (a === "rule") return `M${ax + W / 2} ${ay} V${by + H}`;
  const mid = (x0 + x1) / 2;
  return `M${x0} ${y0} C${mid} ${y0} ${mid} ${y1} ${x1} ${y1}`;
}

function archSvg(s) {
  const ready = Object.fromEntries(NODES.map((n) => [n[0], n[6](s)]));
  const wires = WIRES.map(([a, b], i) => {
    const live = ready[a] && ready[b];
    const d = wirePath(a, b); const dot = live && !reduced()
      ? `<circle class="pulse" r="3.2" opacity="0"><set attributeName="opacity" to="1" begin="${(i % 5) * .35}s"/><animateMotion dur="${2.4 + (i % 4) * .4}s" repeatCount="indefinite" begin="${(i % 5) * .35}s"><mpath href="#w${i}"/></animateMotion></circle>` : "";
    return `<path id="w${i}" class="wire ${live ? "live" : "off"}" d="${d}"/>${dot}`;
  }).join("");
  const nodes = NODES.map(([id, x, y, title, page, sub]) => `
    <g class="node ${ready[id] ? "live" : ""} ${id === "hub" ? "hub" : ""}" data-go="${page}" transform="translate(${x} ${y})" tabindex="0" role="link" aria-label="${esc(title)}">
      <rect width="${W}" height="${H}" rx="2"/><text x="12" y="23">${esc(title)}</text><text class="sub" x="12" y="42">${esc(sub(s))}</text></g>`).join("");
  return `<svg class="arch" viewBox="0 0 900 416" role="img" aria-label="Sơ đồ kiến trúc DiaNexus">
    <text class="branch" x="30" y="24">NHÁNH DỮ LIỆU</text><text class="branch" x="30" y="264">NHÁNH TRI THỨC</text>${wires}${nodes}</svg>`;
}

const STATS = [["rows", "dòng dữ liệu SQLite", (s) => (s.db.ok ? s.db.rows : null)], ["chunks", "đoạn tri thức", (s) => s.chunks],
  ["nodes", "nút Neo4j", (s) => (s.neo4j.ok ? s.neo4j.nodes : null)], ["edges", "cạnh Neo4j", (s) => (s.neo4j.ok ? s.neo4j.edges : null)],
  ["runs", "lượt huấn luyện đã ghi", (s) => s.runs], ["metrics", "tệp số liệu", (s) => s.metrics]];

export async function mount(el) {
  el.innerHTML = `<div class="ov-stats">${STATS.map(([k, label]) => `<div class="stat wait" data-k="${k}"><b>đang chuẩn bị</b><span>${label}</span></div>`).join("")}</div>
    <div class="two wide"><section class="lead"><h2>Kiến trúc</h2><div class="arch-wrap" id="arch"></div></section>
    <section class="lead"><h2>Trạng thái thành phần</h2><ul class="ready" id="ready"></ul></section></div>`;
  let shape = "";
  const paint = (s) => {
    STATS.forEach(([k, , get]) => {
      const box = el.querySelector(`[data-k="${k}"]`); const v = get(s);
      if (v === null || v === undefined) return;
      box.classList.remove("wait"); countUp(box.querySelector("b"), v);
    });
    // chỉ dựng lại sơ đồ khi trạng thái đổi, để hạt dữ liệu không nhảy về đầu mỗi 3 giây
    const key = JSON.stringify([NODES.map((n) => n[5](s)), NODES.map((n) => n[6](s))]);
    if (key !== shape) { shape = key; el.querySelector("#arch").innerHTML = archSvg(s); }
    const mods = Object.values(s.modules).map((m) => `<li><span>${esc(m.what)}</span><b class="${m.ready ? "ok" : "wait"}">${m.ready ? "có mô-đun" : "đang chuẩn bị"}</b></li>`);
    mods.push(`<li><span>Neo4j</span><b class="${s.neo4j.ok ? "ok" : "wait"}">${s.neo4j.ok ? "đang chạy" : "chưa chạy"}</b></li>`);
    mods.push(`<li><span>Ollama</span><b class="${s.ollama.ok ? "ok" : "wait"}">${s.ollama.ok ? "đang chạy" : "chưa chạy"}</b></li>`);
    mods.push(`<li><span>GPU</span><b class="${s.gpu.ok ? "ok" : "wait"}">${s.gpu.ok ? esc(s.gpu.name.replace("NVIDIA GeForce ", "")) : "không đọc được"}</b></li>`);
    if (s.gpu_lock) mods.push(`<li><span>Khóa GPU nặng</span><b class="wait">${esc(s.gpu_lock.split(" ").slice(1, 4).join(" "))}</b></li>`);
    el.querySelector("#ready").innerHTML = mods.join("");
  };
  const onClick = (e) => { const g = e.target.closest("[data-go]"); if (g) go(`/${g.dataset.go}`); };
  const onKey = (e) => { if (e.key === "Enter") onClick(e); };
  el.addEventListener("click", onClick); el.addEventListener("keydown", onKey);
  const onStatus = (e) => paint(e.detail);
  addEventListener("dnx-status", onStatus);
  paint(lastStatus || await api("/status"));
  return () => removeEventListener("dnx-status", onStatus);
}
