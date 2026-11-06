// Khung ứng dụng: định tuyến phía khách, chủ đề sáng tối, bảng lệnh Ctrl+K, nền vòng đồng tâm.
import { $, api, disposeCharts, errBox, esc } from "./lib.js";

const PAGES = [["", "Tổng quan", "home"], ["graph", "Đồ thị tri thức", "graph"], ["chat", "Trợ lý GraphRAG", "chat"], ["arena", "Agent Arena", "arena"],
  ["lab", "Model Lab", "lab"], ["atlas", "Data Atlas", "atlas"], ["tracker", "Experiment Tracker", "tracker"], ["logs", "Live Logs", "logs"]];
let leave = null;

async function render() {
  const key = location.pathname.replace(/^\/|\/$/g, "");
  const page = PAGES.find((p) => p[0] === key) ?? PAGES[0];
  $("#nav").innerHTML = PAGES.map(([p, t]) => `<a href="/${p}" data-nav ${p === page[0] ? 'aria-current="page"' : ""}>${t}</a>`).join("");
  document.title = `${page[1]} · Đồng Tâm AI Lab`;
  leave?.(); leave = null; disposeCharts();
  const view = $("#view");
  view.innerHTML = '<p class="lead">Đang tải...</p>';
  try {
    const mod = await import(`./pages/${page[2]}.js`);
    leave = (await mod.default(view)) ?? null;
  } catch (e) { view.innerHTML = errBox(`Không mở được trang: ${e.message}`); }
  view.focus({ preventScroll: true });
}

export function go(path) { history.pushState(null, "", path); render(); }
addEventListener("popstate", render);
document.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-nav]");
  if (a && !e.ctrlKey && !e.metaKey) { e.preventDefault(); go(a.getAttribute("href")); }
});

const theme = localStorage.getItem("theme");
if (theme) document.documentElement.dataset.theme = theme;
$("#theme").onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("theme", next);
  render();
};

// Bảng lệnh: tìm trang, dataset và mô hình thật lấy từ API.
const dlg = $("#palette"), q = $("#palette-q"), list = $("#palette-list");
let items = [], sel = 0;
async function openPalette() {
  items = PAGES.map(([p, t]) => ({ text: t, kind: "Trang", path: "/" + p }));
  dlg.showModal(); q.value = ""; draw();
  const [ds, runs] = await Promise.all([api("/datasets").catch(() => ({ datasets: [] })), api("/runs").catch(() => ({ rows: [] }))]);
  items.push(...ds.datasets.map((d) => ({ text: d.name, kind: `Dataset · Ch${d.chapter}`, path: "/atlas" })));
  items.push(...[...new Set(runs.rows.map((r) => r.model))].map((m) => ({ text: m, kind: "Mô hình", path: "/tracker" })));
  draw();
}
function draw() {
  const k = q.value.toLowerCase();
  const hits = items.filter((i) => i.text.toLowerCase().includes(k)).slice(0, 12);
  sel = Math.min(sel, Math.max(0, hits.length - 1));
  list.innerHTML = hits.map((h, i) => `<li data-path="${esc(h.path)}" aria-selected="${i === sel}">${esc(h.text)} <span class="chip">${esc(h.kind)}</span></li>`).join("");
}
q.oninput = () => { sel = 0; draw(); };
q.onkeydown = (e) => {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); sel = Math.max(0, sel + (e.key === "ArrowDown" ? 1 : -1)); draw(); }
  if (e.key === "Enter") { const li = list.children[sel]; if (li) { dlg.close(); go(li.dataset.path); } }
};
list.onclick = (e) => { const li = e.target.closest("li"); if (li) { dlg.close(); go(li.dataset.path); } };
$("#palette-open").onclick = openPalette;
addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "k") { e.preventDefault(); openPalette(); } });

// Nền: các vòng đồng tâm lan chậm từ góc, dừng khi người dùng chọn giảm chuyển động.
(function background() {
  const c = $("#bg"), g = c.getContext("2d");
  const size = () => { c.width = innerWidth; c.height = innerHeight; };
  size(); addEventListener("resize", size);
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const frame = (t) => {
    g.clearRect(0, 0, c.width, c.height);
    g.strokeStyle = getComputedStyle(document.documentElement).getPropertyValue("--line").trim();
    g.lineWidth = 1;
    for (let i = 0; i < 16; i++) {
      const r = (i * 90 + t / 60) % 1440;
      g.globalAlpha = Math.max(0, .9 - r / 1440);
      g.beginPath(); g.arc(c.width * .86, c.height * .12, r, 0, 7); g.stroke();
    }
    if (!still) requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
})();

async function footer() {
  try {
    const s = await api("/status");
    const gpu = s.gpu.ok ? `GPU ${s.gpu.mem_used_mb}/${s.gpu.mem_total_mb} MB` : "GPU không đọc được";
    $("#foot-status").textContent = `${gpu} · commit trống ${s.commit.free_gb ?? "?"} GB · Neo4j ${s.neo4j.ok ? "đang chạy" : "tắt"}`;
  } catch { $("#foot-status").textContent = "Không đọc được trạng thái máy"; }
}
footer(); setInterval(footer, 8000);
render();
