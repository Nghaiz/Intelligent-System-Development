// Khung ứng dụng: điều hướng, chuyển trang kiểu vạch quét, bảng lệnh Ctrl+K, đổi nền.
import { $, $$, esc, pendingHtml, reduced } from "./lib.js";
import { startMonitor } from "./monitor.js";

const GROUPS = [
  ["Tổng hợp", [["", "Tổng quan", "overview"]]],
  ["Lâm sàng", [["consult", "Hội chẩn", "consult"], ["screen", "Sàng lọc thích ứng", "screen"], ["risk", "Phòng thí nghiệm nguy cơ", "risk"], ["retina", "Võng mạc", "retina"]]],
  ["Tri thức", [["graph", "Đồ thị tri thức", "graph"], ["search", "Tra cứu", "search"]]],
  ["Vận hành", [["data", "Dữ liệu", "data"], ["lab", "Thí nghiệm", "lab"], ["logs", "Nhật ký", "logs"]]],
];
const PAGES = GROUPS.flatMap(([, items]) => items);
const view = $("#view");
let cleanup = null; let token = 0;

function renderNav() {
  let n = 0;
  $("#nav").innerHTML = GROUPS.map(([title, items]) => `<h2>${title}</h2>${items.map(([path, label]) => {
    n += 1;
    return `<a href="/${path}" data-nav><i>${String(n).padStart(2, "0")}</i><span>${label}</span></a>`;
  }).join("")}`).join("");
}

async function show(path) {
  const key = path.replace(/^\/|\/$/g, "");
  const entry = PAGES.find(([p]) => p === key) || PAGES[0];
  const mine = ++token;
  cleanup?.(); cleanup = null;
  $$("#nav a").forEach((a) => {
    const here = a.getAttribute("href") === `/${key}`;
    if (here) {
      a.setAttribute("aria-current", "page");
      const nav = a.parentElement; // chỉ cuộn thanh điều hướng ngang trên điện thoại, không cuộn cả trang
      if (nav.scrollWidth > nav.clientWidth) nav.scrollLeft = a.offsetLeft - nav.clientWidth / 2 + a.offsetWidth / 2;
    } else a.removeAttribute("aria-current");
  });
  document.title = `${entry[1]} · DiaNexus`;
  try {
    const mod = await import(`./pages/${entry[2]}.js`);
    if (mine !== token) return;
    view.className = reduced() ? "" : "sweep";
    view.innerHTML = `<div class="page-head"><h1>${esc(entry[1])}</h1><p>${esc(mod.lead)}</p></div><div id="page"></div>`;
    if (!reduced()) {
      const bar = document.createElement("i"); bar.className = "sweep-bar"; document.body.append(bar);
      bar.addEventListener("animationend", () => bar.remove());
    }
    scrollTo(0, 0);
    cleanup = await mod.mount($("#page")) || null;
  } catch (e) {
    view.innerHTML = pendingHtml({ status: 500, message: `Không mở được trang: ${e.message}` });
    throw e;
  }
}

export function go(path) {
  history.pushState(null, "", path);
  show(path);
}

function setTheme(next) {
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("dnx-theme", next); } catch { /* không lưu được thì chỉ có hiệu lực ở phiên này */ }
  dispatchEvent(new CustomEvent("dnx-theme"));
}
const toggleTheme = () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");

function palette() {
  const dlg = $("#palette"); const q = $("#palette-q"); const list = $("#palette-list");
  let sel = 0; let items = [];
  const build = () => {
    const text = q.value.trim(); const low = text.toLowerCase();
    items = PAGES.filter(([, label]) => label.toLowerCase().includes(low)).map(([p, label]) => ({ label, hint: "Trang", run: () => go(`/${p}`) }));
    items.push({ label: "Đổi nền sáng hoặc tối", hint: "Lệnh", run: toggleTheme });
    if (text) items.unshift({ label: `Hội chẩn: ${text}`, hint: "Gửi câu hỏi", run: () => { sessionStorage.setItem("dnx-ask", text); go("/consult"); } });
    sel = Math.min(sel, items.length - 1);
    list.innerHTML = items.map((it, i) => `<li role="option" data-i="${i}" aria-selected="${i === sel}"><span>${esc(it.label)}</span><small>${esc(it.hint)}</small></li>`).join("");
  };
  const open = () => { q.value = ""; sel = 0; build(); dlg.showModal(); q.focus(); };
  const pick = (i) => { dlg.close(); items[i]?.run(); };
  q.addEventListener("input", () => { sel = 0; build(); });
  q.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault(); sel = (sel + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length; build();
    } else if (e.key === "Enter") { e.preventDefault(); pick(sel); }
  });
  list.addEventListener("click", (e) => { const li = e.target.closest("li"); if (li) pick(Number(li.dataset.i)); });
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  $("#palette-open").addEventListener("click", open);
  addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); if (dlg.open) dlg.close(); else open(); }
  });
}

document.addEventListener("click", (e) => {
  const a = e.target.closest("a[data-nav]");
  if (!a || e.ctrlKey || e.metaKey || e.shiftKey) return;
  e.preventDefault();
  go(a.getAttribute("href"));
});
addEventListener("popstate", () => show(location.pathname));
$("#theme").addEventListener("click", toggleTheme);
renderNav(); palette(); startMonitor();
addEventListener("load", () => show(location.pathname));
