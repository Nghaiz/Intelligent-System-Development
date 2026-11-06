// Tra cứu: cùng một câu hỏi qua 5 chế độ truy xuất đặt cạnh nhau; rê chuột lên một đoạn để thấy nó ở các cột khác.
import { $, $$, api, ApiError, esc, num, pendingHtml, safeUrl } from "../lib.js";

export const lead = "So năm cách truy xuất cạnh nhau trên cùng câu hỏi. Rê chuột lên một đoạn để thấy chế độ nào khác cũng tìm ra nó.";
const MODES = [["bm25", "BM25"], ["dense", "Vector e5"], ["graph", "Đồ thị"], ["hybrid", "Kết hợp RRF"], ["hybrid_rerank", "Kết hợp, xếp lại"]];

function highlight(text, q) {
  const words = [...new Set(q.toLowerCase().split(/\s+/).filter((w) => w.length > 2))].map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const safe = esc(text);
  if (!words.length) return safe;
  return safe.replace(new RegExp(`(${words.join("|")})`, "gi"), "<mark>$1</mark>");
}

function hit(r, q, i) {
  const url = safeUrl(r.url);
  return `<article class="hit" data-c="${esc(r.chunk_id)}"><h3><span class="mono">${i + 1}.</span> ${url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(r.title)}</a>` : esc(r.title)}</h3>
    <p>${highlight(String(r.text).slice(0, 420), q)}</p><small>${esc(r.source)} · ${esc(r.lang)} · ${num(r.score, 3)}${r.retrievers?.length ? ` · ${esc(r.retrievers.join(", "))}` : ""}</small></article>`;
}

export async function mount(el) {
  el.innerHTML = `<form class="row" id="f"><label class="field grow">Câu hỏi<input type="search" id="q" maxlength="400" value="Ngưỡng HbA1c chẩn đoán đái tháo đường là bao nhiêu?"></label><button class="btn" type="submit">Tra cứu</button></form>
    <div class="modes" id="modes">${MODES.map(([k, v]) => `<section class="mode" data-m="${k}"><h2>${v}<span></span></h2><div></div></section>`).join("")}</div>`;
  let seq = 0; const timers = [];
  const col = (k) => $(`[data-m="${k}"]`, el);
  // Mỗi cột tự thử lại khi chưa sẵn sàng; cột đã có kết quả giữ nguyên, không bị xóa theo cột đang chờ
  const fetchMode = async (k, q, mine) => {
    const c = col(k); const body = c.lastElementChild; const stamp = c.querySelector("h2 span");
    try {
      const r = await api(`/search?q=${encodeURIComponent(q)}&mode=${k}`);
      if (mine !== seq) return;
      stamp.textContent = `${num(r.seconds * 1000)} ms`;
      body.innerHTML = r.results.length ? r.results.map((x, i) => hit(x, q, i)).join("") : '<p class="note" style="padding:8px">Không có kết quả.</p>';
    } catch (e) {
      if (!(e instanceof ApiError)) throw e;
      if (mine !== seq) return;
      body.innerHTML = pendingHtml(e);
      if (e.status === 503) timers.push(setTimeout(() => fetchMode(k, q, mine), 10000));
    }
  };
  const run = () => {
    const q = $("#q", el).value.trim();
    if (!q) return;
    const mine = ++seq; timers.forEach(clearTimeout); timers.length = 0;
    MODES.forEach(([k]) => {
      col(k).lastElementChild.innerHTML = '<p class="note" style="padding:8px">Đang tìm...</p>';
      col(k).querySelector("h2 span").textContent = "";
      fetchMode(k, q, mine);
    });
  };
  const same = (id, on) => $$(".hit", el).forEach((h) => h.classList.toggle("same", on && h.dataset.c === id));
  el.addEventListener("mouseover", (e) => { const h = e.target.closest(".hit"); if (h) same(h.dataset.c, true); });
  el.addEventListener("mouseout", (e) => { const h = e.target.closest(".hit"); if (h) same(h.dataset.c, false); });
  $("#f", el).addEventListener("submit", (e) => { e.preventDefault(); run(); });
  run();
  return () => { seq += 1; timers.forEach(clearTimeout); };
}
