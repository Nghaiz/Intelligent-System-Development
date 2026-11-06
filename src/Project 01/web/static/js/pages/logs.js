// Nhật ký: log trực tiếp qua SSE với bộ lọc mức, và vết tác tử của phiên đã lưu, phát lại được.
import { $, $$, api, ApiError, esc, guarded, num, pendingHtml, reduced } from "../lib.js";
import { createStrip } from "../strip.js";

export const lead = "Log thật từ các tiến trình đang chạy, kèm vết của mọi phiên hội chẩn đã lưu để xem lại từng bước.";
const LEVELS = [["error", "Lỗi"], ["warn", "Cảnh báo"], ["info", "Thông tin"], ["debug", "Gỡ lỗi"]];
const MAX_LINES = 800;
const REPLAY_MS = 90;

export async function mount(el) {
  el.innerHTML = `<div class="stack">
    <section class="lead"><header><h2>Log trực tiếp</h2><div class="row">
      <label class="field">Tệp<select id="file"><option value="all">Tất cả</option></select></label>
      ${LEVELS.map(([k, v]) => `<button class="chip" data-l="${k}" aria-pressed="true">${v}</button>`).join("")}
      <button class="btn-ghost" id="pause" aria-pressed="false">Tạm dừng</button></div></header>
      <div class="logbox" id="box" role="log" aria-label="Log trực tiếp"></div></section>
    <section class="lead"><h2>Vết tác tử đã lưu</h2><div class="two"><ul class="sess" id="sess"></ul><div><div class="row" style="margin-bottom:8px"><button class="btn-ghost" id="replay" disabled>Phát lại</button><span class="note" id="sess-title"></span></div><div id="strip"></div></div></div></section></div>`;
  const box = $("#box", el); let es = null; let paused = false; let current = []; let timer = null;
  const off = new Set();
  const line = (d) => `<div class="${esc(d.level)}" data-l="${esc(d.level)}" ${off.has(d.level) ? "hidden" : ""}><i>${esc(d.file)}</i><span>${esc(d.text)}</span></div>`;
  const connect = (file) => {
    es?.close();
    box.innerHTML = "";
    es = new EventSource(`/api/logs/stream?file=${encodeURIComponent(file)}`);
    es.addEventListener("line", (m) => {
      if (paused) return;
      const stick = box.scrollTop + box.clientHeight >= box.scrollHeight - 20;
      box.insertAdjacentHTML("beforeend", line(JSON.parse(m.data)));
      while (box.childElementCount > MAX_LINES) box.firstElementChild.remove();
      if (stick) box.scrollTop = box.scrollHeight;
    });
  };
  const { files } = await api("/logs");
  $("#file", el).insertAdjacentHTML("beforeend", files.map((f) => `<option>${esc(f.name)}</option>`).join(""));
  $("#file", el).addEventListener("change", (e) => connect(e.target.value));
  connect("all");

  const strip = createStrip($("#strip", el));
  const replay = () => {
    clearInterval(timer); strip.clear();
    if (reduced()) { current.forEach(strip.add); return; }
    let i = 0;
    timer = setInterval(() => { if (i >= current.length) { clearInterval(timer); return; } strip.add(current[i]); i += 1; }, REPLAY_MS);
  };
  const stop = guarded($("#sess", el), async () => {
    const { sessions } = await api("/sessions");
    $("#sess", el).innerHTML = sessions.length ? sessions.map((s) => `<li><button data-s="${esc(s.id)}" aria-pressed="false"><span>${esc(s.title || s.first_question || "Phiên chưa đặt tên")}</span><small>${esc(s.created_at)} · ${num(s.n_events)} sự kiện</small></button></li>`).join("") : '<li class="note">Chưa có phiên nào được lưu.</li>';
  }, 20000);

  el.addEventListener("click", async (e) => {
    const l = e.target.closest(".chip[data-l]");
    if (l) {
      const on = l.getAttribute("aria-pressed") !== "true";
      l.setAttribute("aria-pressed", on);
      if (on) off.delete(l.dataset.l); else off.add(l.dataset.l);
      $$(`[data-l="${l.dataset.l}"]`, box).forEach((n) => { n.hidden = !on; });
      return;
    }
    if (e.target.id === "pause") { paused = !paused; e.target.setAttribute("aria-pressed", paused); e.target.textContent = paused ? "Tiếp tục" : "Tạm dừng"; return; }
    if (e.target.id === "replay") { replay(); return; }
    const s = e.target.closest("[data-s]");
    if (s) {
      $$("[data-s]", el).forEach((b) => b.setAttribute("aria-pressed", b === s));
      try {
        const d = await api(`/sessions/${encodeURIComponent(s.dataset.s)}`);
        current = d.events; $("#replay", el).disabled = !current.length;
        $("#sess-title", el).textContent = `${num(current.length)} sự kiện`;
        replay();
      } catch (err) { if (!(err instanceof ApiError)) throw err; $("#strip", el).innerHTML = pendingHtml(err); }
    }
  });
  return () => { es?.close(); clearInterval(timer); stop(); };
}
