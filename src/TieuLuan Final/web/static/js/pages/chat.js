// Trợ lý GraphRAG: dải đèn pipeline sáng theo từng sự kiện SSE thật, kèm nguồn đã truy xuất.
import { $, api, empty, errBox, esc, sse } from "../lib.js";

const STAGES = ["Câu hỏi", "Truy xuất", "Sinh câu trả lời", "Xong"];
const DATASETS = [["hetionet", "Y sinh (Hetionet)"], ["vn30", "Tài chính (VN30)"]];

export default async function chat(view) {
  const o = await api("/chat/options");
  view.innerHTML = `<h2>Trợ lý GraphRAG</h2><p class="lead">Câu trả lời chạy trên LLM cục bộ. Nội dung y tế và đầu tư chỉ mang tính học thuật.</p>
    ${o.status === "ok" ? "" : errBox(o.message)}
    <div class="two"><div class="card"><div class="pipeline" id="pipe">${STAGES.map((s) => `<span>${s}</span>`).join("")}</div>
      <div class="msgs" id="msgs">${empty("Hãy đặt một câu hỏi.")}</div>
      <div class="row"><textarea id="q" placeholder="Ví dụ: Metformin điều trị bệnh gì và tác động lên gen nào?"></textarea></div>
      <div class="row"><select id="mode">${o.modes.map((m) => `<option>${esc(m)}</option>`).join("")}</select>
        <select id="model">${o.models.map((m) => `<option>${esc(m)}</option>`).join("")}</select>
        <select id="ds">${DATASETS.map(([k, t]) => `<option value="${k}">${t}</option>`).join("")}</select>
        <button class="primary" id="send">Gửi</button></div></div>
    <div class="card"><h3>Nguồn đã truy xuất</h3><div id="src">${empty("Chưa có nguồn.")}</div></div></div>`;
  let stop = null;
  const light = (i) => [...$("#pipe").children].forEach((s, j) => { s.className = j < i ? "done" : j === i ? "on" : ""; });
  $("#send").onclick = () => {
    const q = $("#q").value.trim();
    if (q.length < 2) return;
    const msgs = $("#msgs"), btn = $("#send");
    if (msgs.querySelector(".empty")) msgs.innerHTML = "";
    msgs.insertAdjacentHTML("beforeend", `<div class="msg user">${esc(q)}</div><div class="msg" id="live"><small>Đang chờ GPU và mô hình...</small><span></span></div>`);
    const live = msgs.lastElementChild, text = live.querySelector("span"), note = live.querySelector("small");
    live.removeAttribute("id"); btn.disabled = true; light(0);
    const done = () => { btn.disabled = false; stop?.(); };
    stop = sse(`/api/chat/stream?q=${encodeURIComponent(q)}&mode=${$("#mode").value}&model=${$("#model").value}&dataset=${$("#ds").value}`, "ev", (ev) => {
      if (ev.type === "step") { light(1); note.textContent = ev.text ?? ev.kind; }
      if (ev.type === "retrieved") {
        light(2);
        $("#src").innerHTML = ev.chunks.map((c) => `<div class="msg"><small>${esc(c.id)} · điểm ${esc(c.score)}</small>${esc(c.title ?? "")}</div>`).join("");
      }
      if (ev.type === "token") { light(2); text.textContent += ev.text; msgs.scrollTop = msgs.scrollHeight; }
      if (ev.type === "final") {
        light(3); note.textContent = `${ev.model ?? ""} · ${ev.mode ?? ""} · ${ev.seconds ?? "?"} giây`;
        if (ev.error) live.insertAdjacentHTML("beforeend", errBox(ev.error)); else if (!text.textContent) text.textContent = ev.answer;
        done();
      }
      if (ev.type === "error") { live.insertAdjacentHTML("beforeend", errBox(ev.message)); done(); }
    }, () => { btn.disabled = false; });
  };
  return () => stop?.();
}
