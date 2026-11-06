// Agent Arena: xem các tác tử trao đổi trực tiếp, hoặc phát lại vết thông điệp thật đã ghi mà không cần nạp LLM.
import { $, api, empty, errBox, esc, palette, sse } from "../lib.js";

const ARCHS = { single: "ReAct đơn", supervisor: "Giám sát", debate: "Tranh luận" };
const color = (name, seen, cols) => { if (!seen.has(name)) seen.set(name, cols[seen.size % cols.length]); return seen.get(name); };

export default async function arena(view) {
  const o = await api("/arena/options");
  const opt = o.inputs.find((i) => i.name === "dataset")?.options ?? ["clinical"];
  view.innerHTML = `<h2>Agent Arena</h2><p class="lead">So sánh ba kiến trúc trên cùng một câu hỏi: từng thông điệp, số token và độ trễ đều là số đo thật.</p>
    ${o.status === "ok" ? "" : errBox(o.message)}
    <div class="two"><div class="card"><div id="stage" class="msgs">${empty("Chọn một vết đã ghi hoặc chạy trực tiếp.")}</div></div><div class="side">
      <div class="card"><h3>Chạy trực tiếp</h3><textarea id="q" placeholder="Câu hỏi hoặc ca bệnh"></textarea>
        <div class="row"><select id="arch">${o.archs.map((a) => `<option value="${a}">${ARCHS[a] ?? a}</option>`).join("")}</select>
        <select id="model">${o.models.map((m) => `<option>${esc(m)}</option>`).join("")}</select>
        <select id="ds">${opt.map((d) => `<option>${esc(d)}</option>`).join("")}</select><button class="primary" id="go">Chạy</button></div></div>
      <div class="card"><h3>Vết đã ghi (${o.replays.length})</h3><div class="tbl" style="max-height:320px" id="list">${o.replays.length ? `<table><tr><th>Miền</th><th>Kiến trúc</th><th>Đúng</th><th>Token</th></tr>${o.replays.map((r) => `<tr data-i="${r.index}" style="cursor:pointer"><td>${esc(r.dataset)}</td><td>${esc(ARCHS[r.arch] ?? r.arch)}</td><td>${r.correct === null || r.correct === undefined ? "—" : r.correct ? "✓" : "✗"}</td><td class="num">${r.tokens}</td></tr>`).join("")}</table>` : empty("Chưa có vết nào.")}</div></div>
    </div></div>`;
  const stage = $("#stage"), cols = palette();
  let seen = new Map(), timer = null, stop = null;
  const reset = () => { clearTimeout(timer); stop?.(); seen = new Map(); stage.innerHTML = ""; };
  const add = (m) => {
    const who = m.from ?? m.agent ?? "tác tử";
    stage.insertAdjacentHTML("beforeend", `<div class="msg" style="--c1:${color(who, seen, cols)}"><small>${esc(who)}${m.to ? " → " + esc(m.to) : ""}${m.tokens ? ` · ${m.tokens} token` : ""}</small>${esc(m.content ?? m.text ?? "")}</div>`);
    stage.scrollTop = stage.scrollHeight;
  };
  $("#list").onclick = async (e) => {
    const tr = e.target.closest("tr[data-i]");
    if (!tr) return;
    reset();
    try {
      const item = await api(`/arena/replay/${tr.dataset.i}`);
      const play = (i) => { if (i < item.messages.length) { add(item.messages[i]); timer = setTimeout(() => play(i + 1), 650); } };
      play(0);
    } catch (err) { stage.innerHTML = errBox(err.message); }
  };
  $("#go").onclick = () => {
    const q = $("#q").value.trim();
    if (q.length < 2) return;
    reset(); $("#go").disabled = true;
    stage.innerHTML = '<div class="msg user">Đang chờ GPU và mô hình...</div>';
    const done = () => { $("#go").disabled = false; stop?.(); };
    stop = sse(`/api/arena/stream?question=${encodeURIComponent(q)}&arch=${$("#arch").value}&model=${$("#model").value}&dataset=${$("#ds").value}`, "ev", (ev) => {
      if (ev.type === "message") add(ev);
      if (ev.type === "final") { add({ from: "Kết quả", content: ev.answer ?? "" }); done(); }
      if (ev.type === "error") { stage.insertAdjacentHTML("beforeend", errBox(ev.message)); done(); }
    }, () => { $("#go").disabled = false; });
  };
  return () => { clearTimeout(timer); stop?.(); };
}
