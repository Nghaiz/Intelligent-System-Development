// Live Logs: tệp log thật phát qua SSE, kèm đồng hồ GPU, commit, Neo4j và khóa GPU.
import { $, api, esc, sse } from "../lib.js";

const MAX_LINES = 600;
const COMMIT_SCALE_GB = 42;

function gauge(title, value, pct) {
  return `<div>${esc(title)} <b>${esc(value)}</b></div><div class="gauge"><i style="width:${Math.min(100, pct)}%"></i></div>`;
}

export default async function logs(view) {
  const { files } = await api("/logs");
  view.innerHTML = `<h2>Live Logs</h2><p class="lead">Nhật ký huấn luyện và đánh giá, đọc trực tiếp từ đĩa.</p>
    <div class="two"><div class="card"><div class="row"><select id="file"><option value="all">Mọi tệp</option>${files.slice(0, 60).map((f) => `<option>${esc(f.name)}</option>`).join("")}</select>
      <label><input type="checkbox" id="only"> Chỉ lỗi và cảnh báo</label><span class="chip" id="count">0 dòng</span></div><div class="log" id="log"></div></div>
    <div class="card"><h3>Máy</h3><div id="sys"></div></div></div>`;
  const box = $("#log");
  let stop = null, n = 0;
  const open = () => {
    stop?.(); box.innerHTML = ""; n = 0;
    stop = sse(`/api/logs/stream?file=${encodeURIComponent($("#file").value)}`, "line", (ev) => {
      if ($("#only").checked && ev.level === "INFO") return;
      const follow = box.scrollTop + box.clientHeight > box.scrollHeight - 40;
      box.insertAdjacentHTML("beforeend", `<div class="${esc(ev.level)}"><i>${esc(ev.file)}</i> ${esc(ev.text)}</div>`);
      if (box.children.length > MAX_LINES) box.firstChild.remove();
      $("#count").textContent = `${++n} dòng`;
      if (follow) box.scrollTop = box.scrollHeight;
    });
  };
  $("#file").onchange = $("#only").onchange = open;
  open();
  const sys = async () => {
    const s = await api("/status").catch(() => null);
    if (!s) return;
    const g = s.gpu;
    $("#sys").innerHTML = (g.ok ? gauge(`${g.name}, bộ nhớ`, `${g.mem_used_mb}/${g.mem_total_mb} MB`, 100 * g.mem_used_mb / g.mem_total_mb) + gauge("Tải GPU", `${g.util_pct}% · ${g.temp_c}°C`, g.util_pct) : `<p>GPU: ${esc(g.error)}</p>`)
      + gauge("Commit trống", `${s.commit.free_gb} GB`, 100 * s.commit.free_gb / COMMIT_SCALE_GB)
      + `<p><span class="chip ${s.neo4j.ok ? "ok" : "err"}">Neo4j ${s.neo4j.ok ? "đang chạy" : "tắt"}</span></p><p>Khóa GPU: ${esc(s.gpu_lock ?? "trống")}</p><p>Cập nhật ${esc(s.ts)}</p>`;
  };
  sys();
  const t = setInterval(sys, 5000);
  return () => { clearInterval(t); stop?.(); };
}
