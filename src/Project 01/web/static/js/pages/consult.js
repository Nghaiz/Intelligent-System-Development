// Hội chẩn: gửi câu hỏi, băng tác tử trực tiếp qua SSE, câu trả lời gõ dần, trích dẫn [n], thẻ nguy cơ, đồ thị con, phản hồi.
import { $, api, ApiError, esc, pct, pendingHtml, reduced, rulesHtml, safeUrl, shapHtml } from "../lib.js";
import { createStrip } from "../strip.js";
import { miniGraph } from "../minigraph.js";

export const lead = "Tám tác tử cùng làm việc trên một hồ sơ. Băng bên phải cho thấy ai đang làm gì, từng câu trả lời đều dẫn nguồn.";

const CONFIGS = [["full", "Đầy đủ"], ["no_critic", "Bỏ Phản biện"], ["vector_only", "Chỉ vector"], ["no_rules", "Bỏ luật"], ["single_agent", "Một tác tử"]];
const FIELDS = [["age", "Tuổi", "number"], ["bmi", "BMI", "number"], ["hba1c", "HbA1c (%)", "number"], ["glucose", "Đường huyết (mg/dL)", "number"]];

const body = () => `<div class="consult">
  <div>
    <div class="log" id="log" aria-live="polite"><p class="note">Hãy mô tả tình trạng, kết quả xét nghiệm hoặc câu hỏi về đái tháo đường.</p></div>
    <form class="composer" id="form">
      <label class="field">Câu hỏi<textarea id="q" rows="3" maxlength="2000" required></textarea></label>
      <details><summary>Hồ sơ và ảnh đính kèm (không bắt buộc)</summary>
        <div class="fields">${FIELDS.map(([k, l, t]) => `<label class="field">${l}<input type="${t}" step="any" data-p="${k}"></label>`).join("")}
          <label class="field">Giới tính<select data-p="sex"><option value="">Chưa rõ</option><option value="male">Nam</option><option value="female">Nữ</option></select></label>
          <label class="field">Ảnh võng mạc<input type="file" id="img" accept="image/png,image/jpeg"></label></div>
      </details>
      <div class="row"><label class="field grow">Cấu hình<select id="config">${CONFIGS.map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select></label>
      <button class="btn" id="send" type="submit">Gửi hội chẩn</button></div>
    </form>
  </div>
  <div class="stack">
    <section class="lead"><h2>Băng tác tử</h2><div id="strip"></div></section>
    <section class="lead" id="risk-box" hidden><h2>Thẻ nguy cơ</h2><div id="risk"></div></section>
    <section class="lead" id="graph-box" hidden><h2>Đồ thị con liên quan</h2><div id="mini"></div></section>
  </div></div>`;

const profileOf = (el) => {
  const p = {};
  el.querySelectorAll("[data-p]").forEach((i) => {
    if (i.value !== "") p[i.dataset.p] = i.type === "number" ? Number(i.value) : i.value;
  });
  return Object.keys(p).length ? p : null;
};
const fileB64 = (file) => new Promise((ok, no) => {
  const r = new FileReader(); r.onload = () => ok(r.result); r.onerror = () => no(r.error); r.readAsDataURL(file);
});

const answerHtml = (text) => esc(text).replace(/\[(\d{1,2})\]/g, '<a class="cite" href="#" data-n="$1">[$1]</a>');

function typeOut(node, text, done) {
  if (reduced() || !text) { node.innerHTML = answerHtml(text); done(); return; }
  node.classList.add("typing");
  let i = 0; const stepSize = Math.max(2, Math.round(text.length / 160));
  const tick = () => {
    i = Math.min(text.length, i + stepSize);
    node.textContent = text.slice(0, i);
    if (i < text.length) requestAnimationFrame(tick);
    else { node.classList.remove("typing"); node.innerHTML = answerHtml(text); done(); }
  };
  tick();
}

function sources(cites) {
  if (!cites?.length) return "";
  return `<ol class="sources">${cites.map((c) => {
    const url = safeUrl(c.url);
    return `<li id="cite-${esc(c.n)}"><span class="mono">[${esc(c.n)}]</span><span>${url ? `<a href="${esc(url)}" target="_blank" rel="noopener noreferrer">${esc(c.title)}</a>` : esc(c.title)}${c.quote ? `<q>${esc(c.quote)}</q>` : ""}</span></li>`;
  }).join("")}</ol>`;
}

function riskCard(r, rules, retina) {
  const risk = r ? `<div class="risk-num">${pct(r.prob)}</div><p class="note">Mô hình ${esc(r.model)}, chế độ ${r.mode === "full" ? "có xét nghiệm" : "sàng lọc"}, ngưỡng ${pct(r.threshold)}</p>${shapHtml(r.shap, 6)}` : "";
  const ret = retina ? `<p>Võng mạc: cấp ${esc(retina.grade)} · ${esc(retina.label_vi)}</p>` : "";
  return risk + ret + rulesHtml(rules);
}

export async function mount(el) {
  el.innerHTML = body();
  const strip = createStrip($("#strip", el)); const log = $("#log", el); const form = $("#form", el);
  let es = null; let sid = null; let hint = true;
  const add = (cls, html) => {
    if (hint) { log.innerHTML = ""; hint = false; }
    log.insertAdjacentHTML("beforeend", `<div class="msg ${cls}">${html}</div>`);
    return log.lastElementChild;
  };
  const finish = () => { es?.close(); es = null; $("#send", el).disabled = false; };

  const showResult = (res, q) => {
    const box = add("bot", "");
    typeOut(box, res.answer || "", () => {
      box.insertAdjacentHTML("beforeend", sources(res.citations));
      box.insertAdjacentHTML("afterend", `<div class="feedback" data-sid="${esc(sid)}"><span>Câu trả lời này có ích không?</span>
        <button class="btn-ghost" data-r="1">Có ích</button><button class="btn-ghost" data-r="-1">Chưa ổn</button></div>`);
    });
    if (res.risk || res.rules?.length || res.retina) { $("#risk-box", el).hidden = false; $("#risk", el).innerHTML = riskCard(res.risk, res.rules, res.retina); }
    miniGraph($("#mini", el), $("#graph-box", el), q);
  };

  const listen = (q) => {
    es = new EventSource(`/api/chat/${sid}/stream`);
    es.addEventListener("ev", (m) => {
      const ev = JSON.parse(m.data);
      if (ev.kind === "result") showResult(ev.data, q);
      else if (ev.kind === "fatal") add("bot", pendingHtml({ status: 500, message: ev.detail }));
      else strip.add(ev);
    });
    es.addEventListener("done", finish);
    es.onerror = () => { if (es) { add("bot", pendingHtml({ status: 0, message: "Mất kết nối luồng sự kiện." })); finish(); } };
  };

  const send = async () => {
    const q = $("#q", el).value.trim();
    if (!q) return;
    $("#send", el).disabled = true;
    try {
      const file = $("#img", el).files[0];
      if (file && file.size > 8 << 20) throw new ApiError("Ảnh vượt 8 MB.", 413);
      const out = await api("/chat", { json: { query: q, profile: profileOf(el), config: $("#config", el).value, session_id: sid, image_b64: file ? await fileB64(file) : null } });
      sid = out.session_id;
      add("user", esc(q)); $("#q", el).value = ""; strip.clear(); listen(q);
    } catch (e) {
      if (!(e instanceof ApiError)) throw e;
      add("bot", pendingHtml(e)); $("#send", el).disabled = false;
    }
  };
  form.addEventListener("submit", (e) => { e.preventDefault(); send(); });
  $("#q", el).addEventListener("keydown", (e) => { if (e.key === "Enter" && e.ctrlKey) { e.preventDefault(); send(); } });
  log.addEventListener("click", async (e) => {
    const c = e.target.closest("a.cite");
    if (c) {
      e.preventDefault();
      const li = log.querySelector(`#cite-${c.dataset.n}`);
      if (li) { li.scrollIntoView({ block: "nearest" }); li.classList.add("flash"); setTimeout(() => li.classList.remove("flash"), 1200); }
    }
    const r = e.target.closest("[data-r]");
    if (r) {
      const box = r.closest(".feedback");
      try {
        await api("/feedback", { json: { session_id: box.dataset.sid, rating: Number(r.dataset.r), comment: "" } });
        box.textContent = "Đã ghi nhận phản hồi của bạn.";
      } catch (err) { if (!(err instanceof ApiError)) throw err; box.innerHTML = pendingHtml(err); }
    }
  });
  const asked = sessionStorage.getItem("dnx-ask");
  if (asked) { sessionStorage.removeItem("dnx-ask"); $("#q", el).value = asked; send(); }
  return () => { es?.close(); es = null; };
}
