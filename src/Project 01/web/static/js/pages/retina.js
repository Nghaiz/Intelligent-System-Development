// Võng mạc: kéo-thả ảnh hoặc chọn mẫu, phân cấp 5 mức, Grad-CAM với thanh trượt trước-sau.
import { $, $$, api, ApiError, esc, pct, pendingHtml } from "../lib.js";

export const lead = "Ảnh đáy mắt được phân thành 5 cấp bệnh võng mạc đái tháo đường. Grad-CAM cho thấy vùng ảnh mà mô hình đã dựa vào.";
const GRADES = ["Không bệnh", "Nhẹ", "Trung bình", "Nặng", "Tăng sinh"];
const MAX = 8 << 20;

export async function mount(el) {
  el.innerHTML = `<div class="two"><div class="stack">
      <section class="lead"><h2>Ảnh đầu vào</h2><div class="stack" style="gap:12px">
        <label class="drop" id="drop" tabindex="0"><span>Kéo ảnh vào đây hoặc bấm để chọn</span><span class="note">PNG hoặc JPEG, tối đa 8 MB</span><input type="file" id="file" accept="image/png,image/jpeg" hidden></label>
        <div class="samples" id="samples"></div></div></section>
      <section class="lead"><h2>Phân cấp</h2><div id="out"><p class="note">Chưa có ảnh.</p></div></section>
    </div>
    <section class="lead"><h2>Grad-CAM trước và sau</h2><div id="cmp"><p class="note">Kết quả sẽ hiện ở đây.</p></div></section></div>`;
  const out = $("#out", el); const cmp = $("#cmp", el); let seq = 0;

  const send = async (blob) => {
    const mine = ++seq;
    if (blob.size > MAX) { out.innerHTML = pendingHtml({ status: 413, message: "Ảnh vượt 8 MB." }); return; }
    out.innerHTML = '<p class="note">Đang phân tích ảnh...</p>';
    const fd = new FormData(); fd.append("file", blob, "retina.png");
    try {
      const r = await api("/retina", { method: "POST", body: fd });
      if (mine !== seq) return;
      out.innerHTML = `<div class="ladder">${r.probs.map((p, i) => `<div class="${i === r.grade ? "me" : ""}"><span class="mono">${i}</span><span>${esc(GRADES[i])}</span><u style="width:${p * 100}%"></u><span class="mono">${pct(p, 0)}</span></div>`).join("")}</div>
        <p style="margin-top:12px">Cấp ${esc(r.grade)}: ${esc(r.label_vi)}. Kết quả chỉ mang tính tham khảo.</p>`;
      cmp.innerHTML = `<div class="compare" id="cc"><img src="data:image/png;base64,${esc(r.input_png_b64)}" alt="Ảnh đáy mắt mô hình đã đọc"><img class="over" src="data:image/png;base64,${esc(r.gradcam_png_b64)}" alt="Bản đồ Grad-CAM"><i class="bar"></i></div>
        <label class="field" style="max-width:460px;margin-top:8px">Vị trí so sánh<input type="range" id="at" min="0" max="100" value="50"></label>`;
      $("#at", el).addEventListener("input", (e) => $("#cc", el).style.setProperty("--at", `${e.target.value}%`));
    } catch (e) {
      if (!(e instanceof ApiError)) throw e;
      if (mine === seq) { out.innerHTML = pendingHtml(e); cmp.innerHTML = ""; }
    }
  };
  const drop = $("#drop", el); const file = $("#file", el);
  drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); if (e.dataTransfer.files[0]) send(e.dataTransfer.files[0]); });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); file.click(); } });
  file.addEventListener("change", () => { if (file.files[0]) send(file.files[0]); });

  const box = $("#samples", el);
  const { samples } = await api("/retina/samples");
  box.innerHTML = samples.length ? samples.map((s, i) => `<button data-s="${esc(s)}" aria-label="Ảnh mẫu ${i + 1}" aria-pressed="false"><img src="${esc(s)}" alt="" loading="lazy"></button>`).join("")
    : '<p class="note">Ảnh mẫu đang được chuẩn bị.</p>';
  box.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-s]");
    if (!b) return;
    $$("button", box).forEach((x) => x.setAttribute("aria-pressed", x === b));
    send(await (await fetch(b.dataset.s)).blob());
  });
  return () => { seq += 1; };
}
