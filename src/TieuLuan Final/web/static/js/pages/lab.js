// Model Lab: biểu mẫu sinh tự động từ SPECS của serve_chN, thêm mô hình mới không phải sửa trang.
import { $, api, axis, chart, css, empty, errBox, esc, fmt, post } from "../lib.js";

function field(i) {
  const id = `f-${i.name}`, lab = `<label for="${id}">${esc(i.label ?? i.name)}</label>`;
  if (i.options) return `<div>${lab}<br><select id="${id}" data-n="${esc(i.name)}">${i.options.map((o) => { const v = o?.value ?? o; return `<option value="${esc(v)}" ${v === i.default ? "selected" : ""}>${esc(o?.label ?? o)}</option>`; }).join("")}</select></div>`;
  if (i.type === "number" || i.type === "slider") return `<div>${lab}<br><input id="${id}" data-n="${esc(i.name)}" type="number" step="any" value="${esc(i.default)}" ${i.min !== undefined ? `min="${i.min}"` : ""} ${i.max !== undefined ? `max="${i.max}"` : ""}></div>`;
  return `<div style="flex:1 1 100%">${lab}<br><textarea id="${id}" data-n="${esc(i.name)}">${esc(i.default)}</textarea></div>`;
}

function result(box, r) {
  const parts = [];
  if (r.label !== undefined) parts.push(`<h2>${esc(r.label)}</h2>`);
  if (r.text) parts.push(`<div class="msg">${esc(r.text)}</div>`);
  if (r.image) parts.push(`<img alt="Kết quả" style="max-width:100%;border-radius:8px" src="${esc(r.image)}">`);
  if (r.note) parts.push(`<p class="lead">${esc(r.note)}</p>`);
  const known = new Set(["label", "text", "image", "note", "probs", "series", "history", "forecast"]);
  const isImage = (v) => typeof v === "string" && v.startsWith("data:image");
  Object.entries(r).filter(([, v]) => isImage(v)).forEach(([k, v]) => parts.push(`<img alt="${esc(k)}" style="max-width:260px;image-rendering:pixelated;border-radius:8px" src="${esc(v)}">`));
  const rest = Object.entries(r).filter(([k, v]) => !known.has(k) && typeof v !== "object" && !isImage(v));
  if (rest.length) parts.push(`<table>${rest.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${fmt(v)}</td></tr>`).join("")}</table>`);
  parts.push('<div class="chart" id="r-c" hidden></div>');
  box.innerHTML = parts.join("");
  const el = $("#r-c");
  if (r.probs) {
    const e = Array.isArray(r.probs) ? r.probs.map((v, i) => [String(i), v]) : Object.entries(r.probs);
    e.sort((a, b) => b[1] - a[1]); el.hidden = false;
    chart(el, { grid: { left: 110, right: 16, top: 8, bottom: 24 }, xAxis: axis({ type: "value", max: 1 }), yAxis: axis({ type: "category", inverse: true, data: e.map((x) => x[0]) }),
      series: [{ type: "bar", data: e.map((x) => +x[1].toFixed(4)), itemStyle: { borderRadius: [0, 5, 5, 0] } }] });
  } else if (r.history) {
    const t = [...r.history.t, ...r.forecast.t], pad = (v) => [...Array(r.history.t.length).fill(null), ...v], line = (name, data, dash) =>
      ({ name, type: "line", showSymbol: false, data, connectNulls: false, lineStyle: dash ? { type: "dashed", width: 1 } : {} });
    el.hidden = false;
    chart(el, { tooltip: { trigger: "axis" }, legend: { top: 0, textStyle: { color: css("--dim") } }, grid: { left: 62, right: 16, top: 30, bottom: 24 },
      xAxis: axis({ type: "category", data: t }), yAxis: axis({ type: "value", scale: true, name: r.unit }),
      series: [line("Lịch sử", r.history.y), line("Dự báo", pad(r.forecast.y)), line("Cận dưới", pad(r.forecast.lo), true), line("Cận trên", pad(r.forecast.hi), true)] });
  } else if (r.series) {
    const { x, ...rest } = r.series;
    el.hidden = false;
    chart(el, { tooltip: { trigger: "axis" }, legend: { top: 0, textStyle: { color: css("--dim") } }, grid: { left: 54, right: 16, top: 30, bottom: 24 },
      xAxis: axis({ type: "category", data: x }), yAxis: axis({ type: "value", scale: true }),
      series: Object.entries(rest).filter(([, v]) => Array.isArray(v)).map(([k, v]) => ({ name: k, type: "line", showSymbol: false, smooth: true, data: v })) });
  }
}

export default async function lab(view) {
  const { chapters } = await api("/lab/specs");
  const specs = chapters.flatMap((c) => c.specs.map((s) => ({ ...s, chapter: c.chapter })));
  const wait = chapters.filter((c) => c.status !== "ok");
  view.innerHTML = `<h2>Model Lab</h2><p class="lead">Thử trực tiếp các mô hình đã huấn luyện, mỗi kết quả là một lần suy luận thật trên máy này.</p>
    ${wait.map((c) => `<div class="chip warn">Ch${c.chapter}: ${esc(c.message)}</div>`).join(" ")}
    <div class="two" style="margin-top:.8rem"><div class="card"><select id="spec" style="width:100%">${specs.map((s, i) => `<option value="${i}">Ch${s.chapter} · ${esc(s.title)}</option>`).join("")}</select><div class="row" id="form"></div><button class="primary" id="go">Dự đoán</button></div>
    <div class="card"><h3>Kết quả</h3><div id="out">${empty("Chưa chạy.")}</div></div></div>`;
  if (!specs.length) return;
  const draw = () => { $("#form").innerHTML = specs[+$("#spec").value].inputs.map(field).join(""); };
  $("#spec").onchange = draw; draw();
  $("#go").onclick = async () => {
    const s = specs[+$("#spec").value], box = $("#out"), btn = $("#go"), data = {};
    view.querySelectorAll("#form [data-n]").forEach((f) => { data[f.dataset.n] = f.type === "number" ? +f.value : f.value; });
    btn.disabled = true; box.innerHTML = '<p class="lead">Đang nạp mô hình...</p>';
    try { result(box, await post(`/lab/predict/${s.chapter}/${s.id}`, data)); } catch (e) { box.innerHTML = errBox(e.message); }
    btn.disabled = false;
  };
}
