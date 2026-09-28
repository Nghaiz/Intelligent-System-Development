// Giao diện hai tab gọi REST API của app.py và vẽ bằng Chart.js.
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)});
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

function fillModels(select, meta, dataset, fw) {
  select.innerHTML = Object.entries(meta.models).map(([k, v]) =>
    `<option value="${k}" ${meta.best[`${dataset}_${fw}`] === k ? "selected" : ""}>${v}` +
    `${meta.best[`${dataset}_${fw}`] === k ? " (tốt nhất trên val)" : ""}</option>`).join("");
}

let stockChart, churnChart;

async function initStock(meta) {
  const h = await api("/api/stock/history");
  stockChart = new Chart(document.getElementById("s-chart"), {
    type: "line",
    data: {labels: h.dates, datasets: [
      {label: "Giá đóng cửa (USD)", data: h.close, borderColor: "#0072B2", pointRadius: 0, borderWidth: 2},
      {label: "MA20", data: h.ma20, borderColor: "#E69F00", pointRadius: 0, borderWidth: 1.5, borderDash: [5, 4]}]},
    options: {plugins: {title: {display: true, text: "100 phiên gần nhất"}}}});
  const fw = document.getElementById("s-fw"), model = document.getElementById("s-model");
  fillModels(model, meta, "amzn", fw.value);
  fw.onchange = () => fillModels(model, meta, "amzn", fw.value);
  document.getElementById("s-run").onclick = async () => {
    const r = await api("/api/stock/predict", {framework: fw.value, model: model.value});
    const up = r.change_pct >= 0;
    document.getElementById("s-out").innerHTML =
      `Giá đóng cửa ${r.last_date}: <b>${r.last_close}</b> USD<br>` +
      `Dự báo phiên kế tiếp: <span class="so">${r.pred_close}</span> USD ` +
      `<span class="${up ? "thap" : "cao"}">(${up ? "+" : ""}${r.change_pct} %)</span><br>` +
      `${r.model_label} · ${fw.value === "torch" ? "PyTorch" : "Keras"} · ${r.latency_ms} ms`;
    const ds = stockChart.data;
    ds.labels = [...ds.labels.slice(0, 100), "phiên kế tiếp"];
    ds.datasets[2] = {label: "Dự báo", data: [...Array(99).fill(null), r.last_close, r.pred_close],
                      borderColor: "#C0392B", pointRadius: 4, borderWidth: 2};
    stockChart.update();
  };
}

async function initChurn(meta) {
  const preset = document.getElementById("c-preset");
  preset.innerHTML = Object.entries(meta.presets).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
  const fw = document.getElementById("c-fw"), model = document.getElementById("c-model");
  fillModels(model, meta, "kkbox", fw.value);
  fw.onchange = () => fillModels(model, meta, "kkbox", fw.value);
  const secs = meta.features.indexOf("total_secs"), unq = meta.features.indexOf("num_unq");
  document.getElementById("c-run").onclick = async () => {
    const r = await api("/api/churn/predict", {preset: preset.value, framework: fw.value, model: model.value});
    const cls = r.risk === "cao" ? "cao" : r.risk === "thấp" ? "thap" : "";
    document.getElementById("c-out").innerHTML =
      `Xác suất churn: <span class="so">${(100 * r.prob).toFixed(1)} %</span><br>` +
      `Mức rủi ro: <b class="${cls}">${r.risk}</b> (ngưỡng từ val: ${(100 * r.threshold).toFixed(1)} %)<br>` +
      `${r.model_label} · ${fw.value === "torch" ? "PyTorch" : "Keras"} · ${r.latency_ms} ms`;
    const days = r.sequence.map((_, i) => i + 1);
    const data = {labels: days, datasets: [
      {label: "log(1 + giây nghe)", data: r.sequence.map(d => d[secs]), borderColor: "#0072B2", yAxisID: "y"},
      {label: "log(1 + số bài khác nhau)", data: r.sequence.map(d => d[unq]), borderColor: "#009E73", yAxisID: "y"}]};
    if (churnChart) churnChart.destroy();
    churnChart = new Chart(document.getElementById("c-chart"), {type: "line", data,
      options: {plugins: {title: {display: true, text: "Hành vi 31 ngày tháng 3/2017"}}}});
  };
}

document.querySelectorAll("nav button").forEach(b => b.onclick = () => {
  document.querySelectorAll("nav button, .tab").forEach(e => e.classList.remove("active"));
  b.classList.add("active");
  document.getElementById(b.dataset.tab).classList.add("active");
});

api("/api/meta").then(meta => { initStock(meta); initChurn(meta); });
