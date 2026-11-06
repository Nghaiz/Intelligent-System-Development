// Đồ thị tri thức 3D: nhấp nút để mở rộng hàng xóm từ Neo4j, tìm nút, và bàn điều khiển Cypher chỉ đọc.
import { $, api, empty, errBox, esc, fmt, palette, post } from "../lib.js";

const EXAMPLES = ["MATCH (n) RETURN labels(n)[0] AS nhan, count(*) AS so ORDER BY so DESC LIMIT 10",
  "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 40",
  "MATCH ()-[r]->() RETURN type(r) AS quan_he, count(*) AS so ORDER BY so DESC LIMIT 10"];

export default async function graph(view) {
  view.innerHTML = `<h2>Đồ thị tri thức Neo4j</h2><p class="lead">Nhấp một nút để nạp hàng xóm trực tiếp từ cơ sở dữ liệu.</p>
    <div class="two"><div id="graph3d"></div><div class="side">
      <div class="card"><h3>Tìm nút</h3><div class="row"><input id="q" placeholder="Ví dụ: Metformin, FPT" style="flex:1"><button id="go">Tìm</button></div><div id="hits"></div></div>
      <div class="card"><h3>Chi tiết</h3><div id="detail">${empty("Chưa chọn nút.")}</div></div>
      <div class="card"><h3>Cypher chỉ đọc</h3><textarea id="cy" class="cypher">${esc(EXAMPLES[0])}</textarea>
        <div class="row"><button class="primary" id="run">Chạy</button><select id="ex">${EXAMPLES.map((e, i) => `<option value="${i}">Mẫu ${i + 1}</option>`).join("")}</select></div><div id="res" class="tbl"></div></div>
    </div></div>`;
  let data;
  try { data = await api("/graph/sample?limit=220"); } catch (e) { $("#graph3d").innerHTML = errBox(e.message); return; }
  // Bỏ nút rời để cụm liên thông chiếm khung nhìn; chúng vẫn tìm được ở ô tìm kiếm.
  const linked = new Set(data.links.flatMap((l) => [l.source, l.target]));
  data.nodes = data.nodes.filter((n) => linked.has(n.id));
  const labels = [...new Set(data.nodes.map((n) => n.label))], cols = palette();
  const color = (n) => { const i = labels.indexOf(n.label); return cols[(i < 0 ? labels.push(n.label) - 1 : i) % cols.length]; };
  const fg = ForceGraph3D()($("#graph3d")).backgroundColor("#070a0f").graphData(data).nodeLabel((n) => `${esc(n.name)} (${esc(n.label)})`)
    .nodeColor(color).nodeRelSize(5).nodeOpacity(.95).linkColor(() => "#4a5a6c").linkOpacity(.22).cooldownTime(5000).onEngineStop(() => fg.zoomToFit(500, 20)).linkLabel((l) => esc(l.type))
    .width($("#graph3d").clientWidth).height($("#graph3d").clientHeight);
  fg.d3Force("charge").strength(-90);
  const merge = (add) => {
    const cur = fg.graphData(), ids = new Set(cur.nodes.map((n) => n.id));
    const key = (l) => `${l.source?.id ?? l.source}|${l.target?.id ?? l.target}|${l.type}`, seen = new Set(cur.links.map(key));
    fg.graphData({ nodes: [...cur.nodes, ...add.nodes.filter((n) => !ids.has(n.id))], links: [...cur.links, ...(add.links ?? []).filter((l) => !seen.has(key(l)))] });
  };
  const show = async (id) => {
    const n = await api(`/graph/node?id=${encodeURIComponent(id)}`);
    $("#detail").innerHTML = `<h3>${esc(n.name)} <span class="chip">${esc(n.label)}</span></h3><div class="tbl" style="max-height:180px"><table>${Object.entries(n.properties ?? {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td style="white-space:normal">${esc(typeof v === "object" ? JSON.stringify(v) : v)}</td></tr>`).join("")}</table></div>`;
    merge(await api(`/graph/expand?id=${encodeURIComponent(id)}`));
  };
  fg.onNodeClick((n) => show(n.id).catch((e) => { $("#detail").innerHTML = errBox(e.message); }));
  $("#go").onclick = async () => {
    try {
      const { nodes } = await api(`/graph/search?q=${encodeURIComponent($("#q").value)}`);
      $("#hits").innerHTML = nodes.length ? nodes.map((n) => `<button data-id="${esc(n.id)}" style="margin:.15rem">${esc(n.name)}</button>`).join("") : empty("Không tìm thấy nút nào.");
    } catch (e) { $("#hits").innerHTML = errBox(e.message); }
  };
  $("#hits").onclick = (e) => { const id = e.target.dataset?.id; if (id) show(id); };
  $("#ex").onchange = (e) => { $("#cy").value = EXAMPLES[+e.target.value]; };
  $("#run").onclick = async () => {
    try {
      const r = await post("/graph/cypher", { q: $("#cy").value });
      $("#res").innerHTML = `<table><tr>${r.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>${r.rows.slice(0, 50).map((row) => `<tr>${r.columns.map((c, i) => { const v = Array.isArray(row) ? row[i] : row[c]; return `<td>${typeof v === "object" ? esc(v?.name ?? v?.type ?? JSON.stringify(v)) : fmt(v)}</td>`; }).join("")}</tr>`).join("")}</table>`;
      if (r.graph?.nodes?.length) merge(r.graph);
    } catch (e) { $("#res").innerHTML = errBox(e.message); }
  };
  return () => { fg.pauseAnimation(); fg._destructor?.(); };
}
