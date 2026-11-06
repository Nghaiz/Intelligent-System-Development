// Đồ thị tri thức 3D: dữ liệu Neo4j thật, tìm thực thể, lọc nhãn, bấm nút để mở rộng, hiện Cypher đã chạy.
import { $, $$, api, ApiError, css, esc, guarded, num, pendingHtml, reduced, seriesColors } from "../lib.js";

export const lead = "Đồ thị đọc trực tiếp từ Neo4j. Bấm một nút để mở rộng lân cận, câu Cypher vừa chạy hiện ở cột bên.";

export async function mount(el) {
  el.innerHTML = `<form class="row" id="f" style="margin-bottom:12px"><label class="field grow">Tìm thực thể<input type="search" id="q" maxlength="200" placeholder="Ví dụ: metformin, bệnh võng mạc"></label>
      <label class="field">Số bước<select id="hops"><option>1</option><option>2</option><option>3</option></select></label><button class="btn" type="submit">Tìm</button>
      <button class="btn-ghost" type="button" id="reset" style="min-height:42px">Toàn cảnh</button></form>
    <div class="graph-grid"><div class="kg" id="kg"></div>
      <div class="kg-side"><section class="lead"><h2>Nhãn</h2><div class="row" id="labels"></div></section>
        <section class="lead"><h2>Nút đang chọn</h2><p id="sel" class="note">Bấm một nút trong đồ thị.</p></section>
        <section class="lead"><h2>Quy mô hiển thị</h2><p id="scale" class="note">—</p></section>
        <section class="lead"><h2>Cypher đã chạy</h2><pre class="cypher" id="cy">—</pre></section></div></div>`;
  const box = $("#kg", el); const colors = seriesColors();
  let G = null; let data = { nodes: [], links: [] }; let palette = {}; let label = null;

  const colorOf = (n) => palette[n.label] || css("--ink-3");
  const linkKey = (l) => `${l.source.id ?? l.source}|${l.type}|${l.target.id ?? l.target}`;
  const scale = () => { $("#scale", el).textContent = `${num(data.nodes.length)} nút, ${num(data.links.length)} cạnh đang hiển thị`; };
  const paintBg = () => G?.backgroundColor(css("--surface")).linkColor(() => css("--line-2"));

  const ensure = () => {
    if (G) return;
    G = ForceGraph3D()(box).width(box.clientWidth).height(box.clientHeight).showNavInfo(false).cooldownTime(reduced() ? 0 : 6000)
      .nodeLabel((n) => `<div style="font:13px var(--sans)">${esc(n.label)}: ${esc(n.name)}</div>`)
      .nodeColor(colorOf).nodeRelSize(4).nodeOpacity(.95).linkOpacity(.5).linkWidth(.3)
      .onNodeClick(expand).onEngineStop(() => G.zoomToFit(500, 24));
    G.d3Force("charge").strength(-46);
    paintBg();
  };
  const merge = (g) => {
    const seen = new Set(data.nodes.map((n) => n.id)); const links = new Set(data.links.map(linkKey));
    g.nodes.forEach((n) => { if (!seen.has(n.id)) data.nodes.push(n); });
    g.links.forEach((l) => { if (!links.has(linkKey(l))) data.links.push(l); });
    G.graphData(data); scale();
  };
  const replace = (g) => { data = { nodes: g.nodes, links: g.links }; ensure(); G.graphData(data); scale(); $("#cy", el).textContent = g.cypher.join("\n\n"); };
  const fail = (e) => { if (!(e instanceof ApiError)) throw e; $("#sel", el).innerHTML = pendingHtml(e); };

  async function expand(n) {
    $("#sel", el).innerHTML = `<b>${esc(n.name)}</b><br>${esc(n.label)}`;
    try {
      const g = await api(`/graph?id=${encodeURIComponent(n.id)}&hops=1&limit=60`);
      merge(g); $("#cy", el).textContent = g.cypher.join("\n\n");
    } catch (e) { fail(e); }
  }
  const labelsBar = (labels) => {
    palette = {};
    labels.slice(0, colors.length).forEach((l, i) => { palette[l.label] = colors[i]; }); // nhãn thứ 7 trở đi gộp màu trung tính
    $("#labels", el).innerHTML = labels.map((l) => `<button class="chip" data-l="${esc(l.label)}" aria-pressed="${l.label === label}"><i style="background:${palette[l.label] || css("--ink-3")}"></i>${esc(l.label)} <span class="mono">${num(l.count)}</span></button>`).join("");
  };
  const overview = async () => {
    const qs = label ? `?label=${encodeURIComponent(label)}` : "";
    const g = await api(`/graph/overview${qs}`);
    if (!g.nodes.length) throw new ApiError("Đồ thị đang được nạp, Neo4j chưa có nút nào.", 503);
    if (!G) box.innerHTML = "";
    const shown = new Set(g.nodes.map((n) => n.label)); // màu dành cho nhãn đang có trên hình, không dành cho Chunk bị ẩn
    labelsBar([...g.labels.filter((l) => shown.has(l.label)), ...g.labels.filter((l) => !shown.has(l.label))]); replace(g);
  };
  el.addEventListener("click", (e) => {
    const c = e.target.closest("[data-l]");
    if (!c) return;
    label = label === c.dataset.l ? null : c.dataset.l;
    $$("[data-l]", el).forEach((x) => x.setAttribute("aria-pressed", x.dataset.l === label));
    overview().catch(fail);
  });
  $("#f", el).addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = $("#q", el).value.trim();
    if (!q) return;
    try {
      const g = await api(`/graph?q=${encodeURIComponent(q)}&hops=${$("#hops", el).value}&limit=150`);
      if (!g.nodes.length) { $("#sel", el).textContent = "Không tìm thấy thực thể khớp."; $("#cy", el).textContent = g.cypher.join("\n\n"); return; }
      replace(g);
    } catch (err) { fail(err); }
  });
  $("#reset", el).addEventListener("click", () => { label = null; overview().catch(fail); });
  const onTheme = () => paintBg();
  const ro = new ResizeObserver(() => G?.width(box.clientWidth).height(box.clientHeight));
  ro.observe(box); addEventListener("dnx-theme", onTheme);
  const stop = guarded(box, overview);
  return () => { stop(); ro.disconnect(); removeEventListener("dnx-theme", onTheme); G?._destructor?.(); };
}
