// Đồ thị con mini cho trang Hội chẩn: bố cục vòng tròn, SVG nhẹ, chữ theo màu chủ đề, dữ liệu thật từ /api/graph.
import { api, ApiError, esc, pendingHtml, seriesColors } from "./lib.js";

const SIZE = { w: 460, h: 240, r: 92, labels: 9 };

export async function miniGraph(host, box, q) {
  box.hidden = false;
  host.innerHTML = '<p class="note">Đang đọc đồ thị...</p>';
  try {
    const g = await api(`/graph?q=${encodeURIComponent(q.slice(0, 200))}&hops=1&limit=36`);
    if (!g.nodes.length) { host.innerHTML = '<p class="note">Không tìm thấy thực thể khớp trong đồ thị.</p>'; return; }
    const seeds = new Set(g.seeds); const colors = seriesColors();
    const labels = [...new Set(g.nodes.map((n) => n.label))];
    const pos = new Map();
    const center = g.nodes.filter((n) => seeds.has(n.id)); const ring = g.nodes.filter((n) => !seeds.has(n.id));
    center.forEach((n, i) => pos.set(n.id, [SIZE.w / 2 + (i - (center.length - 1) / 2) * 34, SIZE.h / 2]));
    ring.forEach((n, i) => {
      const a = (i / ring.length) * Math.PI * 2 - Math.PI / 2;
      pos.set(n.id, [SIZE.w / 2 + Math.cos(a) * SIZE.r * 2, SIZE.h / 2 + Math.sin(a) * SIZE.r]);
    });
    const lines = g.links.map((l) => {
      const a = pos.get(l.source); const b = pos.get(l.target);
      return a && b ? `<line x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}"><title>${esc(l.type)}</title></line>` : "";
    }).join("");
    const dots = g.nodes.map((n, i) => {
      const [x, y] = pos.get(n.id); const seed = seeds.has(n.id);
      const color = colors[Math.min(labels.indexOf(n.label), colors.length - 1)];
      const text = seed || i % Math.ceil(g.nodes.length / SIZE.labels) === 0 ? `<text x="${x + 7}" y="${y + 4}">${esc(n.name.slice(0, 22))}</text>` : "";
      return `<circle cx="${x}" cy="${y}" r="${seed ? 7 : 4.5}" fill="${color}"><title>${esc(n.label)}: ${esc(n.name)}</title></circle>${text}`;
    }).join("");
    host.innerHTML = `<svg class="mini" viewBox="0 0 ${SIZE.w} ${SIZE.h}" role="img" aria-label="Đồ thị con ${g.nodes.length} nút">${lines}${dots}</svg>
      <div class="legend">${labels.slice(0, 6).map((l, i) => `<span><i style="background:${colors[i]}"></i>${esc(l)}</span>`).join("")}</div>`;
  } catch (e) {
    if (!(e instanceof ApiError)) throw e;
    host.innerHTML = pendingHtml(e);
  }
}
