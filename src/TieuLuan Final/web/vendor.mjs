// Chép thư viện npm đã cài vào static/vendor để trang chạy không cần mạng và không cần bước build.
import { cpSync, mkdirSync, rmSync, writeFileSync } from "node:fs";

const OUT = "static/vendor";
const FILES = [
  ["3d-force-graph/dist/3d-force-graph.min.js", "3d-force-graph.min.js"],
  ["echarts/dist/echarts.min.js", "echarts.min.js"],
];
const FONTS = [
  ...["vietnamese", "latin", "latin-ext"].map((s) =>
    [`@fontsource-variable/fraunces/files/fraunces-${s}-wght-normal.woff2`, `fraunces-${s}.woff2`]),
  ...[400, 500, 700].flatMap((w) => ["vietnamese", "latin", "latin-ext"].map((s) =>
    [`@fontsource/be-vietnam-pro/files/be-vietnam-pro-${s}-${w}-normal.woff2`, `bvp-${s}-${w}.woff2`])),
  ...[400, 500].flatMap((w) => ["vietnamese", "latin"].map((s) =>
    [`@fontsource/jetbrains-mono/files/jetbrains-mono-${s}-${w}-normal.woff2`, `jbm-${s}-${w}.woff2`])),
];

rmSync(OUT, { recursive: true, force: true });
mkdirSync(`${OUT}/fonts`, { recursive: true });
for (const [src, dst] of [...FILES, ...FONTS.map(([s, d]) => [s, `fonts/${d}`])]) {
  cpSync(`node_modules/${src}`, `${OUT}/${dst}`);
}
writeFileSync(`${OUT}/README.txt`, "Sinh bởi `npm run vendor`, không sửa tay.\n");
console.log(`vendor: ${FILES.length} thư viện, ${FONTS.length} phông`);
