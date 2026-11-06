// Băng đa đạo trình: mỗi tác tử một làn, Event là vạch trên trục thời gian. Dùng ở Hội chẩn (trực tiếp) và Nhật ký (phát lại).
import { esc, num } from "./lib.js";

export const AGENTS = [["graph", "Định tuyến"], ["supervisor", "Điều phối"], ["intake", "Tiếp nhận"], ["risk", "Nguy cơ"], ["retina", "Võng mạc"],
  ["knowledge", "Tri thức"], ["ruleguard", "Luật an toàn"], ["advisor", "Tư vấn"], ["critic", "Phản biện"]];
const KINDS = { start: "bắt đầu", tool: "công cụ", llm: "LLM", decision: "quyết định", end: "kết thúc", error: "lỗi" };

const agentKey = (a) => String(a || "").toLowerCase().replace(/[^a-z]/g, "");
const agentName = (a) => AGENTS.find(([k]) => k === agentKey(a))?.[1] || a;

export function createStrip(host) {
  host.innerHTML = `<div class="strip" role="img" aria-label="Băng sự kiện của các tác tử">${AGENTS.map(([k, v]) =>
    `<div class="lane" data-a="${k}"><span>${v}</span><div class="track"></div></div>`).join("")}</div>
    <ol class="events" aria-live="polite"></ol>`;
  const strip = host.querySelector(".strip"); const list = host.querySelector(".events");
  let events = [];

  const laneOf = (agent) => {
    const key = agentKey(agent);
    let lane = strip.querySelector(`[data-a="${key}"]`);
    if (!lane) { // tác tử ngoài danh sách vẫn phải hiện, không được rơi mất sự kiện
      strip.insertAdjacentHTML("beforeend", `<div class="lane" data-a="${esc(key)}"><span>${esc(agent)}</span><div class="track"></div></div>`);
      lane = strip.lastElementChild;
    }
    return lane;
  };
  const add = (ev) => {
    events.push(ev);
    const lane = laneOf(ev.agent);
    strip.querySelectorAll(".lane.on").forEach((l) => l.classList.remove("on"));
    lane.classList.add("on");
    const tick = document.createElement("button");
    tick.className = "tick"; tick.dataset.k = ev.kind;
    tick.title = `${num(ev.t, 2)} s · ${agentName(ev.agent)} · ${KINDS[ev.kind] || ev.kind}: ${String(ev.detail || "").slice(0, 160)}`;
    tick.setAttribute("aria-label", tick.title);
    tick.dataset.i = events.length - 1;
    tick.addEventListener("click", () => list.children[Number(tick.dataset.i)]?.scrollIntoView({ block: "nearest" }));
    lane.querySelector(".track").append(tick);
    list.insertAdjacentHTML("beforeend", `<li data-k="${esc(ev.kind)}"><time>${num(ev.t, 2)}s</time><b>${esc(agentName(ev.agent))}</b><span>${esc(KINDS[ev.kind] || ev.kind)}${ev.detail ? ` · ${esc(String(ev.detail).slice(0, 220))}` : ""}</span></li>`);
    list.scrollTop = list.scrollHeight;
    strip.querySelectorAll(".tick").forEach((el) => {
      const e = events[Number(el.dataset.i)]; const end = Math.max(events.at(-1).t, 1);
      el.style.left = `${(e.t / end) * 98 + 1}%`;
    });
  };
  return {
    add,
    clear() { events = []; strip.querySelectorAll(".tick").forEach((t) => t.remove()); strip.querySelectorAll(".lane.on").forEach((l) => l.classList.remove("on")); list.innerHTML = ""; },
  };
}
