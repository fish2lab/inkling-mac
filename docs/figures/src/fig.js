// Tiny SVG helpers for the README figures (no dependencies).
const NS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) k === "text" ? (e.textContent = v) : e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
// Bar with 4px rounded data end, square at the baseline. Vertical when h is the length, horizontal when horiz=true.
function bar(svg, x, y, w, h, fill, horiz = false, r = 4) {
  if (horiz) { r = Math.min(r, w); return el("path", { d: `M${x},${y} h${w - r} q${r},0 ${r},${r} v${h - 2 * r} q0,${r} -${r},${r} h-${w - r} z`, fill }, svg); }
  r = Math.min(r, h);
  return el("path", { d: `M${x},${y + h} v-${h - r} q0,-${r} ${r},-${r} h${w - 2 * r} q${r},0 ${r},${r} v${h - r} z`, fill }, svg);
}
// Grouped vertical bars: groups = labels, series = [{name, color, values, fmt}], 2px gap between bars.
function groupedBars(svg, { W, H, groups, series, ymax, ticks, pad = { l: 44, r: 8, t: 26, b: 34 }, yfmt = (v) => v, barW = 34 }) {
  const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b, gw = iw / groups.length;
  const Y = (v) => pad.t + ih - (v / ymax) * ih;
  for (const t of ticks) {
    el("line", { x1: pad.l, x2: W - pad.r, y1: Y(t), y2: Y(t), stroke: t === 0 ? "#bdb8b1" : "#e4e1dc", "stroke-width": 1 }, svg);
    el("text", { x: pad.l - 10, y: Y(t) + 4, "text-anchor": "end", class: "axis", text: yfmt(t) }, svg);
  }
  groups.forEach((g, i) => {
    const total = series.length * barW + (series.length - 1) * 2, x0 = pad.l + i * gw + (gw - total) / 2;
    series.forEach((s, j) => {
      const v = s.values[i]; if (v == null) return;
      const x = x0 + j * (barW + 2);
      bar(svg, x, Y(v), barW, Y(0) - Y(v), s.color);
      el("text", { x: x + barW / 2, y: Y(v) - 7, "text-anchor": "middle", class: "val", text: (s.fmt || String)(v),
                   style: j === 0 ? "font-weight:500" : "fill:#4b5563" }, svg);
    });
    el("text", { x: pad.l + i * gw + gw / 2, y: H - 10, "text-anchor": "middle", class: "cat", text: g }, svg);
  });
}
