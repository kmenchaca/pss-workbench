import { createOrbitMap } from "./product-map.js";
/** Focused presentation helpers for the local thinking workbench. */
export function createPathView({ $, s, el, button, visible, select, notice }) {
  const { renderMap } = createOrbitMap({
    s,
    el,
    button,
    select,
    notice,
    renderPaths,
  });
  let lastSignature = "";
  function renderPaths() {
    const pane = $("paths");
    const nodes = visible();
    $("restore").hidden = !s.record.setAside.length;
    $("compare-button").disabled = s.compared.size < 2;
    $("compare-button").textContent = `Compare (${s.compared.size})`;
    $("paths-view").setAttribute("aria-pressed", String(!s.lineage));
    $("map-view").setAttribute("aria-pressed", String(s.lineage));
    pane.closest(".work-layout").classList.toggle("map-layout", s.lineage);
    const signature = JSON.stringify([
      s.record.id,
      s.lineage,
      nodes,
      s.record.selected,
      s.record.kept,
      [...s.compared],
    ]);
    if (signature === lastSignature) return;
    lastSignature = signature;
    const oldViewport = pane.querySelector(".orbit-viewport");
    const scroll = oldViewport
      ? [oldViewport.scrollLeft, oldViewport.scrollTop]
      : null;
    const focusLabel = pane.contains(document.activeElement)
      ? document.activeElement.getAttribute("aria-label")
      : null;
    pane.replaceChildren();
    if (s.lineage && nodes.length) {
      renderMap(pane, nodes);
      if (scroll) {
        const v = pane.querySelector(".orbit-viewport");
        v.scrollLeft = scroll[0];
        v.scrollTop = scroll[1];
      }
      if (focusLabel)
        [...pane.querySelectorAll("[aria-label]")]
          .find((e) => e.getAttribute("aria-label") === focusLabel)
          ?.focus({ preventScroll: true });
      return;
    }
    if (!nodes.length)
      pane.append(
        el(
          "p",
          "empty",
          "Waiting for the first thought. Local model loading can take a minute.",
        ),
      );
    nodes.forEach((n, i) => {
      const card = el(
        "div",
        `path-card ${n.id === s.record.selected ? "selected" : ""} ${s.lineage ? "lineage-card" : ""}`,
      );
      let depth = 0,
        parent = n.parent,
        seen = new Set([n.id]);
      while (parent && !seen.has(parent)) {
        seen.add(parent);
        depth++;
        parent = s.record.nodes.find((x) => x.id === parent)?.parent;
      }
      if (s.lineage)
        card.style.setProperty("--depth", `${Math.min(depth, 4) * 14}px`);
      const b = button("", () => select(n.id), "path-select");
      b.setAttribute("aria-label", `Read ${n.title}`);
      b.setAttribute("aria-pressed", String(n.id === s.record.selected));
      b.append(
        el(
          "span",
          "card-meta",
          `${String(i + 1).padStart(2, "0")} · ${n.lens || n.status || "DIRECTION"}${s.record.kept.some((k) => k.nodeId === n.id) ? " · KEPT" : ""}`,
        ),
        el("strong", "", n.title),
        el(
          "small",
          "",
          s.lineage && n.parent
            ? `From: ${s.record.nodes.find((x) => x.id === n.parent)?.title || "parent thought"}`
            : n.summary,
        ),
      );
      card.append(b);
      const label = el("label", "compare-check"),
        check = el("input");
      check.type = "checkbox";
      check.checked = s.compared.has(n.id);
      check.setAttribute("aria-label", `Compare ${n.title}`);
      check.addEventListener("change", () => {
        if (check.checked && s.compared.size >= 3) {
          check.checked = false;
          notice("Choose up to three thoughts to compare.");
          return;
        }
        if (check.checked) s.compared.add(n.id);
        else s.compared.delete(n.id);
        renderPaths();
      });
      label.append(check, document.createTextNode("Compare this thought"));
      card.append(label);
      pane.append(card);
    });
  }
  return { renderPaths };
}
