import { orbitLayout } from "./orbit-layout.js";

export function createOrbitMap({ s, el, button, select, notice, renderPaths }) {
  let zoom = 0.7,
    lastRecord = null;
  function renderMap(pane, nodes) {
    const layout = orbitLayout(s.record.nodes);
    const shown = new Set(nodes.map((n) => n.id));
    const toolbar = el("div", "orbit-toolbar");
    const scaleText = el("span");
    const viewport = el("div", "orbit-viewport");
    viewport.tabIndex = 0;
    viewport.setAttribute(
      "aria-label",
      "Thought map. Scroll to pan; use zoom controls to change scale.",
    );
    const bounds = el("div", "orbit-bounds"),
      scene = el("div", "orbit-scene");
    scene.style.width = `${layout.width}px`;
    scene.style.height = `${layout.height}px`;
    function scale(value) {
      zoom = Math.max(0.3, Math.min(1.4, value));
      scene.style.transform = `scale(${zoom})`;
      bounds.style.width = `${layout.width * zoom}px`;
      bounds.style.height = `${layout.height * zoom}px`;
      scaleText.textContent = `${Math.round(zoom * 100)}%`;
    }
    toolbar.append(
      el("span", "orbit-label", "THOUGHT CONSTELLATION"),
      button("−", () => scale(zoom - 0.1)),
      scaleText,
      button("+", () => scale(zoom + 0.1)),
      button("Fit", () => scale((viewport.clientWidth - 16) / layout.width)),
    );
    toolbar.children[1].setAttribute("aria-label", "Zoom out");
    toolbar.children[3].setAttribute("aria-label", "Zoom in");
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", `0 0 ${layout.width} ${layout.height}`);
    svg.setAttribute("aria-hidden", "true");
    svg.classList.add("orbit-connections");
    layout.edges.forEach(({ from, to, root }) => {
      if (!shown.has(to.id) || (from.id && !shown.has(from.id))) return;
      const path = document.createElementNS(svg.namespaceURI, "path");
      const bend = root ? 0 : to.x > from.x ? 65 : -65;
      path.setAttribute(
        "d",
        `M ${from.x} ${from.y} Q ${(from.x + to.x) / 2 + bend} ${(from.y + to.y) / 2 - 70} ${to.x} ${to.y}`,
      );
      path.setAttribute(
        "class",
        `${root ? "question-link" : "branch-link"} ${to.id === s.record.selected || from.id === s.record.selected ? "connected" : ""}`,
      );
      svg.append(path);
    });
    scene.append(svg);
    const core = el("div", "orbit-core");
    core.style.left = `${layout.center.x}px`;
    core.style.top = `${layout.center.y}px`;
    core.append(
      el("span", "overline", "THE QUESTION"),
      el("strong", "", s.record.prompt),
      el(
        "small",
        "",
        `${s.record.nodes.length} thoughts · ${s.record.kept.length} kept`,
      ),
    );
    scene.append(core);
    nodes.forEach((n, i) => {
      const pos = layout.positions.find((p) => p.id === n.id);
      const kept = s.record.kept.some((k) => k.nodeId === n.id);
      const card = el(
        "div",
        `orbit-node ${n.id === s.record.selected ? "selected" : ""} ${kept ? "is-kept" : ""}`,
      );
      card.style.left = `${pos.x}px`;
      card.style.top = `${pos.y}px`;
      card.style.setProperty(
        "--orbit-accent",
        ["#c89edd", "#ff988f", "#8defba"][i % 3],
      );
      const read = button("", () => select(n.id), "orbit-read");
      read.setAttribute("aria-label", `Read ${n.title}`);
      read.setAttribute("aria-pressed", String(n.id === s.record.selected));
      read.append(
        el(
          "span",
          "card-meta",
          `${String(s.record.nodes.indexOf(n) + 1).padStart(2, "0")} / ${kept ? "KEPT" : n.status || n.lens || "THOUGHT"}`,
        ),
        el("strong", "", n.title),
        el(
          "small",
          "",
          n.parent
            ? `Branch of ${s.record.nodes.find((p) => p.id === n.parent)?.title || "earlier thought"}`
            : "Starting direction",
        ),
      );
      const label = el("label", "compare-check"),
        check = el("input");
      check.type = "checkbox";
      check.checked = s.compared.has(n.id);
      check.setAttribute("aria-label", `Compare ${n.title}`);
      check.onchange = () => {
        if (check.checked && s.compared.size >= 3) {
          check.checked = false;
          notice("Choose up to three thoughts to compare.");
          return;
        }
        if (check.checked) s.compared.add(n.id);
        else s.compared.delete(n.id);
        renderPaths();
      };
      label.append(check, document.createTextNode("Compare"));
      card.append(read, label);
      scene.append(card);
    });
    bounds.append(scene);
    viewport.append(bounds);
    pane.append(
      toolbar,
      viewport,
      el(
        "p",
        "orbit-legend",
        "Dotted threads connect the question to starting thoughts. Solid threads show actual branches. Position is not a ranking.",
      ),
    );
    scale(zoom);
    if (lastRecord !== s.record.id) {
      lastRecord = s.record.id;
      requestAnimationFrame(() => {
        scale(
          viewport.clientWidth < 500
            ? 0.85
            : (viewport.clientWidth - 16) / layout.width,
        );
        if (viewport.clientWidth < 500)
          viewport.scrollLeft =
            (layout.width * zoom - viewport.clientWidth) / 2;
      });
    }
  }
  return { renderMap };
}
