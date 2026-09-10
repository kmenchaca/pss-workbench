/** Deterministic placement: selecting a thought never moves its neighbors. */
export function orbitLayout(nodes) {
  const outerRing = Math.max(0, Math.ceil(nodes.length / 8) - 1);
  const width = Math.max(1120, 2 * (530 + outerRing * 300)),
    height = Math.max(820, 2 * (410 + outerRing * 220));
  const center = { x: width / 2, y: height / 2 };
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const depth = (n) => {
    const seen = new Set([n.id]);
    let d = 0;
    while (n.parent && byId.has(n.parent) && !seen.has(n.parent)) {
      seen.add(n.parent);
      d++;
      n = byId.get(n.parent);
    }
    return d;
  };
  const positions = nodes.map((n, i) => {
    const ring = Math.floor(i / 8),
      count = Math.min(8, nodes.length - ring * 8);
    const angle = -Math.PI / 2 + ((i % 8) * 2 * Math.PI) / count;
    return {
      id: n.id,
      depth: depth(n),
      x: center.x + Math.cos(angle) * (390 + ring * 300),
      y: center.y + Math.sin(angle) * (270 + ring * 220),
    };
  });
  const points = new Map(positions.map((p) => [p.id, p]));
  const edges = nodes.map((n) => ({
    from: points.get(n.parent) || center,
    to: points.get(n.id),
    root: !points.has(n.parent),
  }));
  return { width, height, center, positions, edges };
}
