import test from "node:test";
import assert from "node:assert/strict";
import { orbitLayout } from "../../pss/web/static/orbit-layout.js";
test("orbit preserves actual genealogy and stable coordinates", () => {
  const nodes = [
    { id: "a" },
    { id: "b", parent: "a" },
    { id: "c", parent: "b" },
  ];
  const map = orbitLayout(nodes);
  assert.equal(map.edges[0].root, true);
  assert.equal(map.edges[1].from.id, "a");
  assert.equal(map.edges[2].from.id, "b");
  assert.deepEqual(
    map.positions.map((p) => p.depth),
    [0, 1, 2],
  );
  assert.deepEqual(orbitLayout(nodes).positions, map.positions);
});
test("six thought map cards do not overlap and remain inside canvas", () => {
  const { positions, width, height } = orbitLayout(
    Array.from({ length: 6 }, (_, i) => ({ id: String(i) })),
  );
  for (const p of positions) {
    assert.ok(p.x >= 115 && p.x <= width - 115);
    assert.ok(p.y >= 90 && p.y <= height - 90);
    for (const q of positions)
      if (p !== q)
        assert.ok(Math.abs(p.x - q.x) >= 230 || Math.abs(p.y - q.y) >= 180);
  }
});
test("missing parent and cyclic imports are safe", () => {
  assert.equal(
    orbitLayout([{ id: "a", parent: "missing" }]).edges[0].root,
    true,
  );
  assert.ok(
    orbitLayout([
      { id: "a", parent: "b" },
      { id: "b", parent: "a" },
    ]).positions.every((p) => Number.isFinite(p.depth)),
  );
  assert.equal(orbitLayout([]).positions.length, 0);
});
