/** Browser-local work, separate from the engine's single live session. No credentials. */
export const STORAGE_KEY = "pss.workspaces.v3";
export const MAX_BYTES = 3_000_000;
const copy = (value) => JSON.parse(JSON.stringify(value));
const text = (value, max = 100_000) =>
  typeof value === "string" ? value.slice(0, max) : "";
const sourceText = (value) => {
  if (typeof value === "string" && value.length > 100_000)
    throw Error(
      "A source text exceeds the supported size. It was not silently shortened.",
    );
  return text(value);
};
const safeId = (value) =>
  typeof value === "string" &&
  value.length > 0 &&
  value.length <= 200 &&
  !["__proto__", "constructor", "prototype"].includes(value);

export function normaliseRecord(value) {
  if (
    !value ||
    !safeId(value.id) ||
    !Array.isArray(value.nodes) ||
    value.nodes.length > 256
  )
    throw Error("That is not a supported PSS workspace.");
  const ids = new Set();
  const nodes = value.nodes.map((n) => {
    if (!n || !safeId(n.id) || ids.has(n.id))
      throw Error("The workspace has invalid or duplicate thought IDs.");
    ids.add(n.id);
    const node = { id: n.id, parent: safeId(n.parent) ? n.parent : null };
    for (const key of [
      "title",
      "lens",
      "summary",
      "body",
      "tradeoff",
      "next",
      "preview",
      "angle",
      "status",
      "bodySource",
      "terminationReason",
      "branchBrief",
    ])
      node[key] = text(n[key]);
    node.body = sourceText(n.body);
    node.full = n.full !== false;
    node.messages = Array.isArray(n.messages)
      ? n.messages
          .slice(0, 256)
          .map((m) => ({
            role: text(m.role, 40),
            content: sourceText(m.content),
          }))
      : [];
    return node;
  });
  const notes = Object.create(null);
  for (const id of ids)
    if (Object.hasOwn(value.notes ?? {}, id))
      notes[id] = text(value.notes[id], 5000);
  const kept = (Array.isArray(value.kept) ? value.kept : [])
    .slice(0, 64)
    .filter((k) => k && ids.has(k.nodeId))
    .map((k) => ({
      nodeId: k.nodeId,
      title: text(k.title, 2000),
      body: sourceText(k.body),
      source: text(k.source, 500),
      savedAt: Number(k.savedAt) || Date.now(),
    }));
  return {
    id: value.id,
    title: text(value.title, 1000),
    prompt: text(value.prompt, 6000),
    source: ["example", "live", "draft"].includes(value.source)
      ? value.source
      : "draft",
    nodes,
    kept: [...new Map(kept.map((k) => [k.nodeId, k])).values()],
    notes,
    takeaway: text(value.takeaway, 5000),
    nextMove: text(value.nextMove, 5000),
    selected: ids.has(value.selected)
      ? value.selected
      : (nodes.find((n) => n.parent)?.id ?? nodes[0]?.id ?? null),
    setAside: (Array.isArray(value.setAside) ? value.setAside : []).filter(
      (id) => ids.has(id),
    ),
    createdAt: Number(value.createdAt) || Date.now(),
    updatedAt: Number(value.updatedAt) || Date.now(),
    outcome: text(value.outcome, 100),
    message: text(value.message, 2000),
    model: text(value.model, 200),
    provider: text(value.provider, 200),
    origin:
      value.origin && safeId(value.origin.workspaceId)
        ? {
            workspaceId: value.origin.workspaceId,
            nodeId: text(value.origin.nodeId, 200),
            title: text(value.origin.title, 2000),
          }
        : null,
  };
}

export function createRecord(board, source = "example", now = Date.now()) {
  return normaliseRecord({
    ...board,
    id: source === "example" ? `example:${board.id}` : board.id,
    source,
    selected: board.nodes.find((n) => n.parent)?.id,
    kept: [],
    notes: {},
    setAside: [],
    createdAt: now,
    updatedAt: now,
  });
}

export function createWorkspaceStore(storage) {
  let memory = [],
    warning = "",
    corrupt = false;
  try {
    const raw = storage?.getItem(STORAGE_KEY);
    if (raw) {
      if (raw.length > MAX_BYTES)
        throw Error("Saved work exceeds the local safety limit.");
      const parsed = JSON.parse(raw);
      if (
        parsed.version !== 3 ||
        !Array.isArray(parsed.records) ||
        parsed.records.length > 100
      )
        throw Error("Unsupported saved-work format.");
      memory = parsed.records.map(normaliseRecord);
    }
    if (!storage)
      warning =
        "Browser storage is unavailable. Export your work before leaving.";
  } catch {
    corrupt = true;
    warning =
      "Existing browser data could not be read. It has not been overwritten. New work is in memory; export it before leaving.";
  }
  function refresh() {
    if (!storage || corrupt) return;
    try {
      const raw = storage.getItem(STORAGE_KEY);
      if (!raw) return;
      if (raw.length > MAX_BYTES) throw Error("Oversized storage");
      const parsed = JSON.parse(raw);
      if (
        parsed.version !== 3 ||
        !Array.isArray(parsed.records) ||
        parsed.records.length > 100
      )
        throw Error("Unsupported storage");
      const merged = new Map(memory.map((r) => [r.id, r]));
      for (const value of parsed.records) {
        const record = normaliseRecord(value),
          existing = merged.get(record.id);
        if (!existing || record.updatedAt >= existing.updatedAt)
          merged.set(record.id, record);
      }
      memory = [...merged.values()];
    } catch {
      corrupt = true;
      warning =
        "Browser data changed to an unreadable format. It has not been overwritten. Export your current work.";
    }
  }
  return {
    list: () => {
      refresh();
      return copy(memory).sort((a, b) => b.updatedAt - a.updatedAt);
    },
    get: (id) => {
      refresh();
      const record = memory.find((r) => r.id === id);
      return record ? normaliseRecord(copy(record)) : null;
    },
    warning: () => warning,
    save(value) {
      refresh();
      const record = normaliseRecord(value);
      memory = [record, ...memory.filter((r) => r.id !== record.id)];
      if (!storage || corrupt) return { saved: false, message: warning };
      try {
        const raw = JSON.stringify({ version: 3, records: memory });
        if (raw.length > MAX_BYTES || memory.length > 100)
          throw Error("Local save limit reached.");
        storage.setItem(STORAGE_KEY, raw);
        warning = "";
        return { saved: true, message: "Saved in this browser" };
      } catch {
        warning =
          "Browser storage is full or blocked. Your latest changes are only in this tab. Export before leaving.";
        return { saved: false, message: warning };
      }
    },
  };
}

export function toggleKept(record, node, now = Date.now()) {
  if (!node?.body || node.full === false)
    throw Error("Wait for the full thought before keeping it.");
  const existing = record.kept.some((k) => k.nodeId === node.id);
  record.kept = existing
    ? record.kept.filter((k) => k.nodeId !== node.id)
    : [
        ...record.kept,
        {
          nodeId: node.id,
          title: node.title,
          body: node.body,
          source:
            record.source === "example"
              ? "Authored example · not a model run"
              : node.bodySource || "Model output · may contain errors",
          savedAt: now,
        },
      ];
  record.updatedAt = now;
  return !existing;
}

export function continuationPrompt(record, node) {
  if (!node?.body || node.full === false)
    throw Error("Load the full thought before continuing it.");
  const excerpt = record.prompt.length > 1200 || node.body.length > 3200;
  return {
    text: `Original question:\n${record.prompt.slice(0, 1200)}\n\nIdea to take further: ${node.title.slice(0, 300)}\n\n${excerpt ? "Source excerpt" : "Source thought"}:\n${node.body.slice(0, 3200)}\n\nNew direction:\n`,
    excerpt,
    origin: { workspaceId: record.id, nodeId: node.id, title: node.title },
  };
}

export function exportTakeaways(record) {
  const provenance =
    record.source === "example"
      ? "Authored example; not a model run. Selections and notes are the user’s."
      : record.source === "draft"
        ? "Saved question; no model output yet."
        : "Model output may contain errors. Kept thoughts are snapshots; notes are the user’s.";
  const kept = record.kept.map(
    (k, i) =>
      `## ${i + 1}. ${k.title}\n\n${k.source}\n\n${k.body}\n\n${record.notes[k.nodeId] ? `### My note\n\n${record.notes[k.nodeId]}\n\n` : ""}`,
  );
  return `# ${record.title || "PSS exploration"}\n\n${provenance}\n\n## Question\n\n${record.prompt}\n\n${record.takeaway ? `## What I’m taking away\n\n${record.takeaway}\n\n` : ""}${kept.join("---\n\n")}${!kept.length ? "_No thoughts kept yet._\n\n" : ""}${record.nextMove ? `## My next move\n\n${record.nextMove}\n\n` : ""}`;
}

export function exportWorkspace(record) {
  return JSON.stringify(
    { format: "pss-workspace", version: 3, record: normaliseRecord(record) },
    null,
    2,
  );
}

export function importWorkspace(raw) {
  if (typeof raw !== "string" || raw.length > MAX_BYTES)
    throw Error("Choose a PSS workspace under 3 MB.");
  const value = JSON.parse(raw);
  if (value.format !== "pss-workspace" || value.version !== 3)
    throw Error("Choose a JSON workspace exported by this version of PSS.");
  return normaliseRecord(value.record);
}
