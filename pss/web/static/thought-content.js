/** Preserve actual output; keep checkpoint protocol out of the reading surface. */
const checkpointKeys = new Set([
  "action",
  "branches",
  "also_continue",
  "output",
  "confidence",
  "reasoning",
]);
export function isCheckpoint(text) {
  try {
    const value = JSON.parse(text);
    return (
      value &&
      !Array.isArray(value) &&
      ["continue", "branch", "terminate"].includes(value.action) &&
      Object.keys(value).every((key) => checkpointKeys.has(key))
    );
  } catch {
    return false;
  }
}

export function thoughtContent(
  detail,
  fallbackTitle = "Initial direction",
  branchBrief = "",
) {
  const final = typeof detail.output === "string" && detail.output.trim();
  const body = final
    ? detail.output
    : (detail.messages || [])
        .filter(
          (m) =>
            m.role === "assistant" &&
            typeof m.content === "string" &&
            m.content.trim() &&
            !isCheckpoint(m.content.trim()),
        )
        .map((m) => m.content)
        .join("\n\n");
  const lines = body
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
  const first = lines[0] || "";
  const heading = /^(?:#{1,6}\s+|\*\*).+/.test(first) && first.length <= 140;
  const title = heading
    ? first
        .replace(/^#{1,6}\s+/, "")
        .replace(/\*\*/g, "")
        .replace(/^title:\s*/i, "")
    : fallbackTitle.split(":")[0].slice(0, 100);
  const summary = (heading ? lines.slice(1).join(" ") : body)
    .replace(/\*\*/g, "")
    .slice(0, 150);
  return {
    body,
    title: title || "Exploration thought",
    summary: summary || "Waiting for exploration text…",
    branchBrief,
    bodySource: final
      ? "Model output · check factual claims"
      : "Exploration notes · checkpoints are in the full conversation",
  };
}

/** Minimal, text-only Markdown. Never executes HTML, embeds or remote links. */
export function renderThought(container, text, omitHeading = "") {
  container.replaceChildren();
  const inline = (parent, value) => {
    for (const part of value.split(/(\*\*[^*\n]+\*\*|`[^`\n]+`)/g)) {
      if (part.startsWith("**") && part.endsWith("**")) {
        const strong = document.createElement("strong");
        strong.textContent = part.slice(2, -2);
        parent.append(strong);
      } else if (part.startsWith("`") && part.endsWith("`")) {
        const code = document.createElement("code");
        code.textContent = part.slice(1, -1);
        parent.append(code);
      } else parent.append(document.createTextNode(part));
    }
  };
  for (const block of text.split(/\n\s*\n/)) {
    if (!block.trim()) continue;
    if (
      block
        .trim()
        .replace(/^#{1,6}\s+/, "")
        .replace(/\*\*/g, "") === omitHeading
    )
      continue;
    const lines = block.split("\n");
    if (lines.every((line) => /^\s*[-*]\s+/.test(line))) {
      const list = document.createElement("ul");
      for (const line of lines) {
        const item = document.createElement("li");
        inline(item, line.replace(/^\s*[-*]\s+/, ""));
        list.append(item);
      }
      container.append(list);
    } else {
      const heading =
        lines.length === 1 &&
        (/^#{1,6}\s+/.test(block) || /^\*\*[^*]+\*\*$/.test(block.trim()));
      const part = document.createElement(heading ? "h3" : "p");
      inline(part, block.replace(/^#{1,6}\s+/, ""));
      container.append(part);
    }
  }
}
