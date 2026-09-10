/** Focused presentation helpers for the local thinking workbench. */
import { renderThought } from "./thought-content.js";
export function createWorkspaceViews({
  $,
  s,
  el,
  button,
  store,
  openRecord,
  fromLive,
  save,
  render,
  ensureFull,
  notice,
  toggleKept,
}) {
  let recentSignature = "";
  function renderRecents() {
    const pane = $("recents");
    const records = store.list();
    const signature = JSON.stringify(
      records.map((r) => [
        r.id,
        r.title,
        r.updatedAt,
        r.kept.length,
        r.outcome,
      ]),
    );
    if (signature === recentSignature) return;
    recentSignature = signature;
    pane.replaceChildren();
    if (!records.length)
      pane.append(
        el(
          "p",
          "muted",
          "Your questions, kept thoughts and notes will appear here. Stored only in this browser.",
        ),
      );
    records.forEach((r) => {
      const b = button("", () => openRecord(store.get(r.id)), "recent-card");
      b.append(
        el("strong", "", r.title || r.prompt.slice(0, 80)),
        el(
          "small",
          "",
          `${r.source === "example" ? "Example" : r.source === "draft" ? "Saved question" : r.outcome || "Exploration"} · ${r.kept.length} kept · ${new Date(r.updatedAt).toLocaleDateString()}`,
        ),
      );
      pane.append(b);
    });
    if (s.live?.session_id && !records.some((r) => r.id === s.live.session_id))
      pane.prepend(
        button(
          "Open current server exploration",
          () => openRecord(fromLive(s.live)),
          "recent-card",
        ),
      );
  }
  function showTakeaways() {
    const r = s.record;
    $("takeaway").value = r.takeaway;
    $("next-move").value = r.nextMove;
    const pane = $("kept-list");
    pane.replaceChildren();
    if (!r.kept.length)
      pane.append(
        el(
          "p",
          "empty",
          "Nothing kept yet. Close this panel and keep a thought, or write your own takeaway above.",
        ),
      );
    r.kept.forEach((k) => {
      const card = el("section", "kept-card");
      card.append(
        el("h3", "", k.title),
        el("p", "detail-provenance", k.source),
      );
      const details = el("details");
      details.append(
        el("summary", "", "Read kept snapshot"),
        el("div", "detail-body", k.body),
      );
      renderThought(details.querySelector(".detail-body"), k.body, k.title);
      const note = el("textarea");
      note.rows = 2;
      note.maxLength = 5000;
      note.setAttribute("aria-label", `Note on ${k.title}`);
      note.value = r.notes[k.nodeId] || "";
      note.placeholder = "Your note on this thought";
      note.addEventListener("input", () => {
        r.notes[k.nodeId] = note.value;
        save();
      });
      card.append(
        details,
        note,
        button("Remove", () => {
          r.kept = r.kept.filter((x) => x.nodeId !== k.nodeId);
          save();
          showTakeaways();
          render();
        }),
      );
      pane.append(card);
    });
    if (!$("takeaways-dialog").open) $("takeaways-dialog").showModal();
  }
  async function compare() {
    if (!(await ensureFull())) {
      notice("Full text is unavailable. Reconnect before comparing.");
      return;
    }
    const pane = $("comparison");
    pane.replaceChildren();
    [...s.compared].forEach((id) => {
      const n = s.record.nodes.find((n) => n.id === id);
      if (!n) return;
      const col = el("section", "compare-column");
      col.append(
        el("p", "overline", n.lens || "DIRECTION"),
        el("h3", "", n.title),
        el("div", "compare-body", n.body || "No text yet."),
      );
      renderThought(
        col.querySelector(".compare-body"),
        n.body || "No text yet.",
        n.title,
      );
      if (n.tradeoff)
        col.append(el("h4", "", "Tradeoff"), el("p", "", n.tradeoff));
      const keep = button(
        s.record.kept.some((k) => k.nodeId === id)
          ? "Kept"
          : "Keep this thought",
        () => {
          try {
            if (!s.record.kept.some((k) => k.nodeId === id))
              toggleKept(s.record, n);
            save();
            keep.textContent = "Kept";
            render();
          } catch (e) {
            notice(e.message);
          }
        },
      );
      keep.disabled = !n.body || n.full === false;
      col.append(keep);
      pane.append(col);
    });
    $("compare-dialog").showModal();
  }
  return { renderRecents, showTakeaways, compare };
}
