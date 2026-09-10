/** Focused presentation helpers for the local thinking workbench. */
import { renderThought } from "./thought-content.js";
export function createReader({
  $,
  s,
  el,
  button,
  save,
  notice,
  isLive,
  select,
  render,
  showHome,
  renderOrigin,
  command,
  toggleKept,
  continuationPrompt,
  thoughtGuidance,
  visible,
}) {
  function renderReader() {
    const pane = $("detail"),
      n = s.record?.nodes.find((n) => n.id === s.record.selected);
    if (pane.querySelector("textarea:focus")) return;
    pane.replaceChildren();
    if (!n) {
      const failed = s.record?.outcome === "error";
      pane.append(
        el(
          "h2",
          "",
          failed
            ? "This exploration did not get a usable response."
            : "An exploration starts with a commitment.",
        ),
        el(
          "p",
          "empty",
          failed
            ? s.record.message ||
                "Check Model setup, then start another exploration. Your question is still saved."
            : "PSS lets the model choose a direction, then asks it to explore a different one. Thoughts appear here as the run unfolds.",
        ),
      );
      if (failed)
        pane.append(
          button("Edit question and try again", () => {
            const prompt = s.record.prompt;
            showHome(true);
            $("question").value = prompt;
          }),
        );
      return;
    }
    pane.append(
      el("p", "overline", n.lens || n.status || "THOUGHT"),
      el("h2", "", n.title),
    );
    if (n.parent)
      pane.append(
        button(
          `From: ${s.record.nodes.find((x) => x.id === n.parent)?.title || "parent thought"}`,
          () => select(n.parent),
          "parent-link",
        ),
      );
    pane.append(
      el(
        "p",
        "detail-provenance",
        s.record.source === "example"
          ? "Authored example · not a model-generated result"
          : n.full
            ? n.bodySource
            : "Preview only · full text has not loaded",
      ),
      el(
        "div",
        "detail-body",
        n.body ||
          "No text yet. The model may be loading or working through its checkpoint.",
      ),
    );
    const body = pane.querySelector(".detail-body");
    renderThought(body, n.body || "No exploration text yet.", n.title);
    if (n.branchBrief) {
      const brief = el("details", "inline-help");
      brief.append(
        el("summary", "", "Why this direction exists"),
        el("p", "", n.branchBrief),
      );
      pane.append(brief);
    }
    if (n.tradeoff) {
      const t = el("div", "tradeoff");
      t.append(el("h3", "", "THE TRADEOFF"), el("p", "", n.tradeoff));
      pane.append(t);
    }
    const actions = el("div", "detail-actions"),
      keep = button(
        s.record.kept.some((k) => k.nodeId === n.id)
          ? "Remove from takeaways"
          : "Keep this thought",
        () => {
          try {
            toggleKept(s.record, n);
            save();
            render();
          } catch (e) {
            notice(e.message);
          }
        },
        "primary",
      );
    keep.disabled = !n.body || n.full === false;
    actions.append(keep);
    const further = button("Take this further", () => {
      try {
        const p = continuationPrompt(s.record, n);
        showHome(true);
        s.origin = p.origin;
        $("question").value = p.text;
        renderOrigin();
        notice(
          p.excerpt
            ? "A labeled excerpt was carried forward. Edit the new direction before starting."
            : "Add a new direction before starting. The original exploration stays saved.",
        );
      } catch (e) {
        notice(e.message);
      }
    });
    further.disabled = !n.body || n.full === false;
    actions.append(further);
    if (n.next && s.record.nodes.some((x) => x.id === n.next))
      actions.append(
        button("Read authored continuation", () => select(n.next)),
      );
    if (isLive() && s.live.exploration_active && n.status === "running") {
      actions.append(
        button("Nudge direction", () => {
          $("inject-text").value = "";
          $("inject-dialog").showModal();
        }),
        button("Request alternative", () => command("promote", n.id)),
      );
    } else
      actions.append(
        button("Set aside", () => {
          s.record.setAside.push(n.id);
          s.compared.delete(n.id);
          s.record.selected = visible()[0]?.id;
          save();
          render();
        }),
      );
    pane.append(
      actions,
      thoughtGuidance(
        isLive() && s.live.exploration_active && n.status === "running",
      ),
    );
    const note = el("div", "thought-note"),
      label = el("label", "", "Your note"),
      input = el("textarea");
    input.id = "thought-note";
    label.htmlFor = input.id;
    input.rows = 2;
    input.maxLength = 5000;
    input.value = s.record.notes[n.id] || "";
    input.placeholder = "What matters here? What would you change?";
    const record = s.record;
    input.addEventListener("input", () => {
      record.notes[n.id] = input.value;
      save();
    });
    note.append(label, input);
    pane.append(note);
    if (n.messages?.length) {
      const h = el("details", "message-history");
      h.append(
        el(
          "summary",
          "",
          `Inspect conversation (${n.messages.length} messages)`,
        ),
      );
      n.messages.forEach((m) =>
        h.append(el("h3", "", m.role), el("p", "detail-body", m.content)),
      );
      pane.append(h);
    }
  }
  return { renderReader };
}
