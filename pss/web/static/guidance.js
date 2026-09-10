import { element as el } from "./graph.js";

/** Native disclosures work with keyboard and touch; nothing relies on hover. */
export function thoughtGuidance(running = false) {
  const details = el("details", "inline-help thought-help");
  details.append(
    el(
      "summary",
      "",
      running
        ? "What do keep, continue, and live guidance do?"
        : "What do keep, continue, and set aside do?",
    ),
  );
  const items = [
    [
      "Keep this thought",
      "Saves a snapshot in Your takeaways. It does not tell the model that this is the best answer. Add your own note to explain why you kept it.",
    ],
    [
      "Take this further",
      "Copies this idea into a new, editable question. Add the direction you want to investigate, then explicitly start the new exploration. The original stays saved.",
    ],
    running
      ? [
          "Nudge / request alternative",
          "A nudge gives the running thought an instruction. Request alternative asks the engine to branch at a checkpoint; it does not generate an instant answer. Both stay within the current run’s limits.",
        ]
      : [
          "Set aside",
          "Hides this thought from your current view. It does not delete the thought or a kept snapshot. Show set-aside thoughts brings it back.",
        ],
  ];
  for (const [label, text] of items) {
    const p = el("p");
    p.append(el("strong", "", `${label}. `), document.createTextNode(text));
    details.append(p);
  }
  return details;
}

export function wireQuestionIdeas(question, beforeReplace) {
  document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", () => {
      // Never silently replace a question the user has already started writing.
      if (question.value.trim()) {
        beforeReplace();
      }
      question.value = button.dataset.question;
      question.focus();
    });
  });
}
