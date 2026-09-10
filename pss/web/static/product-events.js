import { examples } from "./examples.js";
import { wireQuestionIdeas } from "./guidance.js";
import {
  createRecord,
  exportTakeaways,
  exportWorkspace,
  importWorkspace,
} from "./workspaces.js";
export function wireProductEvents({
  $,
  s,
  store,
  el,
  button,
  openRecord,
  showHome,
  config,
  poll,
  renderPaths,
  save,
  render,
  compare,
  showTakeaways,
  command,
  download,
  ensureFull,
  saveDraft,
  updateConnection,
  api,
  loader,
  notice,
  renderOrigin,
}) {
  examples.forEach((ex, i) => {
    const b = button(
      "",
      () => openRecord(store.get(`example:${ex.id}`) || createRecord(ex)),
      "example-card",
    );
    b.append(
      el("span", "overline", `0${i + 1}`),
      el("strong", "", ex.short),
      el("small", "", ex.subtitle),
    );
    $("examples").append(b);
  });
  $("home-button").onclick = () => showHome();
  $("new-question").onclick = () => showHome(true);
  $("library-button").onclick = () => {
    showHome();
    $("recent-section").scrollIntoView({ block: "start" });
  };
  $("save-draft").onclick = saveDraft;
  $("connection").onclick = () => {
    $("setup-dialog").showModal();
    config();
  };
  $("recheck").onclick = () => config().then(poll);
  $("paths-view").onclick = () => {
    s.lineage = false;
    renderPaths();
  };
  $("map-view").onclick = () => {
    s.lineage = true;
    renderPaths();
  };
  $("restore").onclick = () => {
    s.record.setAside = [];
    save();
    render();
  };
  $("compare-button").onclick = compare;
  $("takeaways-button").onclick = showTakeaways;
  document
    .querySelectorAll("[data-close]")
    .forEach((b) => (b.onclick = () => b.closest("dialog").close()));
  document
    .querySelectorAll("[data-command]")
    .forEach((b) => (b.onclick = () => command(b.dataset.command)));
  $("takeaway").oninput = (e) => {
    s.record.takeaway = e.target.value;
    save();
  };
  $("next-move").oninput = (e) => {
    s.record.nextMove = e.target.value;
    save();
  };
  $("export-markdown").onclick = () =>
    download(exportTakeaways(s.record), "pss-takeaways.md", "text/markdown");
  $("export-workspace").onclick = async () => {
    if (!(await ensureFull())) {
      notice("Full workspace export stopped: reconnect to load all thoughts.");
      return;
    }
    download(
      exportWorkspace(s.record),
      "pss-workspace.json",
      "application/json",
    );
  };
  $("import-button").onclick = () => $("import-file").click();
  $("import-file").onchange = async (e) => {
    try {
      const f = e.target.files[0];
      if (!f) return;
      if (f.size > 3000000) throw Error("Choose a workspace under 3 MB.");
      const r = importWorkspace(await f.text());
      if (store.get(r.id)) {
        r.id = `import:${crypto.randomUUID()}`;
        r.title = `${r.title} (imported copy)`;
      }
      store.save(r);
      openRecord(r);
      notice("Workspace imported. Existing work was preserved.");
    } catch (error) {
      notice(error.message);
    } finally {
      e.target.value = "";
    }
  };
  $("run-form").onsubmit = async (e) => {
    e.preventDefault();
    if (s.busy) return;
    $("run-error").textContent = "";
    if (!s.config?.live_enabled) {
      saveDraft();
      $("setup-dialog").showModal();
      return;
    }
    s.busy = true;
    updateConnection();
    try {
      if (s.live?.session_id) {
        const old = store.get(s.live.session_id);
        if (old && old.nodes.some((n) => !n.full)) {
          const previous = s.record;
          openRecord(old);
          if (!(await ensureFull()))
            throw Error(
              "Reconnect to preserve the previous exploration before starting another.",
            );
          s.record = previous;
        }
      }
      const prompt = $("question").value.trim(),
        r = await api("/api/run", {
          prompt,
          config_overrides: { max_contexts: Number($("context-limit").value) },
        });
      loader.clear();
      s.live = {
        session_id: r.session_id,
        prompt,
        outcome: "running",
        exploration_active: true,
      };
      const record = createRecord(
        {
          id: r.session_id,
          title: prompt.slice(0, 80),
          prompt,
          nodes: [],
          origin: s.origin,
        },
        "live",
      );
      openRecord(record);
      notice(
        "Exploring with your configured model. The first response may take a minute.",
      );
      await poll();
    } catch (error) {
      $("run-error").textContent = error.message;
      notice(error.message);
    } finally {
      s.busy = false;
      updateConnection();
    }
  };
  $("inject-form").onsubmit = async (e) => {
    e.preventDefault();
    if (await command("inject", s.record.selected, $("inject-text").value))
      $("inject-dialog").close();
    else
      $("inject-error").textContent =
        "Instruction was not sent. Check the run is still active.";
  };
  wireQuestionIdeas($("question"), () => {
    saveDraft();
    s.draft = null;
    s.origin = null;
    renderOrigin();
    notice(
      "Your previous question was saved. This example is now ready to edit.",
    );
  });
}
