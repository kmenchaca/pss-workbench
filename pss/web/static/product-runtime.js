import { createRecord } from "./workspaces.js";
import { thoughtContent } from "./thought-content.js";
export function createLiveController({
  $,
  s,
  store,
  loader,
  api,
  render,
  renderRecents,
  notice,
}) {
  let polling = false;
  function fromLive(data) {
    const old = store.get(data.session_id),
      contexts = Array.isArray(data.tree?.contexts)
        ? data.tree.contexts
        : Object.values(data.tree?.contexts ?? {});
    const nodes = contexts.map((c, i) => {
      const prior = old?.nodes.find((n) => n.id === c.id),
        detail = null;
      return {
        id: c.id,
        parent: c.parent_id,
        title:
          c.branch_reason ||
          (!c.parent_id ? "Initial direction" : `Alternative ${i}`),
        lens: c.parent_id ? "ALTERNATIVE" : "STARTING POINT",
        summary: (
          c.output ||
          c.last_message ||
          "Working through this direction…"
        ).slice(0, 130),
        ...(detail
          ? thoughtContent(
              detail,
              c.branch_reason || "Initial direction",
              c.branch_reason || "",
            )
          : prior?.full
            ? {
                body: prior.body,
                bodySource: prior.bodySource,
                title: prior.title,
                summary: prior.summary,
              }
            : { body: c.output || "", bodySource: "Preview only" }),
        full: !!detail || !!prior?.full,
        messages: detail?.messages || prior?.messages || [],
        status: c.status,
        terminationReason: c.termination_reason,
        branchBrief: c.branch_reason || "",
        preview: c.output || "",
        needsRefresh:
          prior?.status !== c.status || prior?.preview !== (c.output || ""),
      };
    });
    return {
      ...(old ||
        createRecord(
          {
            id: data.session_id,
            prompt: data.prompt,
            title: data.prompt.slice(0, 80),
            nodes,
          },
          "live",
        )),
      nodes,
      outcome: data.outcome,
      message: data.message || "",
      model: data.model || "",
      provider: data.provider || "",
    };
  }
  async function poll() {
    if (polling) return;
    polling = true;
    const generation = s.generation,
      session = s.live?.session_id;
    try {
      const data = await api("/api/state");
      if (generation !== s.generation && session !== s.live?.session_id) return;
      s.live = data;
      if (data.session_id) {
        let record = fromLive(data);
        await Promise.all(
          record.nodes.map(async (n) => {
            if (
              n.status !== "running" &&
              n.full &&
              !n.needsRefresh &&
              /^(Model output · check|Exploration notes ·)/.test(n.bodySource)
            )
              return;
            const d = await api(
              `/api/context/${encodeURIComponent(n.id)}?session_id=${encodeURIComponent(data.session_id)}`,
            );
            if (d.session_id !== data.session_id)
              throw Error("The server session changed.");
            Object.assign(n, thoughtContent(d, n.title, n.branchBrief), {
              messages: d.messages ?? [],
              full: true,
            });
          }),
        );
        const latest = store.get(record.id);
        if (latest)
          record = {
            ...latest,
            nodes: record.nodes,
            outcome: data.outcome,
            message: data.message || "",
            model: data.model || "",
            provider: data.provider || "",
          };
        store.save(record);
        if (
          s.live?.session_id === data.session_id &&
          s.record?.id === data.session_id &&
          s.record.source === "live"
        ) {
          const selected = s.record.selected;
          s.record = { ...record, selected: selected || record.nodes[0]?.id };
          render();
        }
      }
      if (!$("home").hidden) renderRecents();
      updateConnection();
    } catch (e) {
      notice(
        `Connection interrupted. Saved work is still available. ${e.message}`,
      );
    } finally {
      polling = false;
    }
  }
  async function config() {
    try {
      s.config = await api("/api/config");
    } catch {
      s.config = null;
    }
    updateConnection();
  }
  function updateConnection() {
    const c = s.config;
    const failed = s.live?.outcome === "error";
    $("connection").classList.toggle("connection-error", failed);
    $("connection").textContent = failed
      ? "Last run needs attention"
      : c?.live_enabled
        ? c.local
          ? c.gpu_only
            ? "Local model ready"
            : "Local model configured"
          : "Cloud model configured"
        : "Model setup";
    $("model-note").textContent = c?.live_enabled
      ? `${c.model} · ${c.gpu_only ? "Local inference. No cloud fallback." : c.local ? "Runs on this computer. No provider charges." : "Cloud provider calls may incur charges."}`
      : "Save a question now, or open Model setup to connect a model.";
    $("start-live").disabled = s.busy || !!s.live?.exploration_active;
    $("start-live").textContent = s.busy
      ? "Starting…"
      : s.live?.exploration_active
        ? "Exploration running"
        : "Explore directions";
    $("setup-status").textContent = c
      ? `${c.provider} · ${c.model}${c.live_enabled ? " · configured" : " · disabled"}`
      : "Local PSS server unavailable";
    $("provider-notice").textContent =
      (failed ? `${s.live.message} ` : "") +
      (c?.notice || "Saved examples and browser work remain available.");
    $("setup-instructions").textContent =
      c?.setup || "Restart the local PSS preview and check again.";
    $("run-limits").textContent =
      `Each run: up to 18 requests, ${c?.max_output_tokens_per_call || 768} output tokens per request, a five-minute request window. Partial work is preserved when a limit is reached.`;
  }
  async function command(type, target, payload) {
    try {
      const r = await api("/api/command", { type, target, payload });
      notice(r.message);
      await poll();
      return true;
    } catch (e) {
      notice(e.message);
      return false;
    }
  }

  return { fromLive, poll, config, updateConnection, command };
}
