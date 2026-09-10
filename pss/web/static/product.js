import { createLiveController } from "./product-runtime.js";
import { thoughtContent } from "./thought-content.js";
import { wireProductEvents } from "./product-events.js";
import { createReader } from "./product-reader.js";
import { createPathView } from "./product-paths.js";
import { createWorkspaceViews } from "./product-library.js";
import { examples } from "./examples.js";
import { thoughtGuidance, wireQuestionIdeas } from "./guidance.js";
import { element as el } from "./graph.js";
import { createDetailLoader, liveBody } from "./live-detail.js";
import {
  createWorkspaceStore,
  createRecord,
  toggleKept,
  continuationPrompt,
  exportTakeaways,
  exportWorkspace,
  importWorkspace,
} from "./workspaces.js";
const $ = (id) => document.getElementById(id);
let storage;
try {
  storage = localStorage;
} catch {}
const store = createWorkspaceStore(storage);
const s = {
  record: null,
  live: null,
  config: null,
  generation: 0,
  compared: new Set(),
  lineage: false,
  busy: false,
  draft: null,
  origin: null,
};
const isLive = () =>
  s.record?.source === "live" && s.record.id === s.live?.session_id;
const loader = createDetailLoader({
  getScope: () => ({
    mode: isLive() ? "live" : "saved",
    sessionId: s.record?.id,
    generation: s.generation,
  }),
  fetchDetail: (id) =>
    api(
      `/api/context/${encodeURIComponent(id)}?session_id=${encodeURIComponent(s.record.id)}`,
    ),
  onDetail: (id, d) => {
    const n = s.record?.nodes.find((n) => n.id === id);
    if (n) {
      Object.assign(n, thoughtContent(d, n.title, n.branchBrief), {
        messages: d.messages ?? [],
        full: true,
      });
      save(false);
      if (s.record.selected === id) renderReader();
    }
  },
  onError: (e) => notice(e.message),
});
const { fromLive, poll, config, updateConnection, command } =
  createLiveController({
    $,
    s,
    store,
    loader,
    api,
    render,
    renderRecents: () => renderRecents(),
    notice,
  });
const { renderReader } = createReader({
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
});
const { renderPaths } = createPathView({
  $,
  s,
  el,
  button,
  visible,
  select,
  notice,
});
const { renderRecents, showTakeaways, compare } = createWorkspaceViews({
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
});
function button(text, fn, cls = "") {
  const b = el("button", cls, text);
  b.type = "button";
  b.addEventListener("click", fn);
  return b;
}
function notice(text) {
  $("notice").textContent = text;
}
function save(touch = true) {
  if (!s.record) return;
  if (touch) s.record.updatedAt = Date.now();
  const result = store.save(s.record);
  $("save-status").textContent = result.message;
}
async function api(path, body) {
  const r = await fetch(path, {
    method: body ? "POST" : "GET",
    headers: body
      ? {
          "Content-Type": "application/json",
          "x-pss-token": s.config?.csrf_token ?? "",
        }
      : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  let d;
  try {
    d = await r.json();
  } catch {
    throw Error("The local server returned an unreadable response.");
  }
  if (!r.ok)
    throw Error(
      typeof d.detail === "string" ? d.detail : `Request failed (${r.status}).`,
    );
  return d;
}
function showHome(clear = false) {
  s.generation++;
  $("home").hidden = false;
  $("workbench").hidden = true;
  if (clear) {
    s.draft = null;
    s.origin = null;
    $("question").value = "";
  }
  renderOrigin();
  renderRecents();
  $("question").focus();
}
function renderOrigin() {
  $("origin").hidden = !s.origin;
  $("origin").textContent = s.origin
    ? `Continuing “${s.origin.title}” · starts a new exploration, preserving the original.`
    : "";
}
function openRecord(record) {
  s.generation++;
  s.record = record;
  s.compared.clear();
  if (record.source === "draft") {
    s.draft = record.id;
    s.origin = record.origin;
    $("question").value = record.prompt;
    showHome();
    return;
  }
  $("home").hidden = true;
  $("workbench").hidden = false;
  save();
  render();
  if (isLive()) refreshDetails();
}
function visible() {
  return s.record.nodes.filter((n) => !s.record.setAside.includes(n.id));
}
function render() {
  if (!s.record) return;
  const r = s.record;
  $("prompt").textContent = r.prompt;
  $("provenance").textContent =
    r.source === "example"
      ? "AUTHORED EXAMPLE · NO MODEL CALLS"
      : isLive()
        ? s.config?.gpu_only
          ? "GPU EXPLORATION · RTX 3090"
          : "MODEL EXPLORATION"
        : "SAVED EXPLORATION";
  $("run-status").textContent = isLive()
    ? `${s.live.outcome} · ${s.live.request_count || 0} / 18 requests · ${s.live.message || "Exploring. You can read and keep thoughts as they arrive."}`
    : `${r.nodes.length} thoughts · ${r.kept.length} kept · ${r.source === "example" ? "Explore this example or take a thought into your own question." : r.message || "Full text saved where available; model output can be wrong."}`;
  $("kept-count").textContent = r.kept.length;
  $("live-controls").hidden = !isLive() || !s.live.exploration_active;
  document.querySelector('[data-command="pause"]').disabled = !!s.live?.paused;
  document.querySelector('[data-command="resume"]').disabled = !s.live?.paused;
  renderPaths();
  renderReader();
}
async function select(id) {
  s.record.selected = id;
  save();
  renderPaths();
  renderReader();
  if (isLive()) await loader.load(id);
}
async function ensureFull() {
  if (!isLive()) return true;
  const generation = s.generation,
    loaded = await Promise.all(s.record.nodes.map((n) => loader.load(n.id)));
  return generation === s.generation && loaded.every(Boolean);
}
async function refreshDetails() {
  if (isLive()) await Promise.all(s.record.nodes.map((n) => loader.load(n.id)));
}
function saveDraft() {
  const prompt = $("question").value.trim();
  if (!prompt) {
    $("question").focus();
    return;
  }
  const r = createRecord(
    {
      id: s.draft || `draft:${crypto.randomUUID()}`,
      prompt,
      title: prompt.slice(0, 80),
      nodes: [],
      origin: s.origin,
    },
    "draft",
  );
  s.draft = r.id;
  const result = store.save(r);
  $("save-status").textContent = result.message;
  renderRecents();
  notice("Question saved. You can return to it from Your explorations.");
}
function download(text, name, type) {
  const url = URL.createObjectURL(new Blob([text], { type })),
    a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
wireProductEvents({
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
});
$("save-status").textContent = store.warning();
renderRecents();
config().then(poll);
setInterval(() => {
  if (s.config) poll();
}, 2200);
