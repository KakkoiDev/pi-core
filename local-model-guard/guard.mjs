// local-model-guard: keeps local models on track in pi, using AI Core's turn policies.
//
// 1. Request fixes  (before_provider_request): e.g. Gemma 4 on Ollama ignores "thinking off"
//    unless the request carries reasoning_effort "none".
// 2. Tool-call repair (message_end): a tool call written as text becomes a real tool call that
//    pi runs through its normal tool path.
// 3. Nudges (agent_before_settle): a turn that ends with thinking only, nothing, a stated intent
//    without a call, or a cut-off answer gets one short follow-up, a bounded number of times per
//    request.
// 4. Stall watch (message_start/update/end): a request with no stream activity at all for too
//    long is aborted and asked to continue, instead of hanging until an outer timeout.
// 5. Reasoning-loop watch (message_update): the same three thinking lines seen twice stops the
//    turn and asks the model to act instead of re-deriving (pi-local-harness, lesson L-007).
// 6. Template tokens (message_end): chat-template tokens leaked into the answer (`<|...|>`,
//    `<channel|>`) are removed, so later turns do not imitate them (pi-local-harness).
// 7. Broken tool JSON (agent_before_settle): when the server could not parse the model's tool
//    call, the model is asked once for a short valid call, unless a tool already changed something
//    in this request, so no write is replayed (pi-local-harness task 370).
// 8. Early compaction (agent_end): after a turn that passes a share of the context window, the
//    session is compacted once, keeping the open work (pi-local-harness task 116).
// 10. Cut-off replies (agent_before_settle): a stream that ends without a finish reason and
//    without a usable answer or tool call (gpt-oss on Ollama does this after a burst of
//    reasoning) is continued at once with a short note ("call a tool with valid JSON, or answer"),
//    instead of pi's retry after an exponential backoff (2 s, 4 s, 8 s) of the same request.
//    Bounded by PI_GUARD_CUTOFF_RETRIES (default 3) per request; PI_GUARD_CUTOFF=off disables it.
//    From rich-ai-chat's chat-attempt layer.
// 9. Model profiles, switched on by model id (see PROFILES). gpt-oss: an opt-in reasoning effort when pi's
//    thinking is off (PI_GUARD_GPT_OSS_EFFORT; pi sends none, so Ollama uses medium), the working folder's files in the system
//    prompt (it opened every agentic-v1 task by reading "" as a file), and a hint when read is given
//    a folder.
import { appendFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { classifyTurn, turnRecovery, parseTextToolCalls, createStallWatch, createReasoningWatch, requestAdjustments } from "@kakkoidev/ai-core";

const STALL_NUDGE = "Your last response stopped producing output and was cancelled. Continue the task: call a tool to make progress, or reply with your final answer.";
const LOOP_NUDGE = "Your reasoning repeated the same lines and was stopped. Do not work it out again: take the next concrete step now with a tool call, another way if the last one failed, or reply with your final answer.";
const TOOL_JSON_NUDGE = "The server returned incomplete tool JSON or another JSON syntax error; that call did not run. Submit a short valid tool call. Read files for facts you need; keep large content out of a single call.";
const COMPACT_NOTE = "Keep the open work: what the person asked for, decisions made, files touched, and what is left to do. Drop finished detail.";
const TEMPLATE_TOKENS = /<channel\|>|<\|[^<>\n]*\|>/g;
const BROKEN_TOOL_JSON = /error parsing tool call:[\s\S]*(?:unexpected end of JSON input|invalid character\b|invalid escape\b|invalid unicode\b)/i;
const CUT_OFF = /without finish_reason/i;
const CUT_OFF_NUDGE = "Your last reply was cut off before any answer or tool call. Continue now: if you meant to use a tool, " +
  "call one of the available tools with valid JSON arguments; otherwise give the answer directly. Keep the reasoning short.";
const READS = new Set(["read", "grep", "find", "ls"]);
const ENTRY = "local-model-guard";

/** Settings switched on by model id. `effort`: reasoning_effort sent when pi's thinking is off and the request has none. */
export const PROFILES = [
  {
    id: "gpt-oss",
    match: model => /^gpt-oss(?::|-|$)/i.test(model ?? ""),
    why: "gpt-oss on Ollama (agentic-v1, 2026-11-14): thinks at Ollama's default medium effort when pi's thinking is off; first call of every task was read on a folder; about twice qwen's tool calls.",
    // No effort by default: "low" made agentic-v1 2.4x faster with the same solved count, but lost on the held-out
    // agentic-v2 (9/10 -> 5/10 solved) and coding-v2 (12/12 at medium -> 8/12). Set PI_GUARD_GPT_OSS_EFFORT to opt in.
    effort: null,
    listFiles: true,
    folderHint: true,
  },
];
export const profileFor = (model, opts = {}) => opts.profiles === false ? null : PROFILES.find(p => p.match(model)) || null;

const SKIP = new Set([".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".idea", ".vscode", "dist", "build"]);
/** The folder's files as an indented list (hidden and build folders skipped), at most `max` lines. */
export function listFolder(dir, { max = 60, depth = 3 } = {}) {
  const lines = [];
  let more = 0;
  const walk = (path, level) => {
    let entries;
    try { entries = readdirSync(path, { withFileTypes: true }); } catch { return; }
    entries = entries.filter(e => !e.name.startsWith(".") && !SKIP.has(e.name)).sort((a, b) => a.name.localeCompare(b.name));
    for (const e of entries) {
      if (lines.length >= max) { more++; continue; }
      lines.push("  ".repeat(level) + e.name + (e.isDirectory() ? "/" : ""));
      if (e.isDirectory() && level + 1 < depth) walk(join(path, e.name), level + 1);
    }
  };
  walk(dir, 0);
  return lines.length ? lines.join("\n") + (more ? `\n… and ${more} more` : "") : "";
}
export const folderSection = listing => `## Working folder\n\nFiles at the start of this request:\n${listing}\n\nread takes a file path, not a folder; use bash (ls, find, grep) to look further.`;

export function options(env = process.env) {
  const num = (name, fallback) => {
    const value = Number(env[name]);
    return Number.isFinite(value) && value >= 0 ? value : fallback;
  };
  return {
    enabled: env.PI_GUARD !== "off",
    maxNudges: num("PI_GUARD_MAX_NUDGES", 2),
    idleMs: num("PI_GUARD_IDLE_SECONDS", 120) * 1000,
    totalMs: num("PI_GUARD_TOTAL_SECONDS", 1200) * 1000,
    log: env.PI_GUARD_LOG || "",
    loopWatch: env.PI_GUARD_LOOP_WATCH !== "off",
    stripTokens: env.PI_GUARD_STRIP_TOKENS !== "off",
    toolJsonRetry: env.PI_GUARD_TOOL_JSON_RETRY !== "off",
    cutoff: env.PI_GUARD_CUTOFF !== "off",
    cutoffRetries: num("PI_GUARD_CUTOFF_RETRIES", 3),
    compactAt: Math.min(num("PI_GUARD_COMPACT_AT", 0.6), 1),
    profiles: env.PI_GUARD_PROFILE !== "off",
    // reasoning_effort for gpt-oss when pi's thinking is off: low, medium, high, "default" to send none,
    // or "auto": low until a tool runs in the request, then default
    gptOssEffort: env.PI_GUARD_GPT_OSS_EFFORT || "",
    // the working folder's files in the system prompt: "auto" (the model's profile decides), "on" or "off"
    listFiles: ["on", "off"].includes(env.PI_GUARD_LIST_FILES) ? env.PI_GUARD_LIST_FILES : "auto",
  };
}

const textOf = message => (message.content || []).filter(c => c?.type === "text").map(c => c.text).join("\n");
const thinkingOf = message => (message.content || []).filter(c => c?.type === "thinking").map(c => c.thinking || "").join("\n");
const callsOf = message => (message.content || []).filter(c => c?.type === "toolCall");

export function installLocalModelGuard(pi, opts = options(), { now = () => Date.now(), timers = globalThis } = {}) {
  const stats = { adjusted: 0, repaired: 0, nudged: 0, stalls: 0, loops: 0, stripped: 0, compacted: 0, stopped: 0, listed: 0, hinted: 0, cutoffs: 0 };
  let cutoff = false, cutoffRetries = 0, worked = false;
  let nudges = 0, lastKind = null, resending = false, stalled = null, watch = null, timer = null, seq = 0;
  let thoughts = createReasoningWatch(), brokenJson = false, mutated = false, compacted = false;

  const record = (event, ctx, data = {}) => {
    const entry = { event, at: new Date(now()).toISOString(), model: ctx?.model?.id, ...data };
    try { pi.appendEntry?.(ENTRY, entry); } catch {}
    if (opts.log) try { appendFileSync(opts.log, JSON.stringify(entry) + "\n"); } catch {}
    if (ctx?.hasUI && event !== "adjusted") ctx.ui?.notify?.(`Local model guard: ${data.reason || event}`, "info");
  };
  const clearTimer = () => { if (timer) timers.clearTimeout(timer); timer = null; };

  if (!opts.enabled) return { stats };

  // A new request from the person resets the nudge budget; the guard's own follow-ups do not.
  pi.on("before_agent_start", () => {
    if (resending) resending = false;
    else { nudges = 0; mutated = false; cutoffRetries = 0; worked = false; }
    lastKind = null;
  });
  pi.on("agent_start", () => { thoughts = createReasoningWatch(); brokenJson = false; compacted = false; });
  pi.on("tool_execution_start", event => { worked = true; if (!READS.has(event?.toolName)) mutated = true; });

  pi.on("before_provider_request", (event, ctx) => {
    if (!event?.payload || typeof event.payload !== "object" || Array.isArray(event.payload)) return undefined;
    const thinking = ctx?.thinkingLevel ?? pi.getThinkingLevel?.();
    const fix = requestAdjustments({ provider: ctx?.model?.provider, model: ctx?.model?.id, thinking }) || { patch: {}, rules: [] };
    const profile = profileFor(ctx?.model?.id, opts);
    let effort = profile && (opts.gptOssEffort || profile.effort);
    // auto: quick replies think briefly; once a tool has run in this request it is real work,
    // and the model gets its default effort (low lost 9/10 -> 5/10 on held-out agent tasks).
    if (effort === "auto") effort = worked ? "default" : "low";
    if (effort && effort !== "default" && (thinking === "off" || thinking === false) && event.payload.reasoning_effort === undefined
        && fix.patch.reasoning_effort === undefined) {
      fix.patch = { ...fix.patch, reasoning_effort: effort };
      fix.rules = [...fix.rules, `${profile.id}-effort-${effort}`];
    }
    if (!fix.rules.length) return undefined;
    stats.adjusted++;
    record("adjusted", ctx, { rules: fix.rules });
    return { ...event.payload, ...fix.patch };
  });

  pi.on("before_agent_start", (event, ctx) => {
    const profile = profileFor(ctx?.model?.id, opts);
    if (opts.listFiles === "off" || (opts.listFiles === "auto" && !profile?.listFiles) || typeof event?.systemPrompt !== "string") return undefined;
    const listing = listFolder(ctx?.cwd || process.cwd());
    if (!listing) return undefined;
    stats.listed++;
    record("listed", ctx, { lines: listing.split("\n").length });
    return { systemPrompt: `${event.systemPrompt}\n\n${folderSection(listing)}` };
  });

  pi.on("tool_result", (event, ctx) => {
    if (!profileFor(ctx?.model?.id, opts)?.folderHint || event?.toolName !== "read" || !event.isError) return undefined;
    const text = (event.content || []).filter(c => c?.type === "text").map(c => c.text || "").join("\n");
    if (!/EISDIR|is a directory/i.test(text)) return undefined;
    const path = String(event.input?.path ?? "") || ".";
    stats.hinted++;
    record("hinted", ctx, { reason: "read on a folder", path });
    return { content: [...(event.content || []), { type: "text", text: `\n[local-model-guard] ${path} is a folder. List it with bash (ls ${path === "." ? "" : path}), then read a file in it.`.replace("(ls )", "(ls)") }] };
  });

  pi.on("message_start", (event, ctx) => {
    if (event?.message?.role !== "assistant" || !opts.idleMs) return;
    clearTimer();
    stalled = null;
    watch = createStallWatch({ now, idleMs: opts.idleMs, totalMs: opts.totalMs || 1e15 });
    const check = () => {
      const reason = watch?.reason();
      if (!reason) { timer = timers.setTimeout(check, Math.max(1000, watch.remainingMs())); return; }
      timer = null; stalled = "stalled"; stats.stalls++;
      record("stall", ctx, { reason });
      try { ctx?.abort?.(); } catch {}
    };
    timer = timers.setTimeout(check, Math.max(1000, watch.remainingMs()));
  });
  pi.on("message_update", (event, ctx) => {
    watch?.observe();
    const stream = event?.assistantMessageEvent;
    if (!opts.loopWatch || stalled || stream?.type !== "thinking_delta" || typeof stream.delta !== "string") return;
    if (!thoughts(stream.delta)) return;
    clearTimer(); watch = null;
    stalled = "reasoning-loop"; stats.loops++;
    record("reasoning-loop", ctx, { reason: "reasoning-loop" });
    try { ctx?.abort?.(); } catch {}
  });
  pi.on("provider_stream_event", () => { watch?.observe(); });

  pi.on("message_end", (event, ctx) => {
    let message = event?.message, cleaned;
    if (message?.role !== "assistant") return undefined;
    clearTimer(); watch = null;
    brokenJson = message.stopReason === "error" && BROKEN_TOOL_JSON.test(String(message.errorMessage || ""));
    cutoff = message.stopReason === "error" && CUT_OFF.test(String(message.errorMessage || "")) && callsOf(message).length === 0;
    if (opts.stripTokens) {
      let count = 0;
      const content = (message.content || []).map(c => c?.type === "text" && typeof c.text === "string"
        ? { ...c, text: c.text.replace(TEMPLATE_TOKENS, () => { count++; return ""; }) } : c);
      if (count) { stats.stripped += count; record("stripped", ctx, { reason: "template tokens", count }); message = cleaned = { ...message, content }; }
    }
    const tools = pi.getActiveTools?.();
    const kind = classifyTurn({ text: textOf(message), thinking: thinkingOf(message), toolCalls: callsOf(message), stopReason: message.stopReason, tools });
    lastKind = kind;
    if (turnRecovery({ kind }).action !== "repair") return cleaned ? { message: cleaned } : undefined;
    const { calls, text } = parseTextToolCalls(textOf(message), { tools });
    const content = [
      ...(message.content || []).filter(c => c?.type === "thinking"),
      ...(text ? [{ type: "text", text }] : []),
      ...calls.map(call => ({ type: "toolCall", id: `guard_${now().toString(36)}_${seq++}`, name: call.name, arguments: call.arguments })),
    ];
    stats.repaired += calls.length;
    lastKind = "tool-call";
    record("repaired", ctx, { tools: calls.map(c => c.name) });
    return { message: { ...message, content, stopReason: "toolUse" } };
  });

  pi.on("agent_before_settle", (event, ctx) => {
    if (event?.continue) return undefined;
    if (event?.outcome === "error" && cutoff && opts.cutoff) {
      cutoff = false;
      // Nothing ran in the cut-off turn (no tool call came through), so asking again is safe.
      if (cutoffRetries >= opts.cutoffRetries) { stats.stopped++; record("out-of-cutoff-retries", ctx, {}); return undefined; }
      cutoffRetries++; stats.cutoffs++;
      record("cutoff", ctx, { retry: cutoffRetries });
      return { entries: [...(event.entries || []), { type: "custom_message", customType: ENTRY, content: CUT_OFF_NUDGE, display: true }], continue: true };
    }
    if (event?.outcome === "error" && brokenJson && opts.toolJsonRetry) {
      brokenJson = false;
      if (mutated) { record("not-retried", ctx, { reason: "broken tool JSON after a change" }); return undefined; }
      if (nudges >= opts.maxNudges) { stats.stopped++; record("out-of-nudges", ctx, { reason: "broken tool JSON" }); return undefined; }
      nudges++; stats.nudged++;
      record("nudged", ctx, { reason: "broken tool JSON", nudge: nudges });
      return { entries: [...(event.entries || []), { type: "custom_message", customType: ENTRY, content: TOOL_JSON_NUDGE, display: true }], continue: true };
    }
    if (event?.outcome !== "completed") return undefined;
    const decision = turnRecovery({ kind: lastKind, nudges, maxNudges: opts.maxNudges, canRepair: false });
    if (decision.action === "stop") { stats.stopped++; record("out-of-nudges", ctx, { reason: decision.reason }); return undefined; }
    if (decision.action !== "nudge") return undefined;
    nudges++; stats.nudged++;
    record("nudged", ctx, { reason: decision.reason, nudge: nudges });
    lastKind = null;
    const entries = [...(event.entries || []), { type: "custom_message", customType: ENTRY, content: decision.message, display: true }];
    return { entries, continue: true };
  });

  pi.on("agent_end", (_event, ctx) => {
    if (!opts.compactAt || compacted || stalled) return;
    let usage;
    try { usage = ctx?.getContextUsage?.(); } catch {}
    if (!usage?.tokens || !usage.contextWindow || usage.tokens <= opts.compactAt * usage.contextWindow) return;
    compacted = true; stats.compacted++;
    record("compacted", ctx, { reason: "context " + Math.round(100 * usage.tokens / usage.contextWindow) + "% full" });
    timers.setTimeout(() => {
      try { ctx.compact?.({ customInstructions: COMPACT_NOTE, onError: error => record("compact-failed", ctx, { reason: error?.message }) }); } catch {}
    }, 0);
  });

  pi.on("agent_settled", (event, ctx) => {
    clearTimer();
    const reason = stalled;
    if (!reason || !event?.aborted) return;
    stalled = null;
    if (nudges >= opts.maxNudges) { stats.stopped++; record("out-of-nudges", ctx, { reason }); return; }
    nudges++; stats.nudged++;
    record("nudged", ctx, { reason, nudge: nudges });
    resending = true;
    pi.sendUserMessage(reason === "reasoning-loop" ? LOOP_NUDGE : STALL_NUDGE);
  });

  pi.on("session_shutdown", () => clearTimer());

  pi.registerCommand?.("guard", {
    description: "Show what the local model guard did in this session",
    handler: async (_args, ctx) => ctx.ui.notify(`Local model guard: ${JSON.stringify(stats)}`, "info"),
  });
  return { stats };
}
