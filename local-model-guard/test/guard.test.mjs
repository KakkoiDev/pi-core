import { test } from "node:test";
import assert from "node:assert/strict";
import { installLocalModelGuard, options, listFolder, profileFor, cutOffNudge, flattenEditTool, FLAT_EDIT } from "../guard.mjs";
import { readFileSync } from "node:fs";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

function fakePi({ tools = ["bash", "read", "edit", "write"], thinking = "off" } = {}) {
  const handlers = {}, sent = [], entries = [];
  const pi = {
    on: (name, fn) => { (handlers[name] ||= []).push(fn); },
    getActiveTools: () => tools,
    getThinkingLevel: () => thinking,
    sendUserMessage: text => sent.push(text),
    appendEntry: (type, data) => entries.push({ type, data }),
    registerCommand: () => {},
  };
  const emit = async (name, event = {}, ctx = {}) => {
    let out;
    for (const fn of handlers[name] || []) out = (await fn({ type: name, ...event }, ctx)) ?? out;
    return out;
  };
  return { pi, emit, sent, entries };
}
const ctx = { model: { provider: "ollama", id: "gemma4:26b-mlx" }, thinkingLevel: "off", abort() { this.aborted = true; } };
const assistant = (content, stopReason = "stop") => ({ role: "assistant", content, stopReason });
const settle = (emit, extra = {}) => emit("agent_before_settle", { outcome: "completed", continue: false, entries: [], ...extra }, ctx);
const opts = { ...options({}), idleMs: 0 };

test("gemma4 on Ollama with thinking off gets reasoning_effort none; other models are untouched", async () => {
  const { pi, emit } = fakePi();
  installLocalModelGuard(pi, opts);
  assert.deepEqual(await emit("before_provider_request", { payload: { model: "gemma4:26b-mlx", stream: true } }, ctx),
    { model: "gemma4:26b-mlx", stream: true, reasoning_effort: "none" });
  assert.equal(await emit("before_provider_request", { payload: { model: "q" } }, { ...ctx, model: { provider: "ollama", id: "qwen3.8:27b-mlx" } }), undefined);
});

test("a tool call written as text (gemma's bank-transfer run) becomes a real tool call", async () => {
  const { pi, emit } = fakePi();
  installLocalModelGuard(pi, opts);
  const out = await emit("message_end", { message: assistant([
    { type: "thinking", thinking: "plan" },
    { type: "text", text: 'I will begin by listing the files.\n<tool> bash {   command: "ls -R" } </tool>' },
  ]) }, ctx);
  assert.equal(out.message.stopReason, "toolUse");
  assert.deepEqual(out.message.content.map(c => c.type), ["thinking", "text", "toolCall"]);
  assert.deepEqual(out.message.content[2].arguments, { command: "ls -R" });
  assert.equal(await settle(emit), undefined, "a repaired turn needs no nudge");
});

test("thinking-only turns (gemma's feature-todo run) are nudged, at most maxNudges times per request", async () => {
  const { pi, emit } = fakePi();
  installLocalModelGuard(pi, { ...opts, maxNudges: 2 });
  await emit("before_agent_start");
  for (const expected of [true, true, false]) {
    await emit("message_end", { message: assistant([{ type: "thinking", thinking: "I should edit todo.py" }]) }, ctx);
    const out = await settle(emit);
    assert.equal(Boolean(out?.continue), expected);
    if (expected) assert.match(out.entries.at(-1).content, /no tool call and no answer/);
  }
  await emit("before_agent_start"); // the person's next request gets a fresh budget
  await emit("message_end", { message: assistant([]) }, ctx);
  assert.equal((await settle(emit)).continue, true);
});

test("answers, real tool calls, aborted runs and runs already continuing are left alone", async () => {
  const { pi, emit } = fakePi();
  installLocalModelGuard(pi, opts);
  await emit("message_end", { message: assistant([{ type: "text", text: "All tests pass." }]) }, ctx);
  assert.equal(await settle(emit), undefined);
  await emit("message_end", { message: assistant([{ type: "toolCall", id: "1", name: "bash", arguments: {} }], "toolUse") }, ctx);
  assert.equal(await settle(emit), undefined);
  await emit("message_end", { message: assistant([{ type: "thinking", thinking: "x" }]) }, ctx);
  assert.equal(await settle(emit, { outcome: "aborted" }), undefined);
  assert.equal(await settle(emit, { continue: true }), undefined);
});

test("a stalled request is aborted and asked to continue once", async () => {
  let t = 0; const pending = [];
  const timers = { setTimeout: (fn, ms) => { pending.push({ fn, at: t + ms }); return pending.length; }, clearTimeout: id => { pending[id - 1] = null; } };
  const { pi, emit, sent } = fakePi();
  installLocalModelGuard(pi, { ...opts, idleMs: 60_000, totalMs: 600_000 }, { now: () => t, timers });
  const c = { ...ctx };
  await emit("message_start", { message: { role: "assistant" } }, c);
  t = 30_000; await emit("message_update", {}, c);
  const run = () => { const due = pending.filter(Boolean).filter(p => p.at <= t); due.forEach(p => { pending[pending.indexOf(p)] = null; p.fn(); }); };
  t = 60_000; run(); assert.equal(c.aborted, undefined, "activity at 30s kept it alive");
  t = 95_000; run(); run();
  assert.equal(c.aborted, true);
  await emit("agent_settled", { aborted: true }, c);
  assert.equal(sent.length, 1);
  await emit("before_agent_start", {}, c); // the guard's own follow-up keeps the nudge count
  await emit("agent_settled", { aborted: true }, c);
  assert.equal(sent.length, 1, "no resend without a new stall");
});

test("PI_GUARD=off disables everything", async () => {
  const { pi, emit } = fakePi();
  installLocalModelGuard(pi, options({ PI_GUARD: "off" }));
  assert.equal(await emit("before_provider_request", { payload: {} }, ctx), undefined);
});

test("leaked template tokens are removed from the answer", async () => {
  const { pi, emit } = fakePi();
  installLocalModelGuard(pi, opts);
  const out = await emit("message_end", { message: assistant([{ type: "text", text: "<|channel|>Done.<channel|>" }]) }, ctx);
  assert.equal(out.message.content[0].text, "Done.");
  assert.equal(await emit("message_end", { message: assistant([{ type: "text", text: "a <b> c" }]) }, ctx), undefined);
});

test("a reasoning loop is stopped and the model is asked to act, within the nudge budget", async () => {
  const { pi, emit, sent } = fakePi();
  installLocalModelGuard(pi, opts);
  const c = { ...ctx };
  await emit("before_agent_start", {}, c);
  await emit("agent_start", {}, c);
  const think = delta => emit("message_update", { assistantMessageEvent: { type: "thinking_delta", delta } }, c);
  await think("The test fails.\nI should read app.py.\nThen fix the import.\n");
  assert.equal(c.aborted, undefined);
  await think("Wait.\nThe test fails.\nI should read app.py.\nThen fix the import.\n");
  assert.equal(c.aborted, true);
  await emit("agent_settled", { aborted: true }, c);
  assert.equal(sent.length, 1);
  assert.match(sent[0], /repeated the same lines/);
});

test("PI_GUARD_LOOP_WATCH=off leaves repeated reasoning alone", async () => {
  const { pi, emit } = fakePi();
  installLocalModelGuard(pi, { ...opts, loopWatch: false });
  const c = { ...ctx };
  await emit("agent_start", {}, c);
  const lines = "a\nb\nc\n";
  await emit("message_update", { assistantMessageEvent: { type: "thinking_delta", delta: lines + "x\n" + lines } }, c);
  assert.equal(c.aborted, undefined);
});

test("broken tool JSON is retried once, but not after a tool changed something", async () => {
  const broken = { role: "assistant", content: [], stopReason: "error", errorMessage: "error parsing tool call: raw='{\"path\":' err=unexpected end of JSON input" };
  const { pi, emit } = fakePi();
  installLocalModelGuard(pi, opts);
  await emit("before_agent_start");
  await emit("agent_start");
  await emit("message_end", { message: broken }, ctx);
  const out = await settle(emit, { outcome: "error" });
  assert.equal(out.continue, true);
  assert.match(out.entries.at(-1).content, /incomplete tool JSON/);

  await emit("before_agent_start");
  await emit("agent_start");
  await emit("tool_execution_start", { toolName: "write" }, ctx);
  await emit("message_end", { message: broken }, ctx);
  assert.equal(await settle(emit, { outcome: "error" }), undefined, "a write may already have run: do not replay");
});

test("a turn past the compaction share compacts once with the open work kept", async () => {
  const later = [];
  const timers = { setTimeout: fn => later.push(fn), clearTimeout() {} };
  const { pi, emit } = fakePi();
  installLocalModelGuard(pi, { ...opts, compactAt: 0.6 }, { timers });
  const compacts = [];
  const c = { ...ctx, getContextUsage: () => ({ tokens: 20_000, contextWindow: 32_768 }), compact: o => compacts.push(o) };
  await emit("agent_start", {}, c);
  await emit("agent_end", {}, c);
  await emit("agent_end", {}, c);
  later.splice(0).forEach(fn => fn());
  assert.equal(compacts.length, 1);
  assert.match(compacts[0].customInstructions, /Keep the open work/);
  const small = { ...c, getContextUsage: () => ({ tokens: 10_000, contextWindow: 32_768 }) };
  await emit("agent_start", {}, small);
  await emit("agent_end", {}, small);
  later.splice(0).forEach(fn => fn());
  assert.equal(compacts.length, 1);
});

// --- gpt-oss profile -----------------------------------------------------------------------------------------------

const gpt = { ...ctx, model: { provider: "ollama", id: "gpt-oss:20b" } };

test("gpt-oss: no reasoning_effort by default; PI_GUARD_GPT_OSS_EFFORT opts in when thinking is off; a set effort is kept", async () => {
  const run = async (o, payload, c = gpt) => { const { pi, emit } = fakePi(); installLocalModelGuard(pi, o); return emit("before_provider_request", { payload }, c); };
  assert.equal(await run(opts, { model: "gpt-oss:20b" }), undefined);
  assert.deepEqual(await run({ ...opts, gptOssEffort: "low" }, { model: "gpt-oss:20b" }), { model: "gpt-oss:20b", reasoning_effort: "low" });
  assert.equal(await run({ ...opts, gptOssEffort: "default" }, { model: "gpt-oss:20b" }), undefined);
  assert.equal(await run({ ...opts, gptOssEffort: "low" }, { model: "gpt-oss:20b", reasoning_effort: "high" }), undefined, "pi already chose an effort");
  assert.equal(await run({ ...opts, gptOssEffort: "low" }, { model: "gpt-oss:20b" }, { ...gpt, thinkingLevel: "medium" }), undefined, "thinking on: pi decides");
  assert.equal(await run({ ...options({ PI_GUARD_PROFILE: "off" }), gptOssEffort: "low" }, { model: "gpt-oss:20b" }), undefined);
  assert.equal(await run({ ...opts, gptOssEffort: "low" }, { model: "q" }, { ...ctx, model: { provider: "ollama", id: "qwen3.8:27b-mlx" } }), undefined, "other models untouched");
  assert.equal(options({ PI_GUARD_GPT_OSS_EFFORT: "high" }).gptOssEffort, "high");
});

test("listFolder lists files with folders indented, skipping hidden and build folders, within a limit", () => {
  const dir = mkdtempSync(join(tmpdir(), "guard-ls-"));
  mkdirSync(join(dir, "bank")); mkdirSync(join(dir, ".git")); mkdirSync(join(dir, "__pycache__"));
  writeFileSync(join(dir, "bank", "ledger.py"), ""); writeFileSync(join(dir, "test_bank.py"), ""); writeFileSync(join(dir, ".env"), "");
  assert.equal(listFolder(dir), "bank/\n  ledger.py\ntest_bank.py");
  assert.equal(listFolder(dir, { max: 1 }), "bank/\n… and 2 more");
});

test("gpt-oss gets the working folder in the system prompt; other models only with PI_GUARD_LIST_FILES=on", async () => {
  const dir = mkdtempSync(join(tmpdir(), "guard-sp-"));
  writeFileSync(join(dir, "words.py"), "");
  const run = async (o, model) => { const { pi, emit } = fakePi(); installLocalModelGuard(pi, o);
    return emit("before_agent_start", { prompt: "x", systemPrompt: "BASE" }, { ...ctx, cwd: dir, model: { provider: "ollama", id: model } }); };
  const out = await run(opts, "gpt-oss:20b");
  assert.match(out.systemPrompt, /^BASE\n\n## Working folder\n[\s\S]*words\.py[\s\S]*read takes a file path, not a folder/);
  assert.equal(await run(opts, "qwen3.8:27b-mlx"), undefined);
  assert.match((await run({ ...opts, listFiles: "on" }, "qwen3.8:27b-mlx")).systemPrompt, /words\.py/);
  assert.equal(await run({ ...opts, listFiles: "off" }, "gpt-oss:20b"), undefined);
});

test("gpt-oss: read on a folder gets a hint to list it with bash", async () => {
  const { pi, emit } = fakePi();
  installLocalModelGuard(pi, opts);
  const ev = { toolName: "read", input: { path: "" }, content: [{ type: "text", text: "EISDIR: illegal operation on a directory, read" }], isError: true };
  const out = await emit("tool_result", ev, gpt);
  assert.match(out.content.at(-1).text, /is a folder\. List it with bash \(ls\)/);
  assert.equal(await emit("tool_result", ev, ctx), undefined, "not for other models");
  assert.equal(await emit("tool_result", { ...ev, isError: false }, gpt), undefined);
  assert.equal(profileFor("gpt-oss:20b").id, "gpt-oss");
});

test("a cut-off reply (gpt-oss: thinking, then no finish reason) is continued at once, up to the limit", async () => {
  const cut = { role: "assistant", content: [{ type: "thinking", thinking: "I should read the config file first." }], stopReason: "error", errorMessage: "Stream ended without finish_reason" };
  const { pi, emit } = fakePi();
  installLocalModelGuard(pi, { ...opts, cutoffRetries: 2 });
  await emit("before_agent_start");
  await emit("agent_start");
  // An earlier write in the same request does not block it: nothing ran in the cut-off turn.
  await emit("tool_execution_start", { toolName: "write" }, ctx);
  for (const _ of [1, 2]) {
    await emit("message_end", { message: cut }, ctx);
    const out = await settle(emit, { outcome: "error" });
    assert.equal(out.continue, true);
    assert.match(out.entries.at(-1).content, /not valid JSON/, "on Ollama the note names the cause: tool JSON the server dropped");
  }
  await emit("message_end", { message: cut }, ctx);
  assert.equal(await settle(emit, { outcome: "error" }), undefined, "out of retries: pi's own handling takes over");

  // The budget resets for the next request.
  await emit("before_agent_start");
  await emit("message_end", { message: cut }, ctx);
  assert.equal((await settle(emit, { outcome: "error" })).continue, true);

  // Other errors are left to pi, and so is a cut-off that already carried a tool call.
  await emit("message_end", { message: { ...cut, errorMessage: "connection refused" } }, ctx);
  assert.equal(await settle(emit, { outcome: "error" }), undefined);
  await emit("message_end", { message: { ...cut, content: [{ type: "toolCall", id: "t", name: "read", arguments: { path: "a" } }] } }, ctx);
  assert.equal(await settle(emit, { outcome: "error" }), undefined);

  // Off switch.
  const off = fakePi();
  installLocalModelGuard(off.pi, { ...opts, cutoff: false });
  await off.emit("before_agent_start");
  await off.emit("message_end", { message: cut }, ctx);
  assert.equal(await settle(off.emit, { outcome: "error" }), undefined);
});

test("the cut-off note names broken tool JSON on Ollama, and stays generic elsewhere", async () => {
  assert.match(cutOffNudge({ model: { provider: "ollama", id: "gpt-oss:20b" } }), /arguments were not valid JSON[\s\S]*one closing brace too many/);
  for (const other of [{ model: { provider: "openrouter", id: "openai/gpt-oss-20b" } }, {}, undefined])
    assert.match(cutOffNudge(other), /cut off before any answer or tool call/);
});

test("gpt-oss auto effort: low for a quick reply, default once a tool has run in the request", async () => {
  const oss = { ...ctx, model: { provider: "ollama", id: "gpt-oss:20b" } };
  const { pi, emit } = fakePi();
  installLocalModelGuard(pi, { ...opts, gptOssEffort: "auto" });
  await emit("before_agent_start");
  assert.equal((await emit("before_provider_request", { payload: { model: "gpt-oss:20b" } }, oss)).reasoning_effort, "low");
  await emit("tool_execution_start", { toolName: "read" }, oss);
  const after = await emit("before_provider_request", { payload: { model: "gpt-oss:20b" } }, oss);
  assert.equal(after?.reasoning_effort, undefined, "default effort: none sent");
  await emit("before_agent_start"); // next request starts low again
  assert.equal((await emit("before_provider_request", { payload: { model: "gpt-oss:20b" } }, oss)).reasoning_effort, "low");
});

// A request pi 1.1.0 sent to Ollama for gpt-oss:20b (system prompt cut to its tools and rules).
const piRequest = () => JSON.parse(readFileSync(new URL("./fixtures/pi-gpt-oss-request.json", import.meta.url), "utf8"));

test("flattenEditTool offers edit as one flat replacement and rewrites the prompt lines about edits[]", () => {
  const req = piRequest();
  const flat = flattenEditTool(req);
  const edit = flat.tools.find(t => t.function.name === "edit").function;
  assert.deepEqual(edit.parameters.required, ["path", "oldText", "newText"]);
  assert.equal(edit.parameters.properties.edits, undefined);
  assert.deepEqual(flat.tools.map(t => t.function.name), ["read", "bash", "edit", "write"], "other tools and their order are kept");
  const prompt = flat.messages[0].content;
  assert.doesNotMatch(prompt, /edits\[\]/);
  assert.match(prompt, /- edit: Make one precise file edit/);
  assert.match(prompt, /- Use edit for precise changes \(oldText must match exactly\)/);
  assert.match(prompt, /- Use write only for new files/, "unrelated rules are kept");
  assert.equal(flat.messages[1].content, "Fix the bug.");
  assert.ok(req.tools.find(t => t.function.name === "edit").function.parameters.properties.edits, "the input is not changed");
  assert.equal(flattenEditTool(flat), null, "already flat");
  assert.equal(flattenEditTool({ messages: [] }), null);
  assert.equal(FLAT_EDIT.name, "edit");
});

test("gpt-oss requests get the flat edit tool; other models only with PI_GUARD_FLAT_EDIT=on, never with off", async () => {
  const oss = { ...ctx, model: { provider: "ollama", id: "gpt-oss:20b" } };
  const qwen = { ...ctx, model: { provider: "ollama", id: "qwen3.8:27b-mlx" } };
  const editOf = payload => payload.tools.find(t => t.function.name === "edit").function.parameters;
  const run = async (o, c) => { const { pi, emit } = fakePi(); installLocalModelGuard(pi, { ...opts, ...o }); return emit("before_provider_request", { payload: piRequest() }, c); };
  assert.deepEqual(editOf(await run({}, oss)).required, ["path", "oldText", "newText"]);
  assert.equal((await run({}, oss)).reasoning_effort, "low", "the rest of the payload is kept");
  assert.equal(await run({}, qwen), undefined);
  assert.deepEqual(editOf(await run({ flatEdit: "on" }, qwen)).required, ["path", "oldText", "newText"]);
  assert.equal(await run({ flatEdit: "off" }, oss), undefined, "nothing else to change in this request");
  assert.equal(options({ PI_GUARD_FLAT_EDIT: "off" }).flatEdit, "off");
  assert.equal(options({}).flatEdit, "auto");
});
