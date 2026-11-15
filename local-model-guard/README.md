# local-model-guard

A pi extension that keeps local models on track as coding agents. It was built after a
benchmark where Gemma 4 (26B, on Ollama) failed agentic tasks in pi without writing bad
code. Instead, it:

- **ended turns with only thinking**, so pi stopped with nothing done, even with thinking set to off;
- **wrote a tool call as text** (`<tool> bash { command: "ls -R" } </tool>`), so it never ran;
- **hung** until the bench's timeout.

## What it does

| Problem | What the guard does |
|---|---|
| A model ignores "thinking off" | Adds the request fields that model needs. Today: `reasoning_effort: "none"` for `gemma4` on Ollama. |
| A tool call written as text | Turns it into a real tool call, which pi runs through its normal tool path (and its tool checks). Recognises `<tool> name {...}`, Gemma's `call:name{...}`, Qwen's `<tool_call>{...}`, `<function=name>...`, and a JSON object naming a tool. Only tools that exist are accepted. |
| A turn with only thinking, nothing at all, or "I will now edit the file." with no call | Sends one short follow-up asking the model to act or answer. At most 2 per request by default. |
| An answer cut off at the length limit | Asks the model to continue, more briefly. |
| A request with no output at all for 2 minutes | Cancels it and asks the model to continue. Thinking counts as output, so long reasoning is not cut off. |
| Reasoning that repeats itself (the same three thinking lines seen twice) | Stops the turn and asks the model to take the next step instead of working it out again. From pi-local-harness (lesson L-007). |
| Chat-template tokens leaked into the answer (`<\|channel\|>`, `<channel\|>`) | Removes them, so later turns do not imitate them. From pi-local-harness. |
| The server could not parse the model's tool call ("error parsing tool call: unexpected end of JSON input") | Asks once for a short valid tool call, unless a tool already changed something in this request, so no write is replayed. From pi-local-harness (task 370). |
| A long session nearing the context limit | After a turn that passes 60% of the context window, compacts once, keeping the open work (request, decisions, files touched, what is left). From pi-local-harness (task 116). |

### Model profiles

Some settings are switched on by model id (`PROFILES` in `guard.mjs`). Today one profile, **gpt-oss**
(`gpt-oss:20b`, `gpt-oss-120b`, …), from its agentic-v1 transcripts:

| What gpt-oss did | What the profile does |
|---|---|
| With pi's thinking off, pi sends no reasoning effort and Ollama thinks at medium | Nothing by default. `PI_GUARD_GPT_OSS_EFFORT=low` sends `reasoning_effort: low`: 2.4x faster on agentic-v1 with the same solved count, but worse on held-out benches (agentic-v2 9/10 → 5/10, coding-v2 12/12 → 8/12), so it is opt-in. A level you set in pi is kept |
| Opened every task by calling read on `""` (a folder) | Adds the working folder's files (up to 60 lines, hidden and build folders skipped) to the system prompt, with "read takes a file path, not a folder" |
| Retried read on folders | A read on a folder gets "List it with bash (ls)" |
| Wrote broken JSON in edit's nested `edits[]` array (a brace too many or too few, an object closed after `oldText`); Ollama's harmony parser rejects the call and its `/v1` endpoint drops the error, so pi sees "Stream ended without finish_reason" | Offers `edit` as one flat replacement, `{path, oldText, newText}` (pi's edit tool accepts this form), and rewrites the system prompt lines about `edits[]`. `PI_GUARD_FLAT_EDIT=off` keeps pi's edit tool |

Every action is recorded as a `local-model-guard` session entry. Type `/guard` to see the counts for the session.

## Measured effect

Agentic bench (5 small Python repositories, graded by hidden tests), pi with thinking off,
Ollama on an Apple M6 with 32 GB, 2026-11-14. Single runs per task, so one task either way is noise.

| Model | Without the guard | With the guard |
|---|---|---|
| gemma4:26b-mlx | 1/5 and 0/5 solved, 25–35 min, one 15–25 min hang per run | 3/5 solved, 12/14 hidden tests, 4.6 min, no hang |
| qwen3.8:27b-mlx | 4/5 and 4/5 solved, 6–7 min | 4/5 solved, 4.6 min (the guard took no action) |

For Gemma, nearly all the gain came from the request fix (`reasoning_effort: "none"`); the
guard also sent one nudge and needed no tool-call repair in that run.

## Settings

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `PI_GUARD` | on | `off` disables the guard |
| `PI_GUARD_MAX_NUDGES` | `2` | Follow-ups per request |
| `PI_GUARD_IDLE_SECONDS` | `120` | Cancel a request after this long with no output (`0` turns the stall watch off) |
| `PI_GUARD_TOTAL_SECONDS` | `1200` | Cancel a single model request after this long |
| `PI_GUARD_LOG` | | Also append each action as a JSON line to this file |
| `PI_GUARD_LOOP_WATCH` | on | `off` turns the reasoning-loop watch off |
| `PI_GUARD_STRIP_TOKENS` | on | `off` keeps leaked template tokens |
| `PI_GUARD_TOOL_JSON_RETRY` | on | `off` does not retry a tool call the server could not parse |
| `PI_GUARD_CUTOFF` | on | `off` leaves cut-off replies (no finish reason, no answer or tool call) to pi's retry with backoff. On Ollama the follow-up says the tool call's JSON was dropped, since that is what a cut-off is there |
| `PI_GUARD_CUTOFF_RETRIES` | 3 | how many times one request continues at once after a cut-off |
| `PI_GUARD_COMPACT_AT` | `0.6` | Compact after a turn that passes this share of the context window (`0` turns it off) |
| `PI_GUARD_PROFILE` | on | `off` turns model profiles off |
| `PI_GUARD_GPT_OSS_EFFORT` | (none) | reasoning effort for gpt-oss when pi's thinking is off: `low`, `medium`, `high`, `default`, or `auto` (low until a tool runs in the request, then default) |
| `PI_GUARD_FLAT_EDIT` | auto | `on` offers edit as one flat replacement to every model, `off` to none, `auto` lets the profile decide (gpt-oss) |
| `PI_GUARD_LIST_FILES` | auto | `on` lists the working folder for every model, `off` for none, `auto` lets the profile decide |

## How it works

The rules are [AI Core](https://github.com/KakkoiDev/ai-core)'s turn policies (`classifyTurn`,
`parseTextToolCalls`, `turnRecovery`, `createStallWatch`, `createReasoningWatch`, `requestAdjustments`); this extension
connects them to pi's hooks:

| pi hook | Use |
|---|---|
| `before_provider_request` | Merge request fixes into the payload |
| `message_end` | Remove template tokens; classify the finished turn; replace a text tool call with a real one |
| `agent_before_settle` | Add a follow-up message and continue once (also after a broken tool call) |
| `message_start` / `message_update` / `provider_stream_event` | Stall watch; reasoning-loop watch on thinking |
| `tool_execution_start` | Note that a tool may have changed something, so a broken tool call is not retried |
| `agent_end` | Compact once when the context passes the set share |
| `agent_settled` | After a stall or a reasoning loop was cancelled, ask the model to continue |

A new message from you resets the follow-up budget; the guard's own messages do not, so it
cannot loop.
