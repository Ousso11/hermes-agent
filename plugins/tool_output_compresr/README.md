# tool_output_compresr

Per-turn **tool-output compression** for Hermes, powered by [Compresr](https://compresr.ai) (YC W26).

Large tool outputs — a 5k-line `read_file`, a verbose `grep`, a noisy
`execute_code`/test run — are the real driver of context-window bloat. This
plugin compresses them **as they arrive** (on the `transform_tool_result` hook),
*before* they ever enter the context, using Compresr's query-specific
tool-output API with the **tool's own intent** as the query.

Crucially, it is **lossless by recovery**: every dropped span is rewritten into
an addressable reference the agent can recover with its native tools:

```
[compresr: 12 lines omitted · .compresr/cache/a3f9c1 L40-51 · Read(offset=41,limit=12) or Grep to recover]
```

The original output is cached verbatim under `<workspace>/.compresr/cache/<id>`,
so the reference resolves to a real file. The cache is written through Hermes's
own file-operations backend (`_get_file_ops(task_id).write_file`) — the same
layer the agent's `write_file`/`read_file` tools use — so the path is resolved
identically to how the agent later reads it, and the write lands in the right
place even when the agent runs in a docker/modal sandbox. The model recovers the
whole thing — or just a slice — on demand. Nothing the agent might need is ever
more than one `Read` away.

## How it works — one path

```
1. Below the threshold (min_tokens)? → return None, original output unchanged.
2. Above it → ALWAYS call the Compresr tool-output API (toc_latte_v2, coarse)
   with the tool's intent as the query.
3. Persist the original to .compresr/cache/<id>, then rewrite the API's drop
   markers into addressable references by anchoring verbatim kept lines.
```

It **fails open**: an API error returns `None` (original output unchanged), with
a short cooldown so a failing endpoint isn't hammered. The size threshold is the
only gate — kept deliberately simple; tiered cleanup can be reintroduced later.

The anchoring is robust: it does **not** trust marker counts or units. It maps
dropped spans back to original line ranges by matching the compressed output's
verbatim kept lines against the cached original (normalized, tolerant of ~6%
whitespace drift), so it works for `[5 lines removed]`, `[54 tokens dropped]`,
and even silent drops with no marker at all.

## Data sent to Compresr

When enabled, this plugin sends large tool outputs to the configured Compresr
API endpoint before those outputs enter the conversation context. Those outputs
can include file contents, command output, logs, and other data returned by
Hermes tools. Do not enable per-turn tool-output compression unless that
third-party processing is acceptable for your deployment.

## Relationship to `context_engine/compresr`

This is the **per-turn** complement to the **compaction-time**
[`context_engine/compresr`](../context_engine/compresr) engine. They compose:
this plugin is a pre-filter that shrinks each tool output once, so compaction has
far less residual to summarize. Enable either or both.

## Activation (opt-in, off by default)

```yaml
# ~/.hermes/config.yaml
compresr:
  tool_output_enabled: true
```
```bash
# ~/.hermes/.env
COMPRESR_API_KEY=cmp_...
```
Or run `hermes setup` and choose **Compresr** for the compression engine.

## Configuration

| env / `compresr:` key | default | meaning |
|---|---|---|
| `COMPRESR_API_KEY` | — | **required** `cmp_…` key, read from `.env` only |
| `COMPRESR_BASE_URL` / `base_url` | `https://api.compresr.ai/api` | API base |
| `COMPRESR_TOOL_OUTPUT_ENABLED` / `tool_output_enabled` | `true` | master switch |
| `COMPRESR_TOOL_OUTPUT_MODEL` / `tool_output_model` | `toc_latte_v2` | model (tool-output endpoint only accepts `toc_*` models) |
| `COMPRESR_TOOL_OUTPUT_MIN_TOKENS` / `tool_output_min_tokens` | `1500` | skip smaller outputs |
| `COMPRESR_TOOL_OUTPUT_TIMEOUT` / `tool_output_timeout` | `30` | request timeout (s) |
| `COMPRESR_TOOL_OUTPUT_MAX_CACHE_MB` / `tool_output_max_cache_mb` | `256` | best-effort `.compresr/cache` cap; `0` disables pruning |

The cache cap is a disk-usage guard, not a retention guarantee. Old cached
originals can be pruned after enough later compressed outputs are written in the
same workspace, so a very old resumed session may contain a recovery reference
whose backing cache file has since been evicted. Set `tool_output_max_cache_mb:
0` if preserving historical recovery references across long-lived resumed
sessions matters more than bounding `.compresr/cache` growth.

## Tests

`tests/test_tool_output_compresr.py` — recovery round-trip (marker, silent drop,
whitespace drift, trailing gap, whole-file fallback) and hook gating / fail-open.
