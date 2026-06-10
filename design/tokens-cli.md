# Design: `tokens` — a CLI token counter

**Status:** agreed design, not yet implemented
**Date:** 2026-06-09

## 1. Purpose

`tokens` counts the number of tokens in text files, as tokenized by a chosen
model's tokenizer. Different models use different tokenization schemes, so the
model (or tokenizer) is a first-class parameter.

```console
$ tokens a.txt
1234
```

## 2. Scope

**In scope:** any tokenizer that works without calling a paid API.
Concretely, two families:

1. **tiktoken encodings** (OpenAI): vocabularies are open and embedded in the
   `tiktoken-rs` crate. Fully offline, exact.
2. **Hugging Face tokenizers**: any model that publishes a `tokenizer.json`
   on the HF Hub (Llama, Mistral, Qwen, Gemma, …), loaded with the
   `tokenizers` crate. Exact; requires a one-time free download per model.

**Out of scope:**

- Anthropic/Claude models — the tokenizer is not public and exact counts
  require the paid `count_tokens` API. We do not ship an offline
  approximation either: a wrong-but-plausible number is worse than a clear
  error from a measuring tool.
- Chat-template/message counting (system prompts, role markers). This tool
  counts text, not conversations.

## 3. CLI

```
tokens [OPTIONS] [PATH]...

ARGS:
  [PATH]...            Files to count. `-` means stdin. If no paths are given
                       and stdin is not a TTY, read stdin. If no paths are
                       given and stdin is a TTY, print usage and exit 2.

OPTIONS:
  -m, --model <NAME>       Model or encoding name (default: o200k_base)
      --tokenizer <PATH>   Load a local tokenizer.json directly
                           (mutually exclusive with --model and --revision)
      --revision <REF>     HF Hub revision (branch, tag, or commit SHA) to pin
                           (default: main; only meaningful for HF models)
      --special-tokens     Count with the tokenizer's native special-token
                           behavior instead of raw-text semantics (see §4)
      --json               Emit JSON instead of plain text
  -h, --help
  -V, --version
```

### 3.1 Model name resolution (strict, no aliases)

`--model` is resolved in this order:

1. **Known tiktoken model name** (e.g. `gpt-4o`, `gpt-4`, `o1`) → the
   corresponding tiktoken encoding, via `tiktoken-rs`'s model→encoding table.
2. **Known tiktoken encoding name** (`o200k_base`, `cl100k_base`,
   `p50k_base`, …) → that encoding directly.
3. **Contains `/`** → treated as a Hugging Face Hub repo id
   (e.g. `meta-llama/Llama-3.1-8B`); its `tokenizer.json` is fetched/cached.
4. **Anything else** → error (exit 2) explaining the three accepted forms,
   e.g.:

   ```
   tokens: unknown model 'llama3'
   accepted: a tiktoken model (gpt-4o), a tiktoken encoding (o200k_base),
   or a Hugging Face repo id (meta-llama/Meta-Llama-3-8B)
   ```

There is no built-in alias table and no fuzzy matching: alias tables go
stale and fuzzy resolution makes counts nondeterministic. `llama3` is an
error, not a guess.

### 3.2 Default model

With no `--model`, the default is the **`o200k_base`** encoding (used by
gpt-4o and the o-series). Defaulting to an encoding name rather than a model
name is honest about what is being measured, works fully offline, and makes
the canonical invocation `tokens a.txt` work with zero setup.

### 3.3 Local tokenizer files

`--tokenizer <path>` loads a `tokenizer.json` directly from disk with no
network access — the escape hatch for air-gapped machines or unpublished
tokenizers. It is mutually exclusive with `--model`/`--revision`; `--model`
remains a pure name namespace (paths are never sniffed out of it). In JSON
output the given path is reported as the `model` value.

## 4. Counting semantics

There are two modes. The default answers exactly one question: **how many
tokens does this text encode to**, with no decoration; `--special-tokens`
opts into the tokenizer's native behavior.

### 4.1 Default: raw-text semantics

- **No special tokens are added.** HF tokenizers are invoked with
  `add_special_tokens = false` (so e.g. Llama 3's automatic
  `<|begin_of_text|>` BOS is not counted); tiktoken uses
  `encode_ordinary`. Consequence: an **empty file counts as 0 tokens for
  every model**.
- **Literal special-token text: each crate's obvious behavior, no
  workarounds.** We call each library the straightforward way and accept
  what it does with special-token strings appearing literally in the input:
  - **tiktoken** (`encode_ordinary`): literal `<|endoftext|>` is encoded as
    plain characters (~7 tokens).
  - **HF `tokenizers`** (`encode(text, false)`): the crate extracts
    registered special tokens found in the input regardless of
    `add_special_tokens` (that flag only controls the post-processor's added
    templates, e.g. BOS), so a literal `<|begin_of_text|>` in a file counts
    as 1 token. The crate offers no simple switch to disable this, and we
    deliberately do **not** build one (splitting input around special-token
    matches, etc.) — that complexity isn't worth it for a counting tool.

  This divergence is documented in `--help`. It only matters for files that
  happen to contain a model's own special-token strings; for ordinary text
  the two families behave identically.

The headline guarantee of the default mode — nothing is ever *added*, empty
file = 0 for every model — holds uniformly across families.

### 4.2 `--special-tokens`: tokenizer-native semantics

Counts what the tokenizer would actually produce for this text, per-family:

- **HF:** `add_special_tokens = true` — post-processor templates are counted
  (Llama 3: empty file = 1, the BOS), and literal special-token strings in
  the input are recognized as single tokens.
- **tiktoken:** `encode_with_special_tokens` — nothing is auto-added (empty
  file is still 0), but literal special-token strings such as
  `<|endoftext|>` are recognized as single tokens.

In this mode counts are intentionally **not** comparable across families;
each tokenizer's own conventions win.

Input must be valid UTF-8 (see §6). Files are read fully into memory and
encoded in one call — tokenizers cannot be safely streamed across arbitrary
chunk boundaries, and memory proportional to file size is acceptable for
this tool's use cases.

## 5. Output

### 5.1 Plain text (default)

- **Single input:** the bare count and nothing else, on stdout:

  ```console
  $ tokens a.txt
  1234
  ```

- **Multiple inputs:** wc-style — one `<count> <path>` line per input plus a
  `total` line (counts right-aligned to a common width; stdin is shown as
  `-`):

  ```console
  $ tokens a.txt b.txt
   1234 a.txt
    987 b.txt
   2221 total
  ```

  The `total` line includes only files that were successfully counted.

### 5.2 JSON (`--json`)

A single JSON document on stdout, regardless of input count:

```json
{
  "model": "o200k_base",
  "special_tokens": false,
  "files": [
    { "path": "a.txt", "tokens": 1234 },
    { "path": "b.txt", "error": "not valid UTF-8 text" }
  ],
  "total": 1234
}
```

- `model` is the resolved identifier: the encoding name, the HF repo id
  (plus `"revision"` key when not `main`), or the `--tokenizer` path.
- `special_tokens` records which counting mode (§4) produced the numbers.
- Failed files appear with an `error` string instead of `tokens`.
- `total` sums successfully counted files only.

## 6. Errors and exit codes

**Per-file errors don't stop the run.** Unreadable files, missing files, and
files that are not valid UTF-8 are reported to **stderr** as
`tokens: <path>: <reason>`; remaining files are still counted. Invalid UTF-8
is an error, not a lossy decode — a token count of a PNG is a garbage number
that looks real, so we refuse to produce it.

**Setup errors abort before any counting:** unknown model name, tokenizer
download/auth failure, unreadable `--tokenizer` file, conflicting flags.

Exit codes (grep/wc conventions):

| Code | Meaning                                                      |
| ---- | ------------------------------------------------------------ |
| 0    | All inputs counted successfully                              |
| 1    | At least one input failed; others (if any) were counted      |
| 2    | Usage or setup error — nothing was counted                   |

## 7. Tokenizer acquisition and caching (HF models)

Downloads go through the official **`hf-hub`** crate (ureq backend), which
provides auth, redirect handling, revision resolution, and atomic cache
writes. Properties we rely on:

- **Cache location:** the standard `~/.cache/huggingface/hub` layout
  (respecting `HF_HOME`), with snapshots stored **by commit SHA**. The cache
  is shared with Python `huggingface_hub` tooling — a tokenizer downloaded by
  either is visible to both.
- **Cache-first lookup:** if the requested repo+revision is already in the
  cache, it is used **without any network access**. A revision like `main`
  is resolved to a commit once, on first download, and never re-checked —
  counts on a given machine never silently shift because upstream moved.
  Once a model has been used once, the tool works fully offline for it.
- **Reproducibility across machines:** `--revision <commit-sha>` pins the
  exact tokenizer, so two machines are guaranteed identical counts. (Two
  machines that first fetched `main` at different times may differ if the
  repo changed in between; pinning is the remedy.)
- **Gated repos** (e.g. `meta-llama/*`): `hf-hub` sends the user's HF token
  from the `HF_TOKEN` env var or `~/.cache/huggingface/token`. A 401/403 is
  reported as a setup error (exit 2) with a hint to set `HF_TOKEN` and accept
  the model license on the Hub.

Only `tokenizer.json` is fetched — never weights or other repo files.

## 8. Implementation

### 8.1 Crates

| Crate         | Role                                              |
| ------------- | ------------------------------------------------- |
| `clap` (derive) | Argument parsing                                |
| `tiktoken-rs` | OpenAI encodings (vocabularies embedded)          |
| `tokenizers`  | Loading and running HF `tokenizer.json`           |
| `hf-hub` (ureq feature) | Hub download + cache                    |
| `anyhow`      | Error propagation in `main`                       |
| `serde_json`  | `--json` output                                   |

Cargo package: `token-counter` (matching the repo), with
`[[bin]] name = "tokens"`. Rust 2024 edition.

### 8.2 Structure

Single small binary crate:

```
src/
  main.rs        # clap definitions, orchestration, output, exit codes
  resolve.rs     # model-name → TokenizerSource (tiktoken | hub | local path)
  counter.rs     # Counter trait + tiktoken/HF implementations
```

Core abstraction — both families behind one trait so the counting loop is
family-agnostic:

```rust
trait Counter {
    /// Token count of `text`. The counting mode (§4: raw-text vs
    /// --special-tokens) is fixed when the Counter is constructed.
    fn count(&self, text: &str) -> usize;
}
```

Flow: parse args → resolve tokenizer (one `Counter`, built once) → for each
input: read file → validate UTF-8 → `count()` → record result → print
plain/JSON → exit per §6.

### 8.3 Testing

**Requirement: the entire test suite must run without network access.**
`cargo test` must pass on an offline machine. No test may download from the
HF Hub or any other remote; HF-tokenizer behavior is tested exclusively
against `tokenizer.json` fixtures checked into `tests/fixtures/` (small,
hand-trimmed vocabularies are fine — only behavior matters, not realism).
tiktoken tests need nothing external since vocabularies are embedded in the
crate. Code paths that would hit the network (Hub resolution, auth errors)
are tested at the seam: resolution logic is unit-tested up to the point of
download, and download/auth failures are exercised by injecting errors, not
by real requests.

- **Unit tests, resolution:** each branch of §3.1, including the
  unknown-name error and `--model`/`--tokenizer` exclusivity.
- **Unit tests, default semantics (§4.1):** empty input → 0 for both
  families; tiktoken counts literal `<|endoftext|>` as ordinary text; an HF
  fixture with a registered special token counts a literal occurrence as 1
  token (pinning the documented crate behavior); a few known strings with
  hardcoded expected counts for `o200k_base` and `cl100k_base`.
- **Unit tests, `--special-tokens` (§4.2):** a fixture tokenizer with a
  BOS-adding post-processor → empty input counts 1 with the flag, 0 without;
  tiktoken `<|endoftext|>` → 1 token with the flag, ~7 without.
- **Integration tests** (e.g. `assert_cmd`): single-file bare-number output,
  multi-file wc-style output and `total`, stdin via `-` and via pipe,
  `--json` shape (including the `special_tokens` field), missing/non-UTF-8
  file → stderr message + exit 1, unknown model → exit 2.

## 9. Decision log

| Decision | Choice | Rejected alternatives |
| --- | --- | --- |
| Model scope | tiktoken + HF tokenizers; no paid APIs | Anthropic via API (paid, network); Anthropic heuristic (misleading) |
| HF tokenizer source | Hub download + cache, plus local path | local-path only (friction); bundling (fat binary, stale list) |
| Name resolution | strict; `/` ⇒ HF repo id; no aliases | curated aliases (stale); Hub search (nondeterministic) |
| Default model | `o200k_base` encoding | gpt-4o framing; no default (breaks `tokens a.txt`); config file |
| Count semantics | raw text by default; `--special-tokens` flag opts into tokenizer-native behavior (revised: flag was originally deferred past v0) | tokenizer defaults as the only behavior (inconsistent across models) |
| Literal special-token text | use each crate's obvious behavior and document the divergence (tiktoken: plain text; HF: extracted as 1 token) | custom input-splitting in the CLI to force uniformity (complexity not worth it) |
| Inputs | many paths + stdin, wc-style | single file only; no stdin |
| Output | bare number / wc-style multi, plus `--json` | always `count path` (breaks scripting) |
| Invalid UTF-8 | per-file error | lossy decode (garbage counts) |
| Failure policy | continue, exit 1; setup errors exit 2 | fail fast; always exit 0 |
| Cache policy | cache forever; `--revision` to pin | check upstream every run; TTL refresh (time-dependent counts) |
| Local tokenizer | dedicated `--tokenizer` flag | overloading `--model` with paths (ambiguous with repo ids) |
| Hub client | `hf-hub` crate, standard HF cache | hand-rolled ureq + own `~/.cache/tokens` (reimplements auth/atomicity) |
| Testing | hard requirement: full suite passes with no network; HF behavior via checked-in fixtures | tests that download from the Hub (flaky, breaks offline `cargo test`) |
