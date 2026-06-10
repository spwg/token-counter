# tokens

A CLI that counts tokens in text files, as tokenized by a chosen model's
tokenizer. Different models tokenize differently, so the model is a
first-class parameter.

```console
$ tokens design/tokens-cli.md
3742
```

Supports two tokenizer families, both free and offline-friendly:

- **tiktoken encodings** (OpenAI): vocabularies embedded in the binary,
  fully offline. `o200k_base` (the default), `cl100k_base`, etc., or model
  names like `gpt-4o`.
- **Hugging Face tokenizers**: any model publishing a `tokenizer.json` on
  the HF Hub (Llama, Mistral, Qwen, Gemma, …). One free download per model,
  cached forever after.

Anthropic/Claude models are deliberately out of scope — their tokenizer is
not public, and a plausible-but-wrong count is worse than a clear error.
See the [design doc](design/tokens-cli.md) for this and other decisions.

## Install

### Prebuilt binary (recommended — no Rust toolchain needed)

```sh
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/spwg/token-counter/releases/latest/download/token-counter-installer.sh | sh
```

Installs the `tokens` binary for Apple Silicon macOS, x64 Linux, or ARM64
Linux. Binaries for each platform are also attached to the
[latest release](https://github.com/spwg/token-counter/releases/latest) if
you'd rather download one directly.

### From source

Requires a Rust toolchain ([rustup](https://rustup.rs)).

```sh
git clone https://github.com/spwg/token-counter
cd token-counter
cargo install --path .
```

This builds and installs the `tokens` binary into `~/.cargo/bin`.

## Usage

All examples below run from the repo checkout and can be pasted as-is.

```console
$ tokens Cargo.toml                     # default model: o200k_base
196

$ tokens Cargo.toml design/tokens-cli.md  # multiple files, wc-style
 196 Cargo.toml
3742 design/tokens-cli.md
3938 total

$ echo "hello world" | tokens           # stdin via pipe (or explicit `-`)
3

$ tokens -m gpt-4 Cargo.toml            # tiktoken model name
195
$ tokens -m cl100k_base Cargo.toml      # tiktoken encoding name
195

$ tokens -m google/gemma-4-12B-it --json design/tokens-cli.md  # HF repo id
{
  "model": "google/gemma-4-12B-it",
  "special_tokens": false,
  "files": [
    { "path": "design/tokens-cli.md", "tokens": 3942 }
  ],
  "total": 3942
}
```

Other flags:

- `--tokenizer <path>` — load a local `tokenizer.json` directly, no network.
- `--revision <ref>` — pin an HF Hub branch, tag, or commit SHA for
  reproducible counts across machines.
- `--special-tokens` — count with the tokenizer's native special-token
  behavior (e.g. Llama 3's BOS) instead of raw-text semantics. By default
  nothing is ever added: an empty file is 0 tokens for every model.

See `tokens --help` for details on counting semantics.

### Gated models

For gated HF repos (e.g. `meta-llama/*`), set `HF_TOKEN` (or log in with the
Hugging Face CLI) and accept the model license on huggingface.co. Downloads
go to the standard HF cache (`~/.cache/huggingface/hub`), shared with Python
tooling; once fetched, a model works fully offline.

## Exit codes

| Code | Meaning                                                 |
| ---- | ------------------------------------------------------- |
| 0    | All inputs counted successfully                         |
| 1    | At least one input failed; others (if any) were counted |
| 2    | Usage or setup error — nothing was counted              |

## Development

```sh
cargo test    # entire suite runs offline — no network required
```

Design: [design/tokens-cli.md](design/tokens-cli.md)
