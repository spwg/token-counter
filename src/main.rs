//! `tokens` — count tokens in text files (see design/tokens-cli.md).

mod counter;
mod resolve;

use std::io::{IsTerminal, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{CommandFactory, Parser};
use serde_json::json;

use counter::{Counter, HfCounter, TiktokenCounter};
use resolve::TokenizerSource;

const AFTER_HELP: &str = "\
Counting semantics:
  By default nothing is ever added by the tokenizer (HF tokenizers run with
  add_special_tokens=false; tiktoken uses encode_ordinary), so an empty file
  counts as 0 tokens for every model. Special-token strings appearing
  literally in the input diverge by family: tiktoken encodes them as plain
  characters (<|endoftext|> is ~7 tokens) while HF tokenizers recognize
  registered special tokens as single tokens.

  With --special-tokens, each tokenizer's native behavior is counted: HF
  post-processor templates are included (e.g. Llama 3 adds BOS, so an empty
  file counts 1) and tiktoken recognizes literal special-token strings such
  as <|endoftext|> as single tokens. Counts in this mode are not comparable
  across tokenizer families.";

#[derive(Parser)]
#[command(
    name = "tokens",
    version,
    about = "Count tokens in text files, as tokenized by a chosen model's tokenizer",
    after_help = AFTER_HELP
)]
struct Cli {
    /// Files to count; `-` means stdin (the default when stdin is piped)
    #[arg(value_name = "PATH")]
    paths: Vec<String>,

    /// Model or encoding name: a tiktoken model (gpt-4o), a tiktoken
    /// encoding (o200k_base), or a Hugging Face repo id (org/name)
    /// [default: o200k_base]
    #[arg(short, long, value_name = "NAME", conflicts_with = "tokenizer")]
    model: Option<String>,

    /// Load a local tokenizer.json directly (no network access)
    #[arg(long, value_name = "PATH", conflicts_with_all = ["model", "revision"])]
    tokenizer: Option<PathBuf>,

    /// HF Hub revision (branch, tag, or commit SHA) to pin [default: main]
    #[arg(long, value_name = "REF")]
    revision: Option<String>,

    /// Count with the tokenizer's native special-token behavior instead of
    /// raw-text semantics
    #[arg(long)]
    special_tokens: bool,

    /// Emit JSON instead of plain text
    #[arg(long)]
    json: bool,
}

/// The resolved tokenizer identity, as reported in `--json` output (§5.2).
struct ResolvedModel {
    id: String,
    /// Set only for HF models pinned to a non-`main` revision.
    revision: Option<String>,
}

struct FileResult {
    path: String,
    outcome: Result<usize, String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let inputs: Vec<String> = if cli.paths.is_empty() {
        if std::io::stdin().is_terminal() {
            let mut cmd = Cli::command();
            eprintln!("{}", cmd.render_usage());
            eprintln!("\nFor more information, try '--help'.");
            return ExitCode::from(2);
        }
        vec!["-".to_string()]
    } else {
        cli.paths.clone()
    };

    // Setup phase: resolve and build the tokenizer. Any failure here aborts
    // before counting anything (exit 2).
    let (counter, resolved) = match build_counter(&cli) {
        Ok(built) => built,
        Err(err) => {
            eprintln!("tokens: {err}");
            return ExitCode::from(2);
        }
    };

    // Counting phase: per-file errors are reported and don't stop the run.
    let mut results = Vec::with_capacity(inputs.len());
    for path in &inputs {
        let outcome = match read_input(path) {
            Ok(text) => Ok(counter.count(&text)),
            Err(reason) => {
                eprintln!("tokens: {path}: {reason}");
                Err(reason)
            }
        };
        results.push(FileResult {
            path: path.clone(),
            outcome,
        });
    }

    if cli.json {
        print_json(&resolved, cli.special_tokens, &results);
    } else {
        print_plain(&results);
    }

    if results.iter().any(|r| r.outcome.is_err()) {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn build_counter(cli: &Cli) -> Result<(Box<dyn Counter>, ResolvedModel)> {
    let source = match &cli.tokenizer {
        Some(path) => TokenizerSource::Local(path.clone()),
        None => {
            let name = cli.model.as_deref().unwrap_or("o200k_base");
            resolve::resolve_model(name, cli.revision.as_deref())?
        }
    };
    match source {
        TokenizerSource::Tiktoken(encoding) => Ok((
            Box::new(TiktokenCounter::new(encoding, cli.special_tokens)),
            ResolvedModel {
                id: encoding.name().to_string(),
                revision: None,
            },
        )),
        TokenizerSource::Hub { repo, revision } => {
            let path = resolve::fetch_hub_tokenizer(&repo, revision.as_deref())?;
            Ok((
                Box::new(HfCounter::from_file(&path, cli.special_tokens)?),
                ResolvedModel {
                    id: repo,
                    revision: revision.filter(|r| r != "main"),
                },
            ))
        }
        TokenizerSource::Local(path) => Ok((
            Box::new(HfCounter::from_file(&path, cli.special_tokens)?),
            ResolvedModel {
                id: path.display().to_string(),
                revision: None,
            },
        )),
    }
}

/// Read one input fully into memory and validate UTF-8 (§4, §6). Invalid
/// UTF-8 is an error, not a lossy decode.
fn read_input(path: &str) -> Result<String, String> {
    let bytes = if path == "-" {
        let mut buf = Vec::new();
        std::io::stdin()
            .read_to_end(&mut buf)
            .map_err(|e| e.to_string())?;
        buf
    } else {
        std::fs::read(path).map_err(|e| e.to_string())?
    };
    String::from_utf8(bytes).map_err(|_| "not valid UTF-8 text".to_string())
}

/// Plain output (§5.1): bare count for a single input; wc-style lines plus a
/// `total` for multiple inputs. Failed files appear only on stderr.
fn print_plain(results: &[FileResult]) {
    if let [single] = results {
        if let Ok(n) = single.outcome {
            println!("{n}");
        }
        return;
    }
    let total: usize = results.iter().filter_map(|r| r.outcome.as_ref().ok()).sum();
    let width = results
        .iter()
        .filter_map(|r| r.outcome.as_ref().ok())
        .chain(std::iter::once(&total))
        .map(|n| n.to_string().len())
        .max()
        .unwrap_or(1);
    for result in results {
        if let Ok(n) = result.outcome {
            println!("{n:>width$} {}", result.path);
        }
    }
    println!("{total:>width$} total");
}

/// JSON output (§5.2): one document regardless of input count.
fn print_json(resolved: &ResolvedModel, special_tokens: bool, results: &[FileResult]) {
    let files: Vec<serde_json::Value> = results
        .iter()
        .map(|r| match &r.outcome {
            Ok(n) => json!({ "path": r.path, "tokens": n }),
            Err(reason) => json!({ "path": r.path, "error": reason }),
        })
        .collect();
    let total: usize = results.iter().filter_map(|r| r.outcome.as_ref().ok()).sum();

    let mut doc = serde_json::Map::new();
    doc.insert("model".into(), json!(resolved.id));
    if let Some(revision) = &resolved.revision {
        doc.insert("revision".into(), json!(revision));
    }
    doc.insert("special_tokens".into(), json!(special_tokens));
    doc.insert("files".into(), json!(files));
    doc.insert("total".into(), json!(total));
    println!(
        "{}",
        serde_json::to_string_pretty(&doc).expect("JSON serialization")
    );
}
