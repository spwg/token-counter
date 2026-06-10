//! Model-name resolution (§3.1 of the design) and HF Hub tokenizer fetching (§7).

use std::path::PathBuf;

use anyhow::{Result, anyhow, bail};
use hf_hub::api::sync::ApiError;

/// A tiktoken encoding whose vocabulary is embedded in the binary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    O200kHarmony,
    O200kBase,
    Cl100kBase,
    P50kBase,
    P50kEdit,
    R50kBase,
}

impl Encoding {
    /// The canonical encoding name, used as the resolved `model` identifier.
    pub fn name(self) -> &'static str {
        match self {
            Encoding::O200kHarmony => "o200k_harmony",
            Encoding::O200kBase => "o200k_base",
            Encoding::Cl100kBase => "cl100k_base",
            Encoding::P50kBase => "p50k_base",
            Encoding::P50kEdit => "p50k_edit",
            Encoding::R50kBase => "r50k_base",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "o200k_harmony" => Encoding::O200kHarmony,
            "o200k_base" => Encoding::O200kBase,
            "cl100k_base" => Encoding::Cl100kBase,
            "p50k_base" => Encoding::P50kBase,
            "p50k_edit" => Encoding::P50kEdit,
            "r50k_base" => Encoding::R50kBase,
            _ => return None,
        })
    }

    fn from_tiktoken(t: tiktoken_rs::tokenizer::Tokenizer) -> Self {
        use tiktoken_rs::tokenizer::Tokenizer;
        match t {
            Tokenizer::O200kHarmony => Encoding::O200kHarmony,
            Tokenizer::O200kBase => Encoding::O200kBase,
            Tokenizer::Cl100kBase => Encoding::Cl100kBase,
            Tokenizer::P50kBase => Encoding::P50kBase,
            Tokenizer::P50kEdit => Encoding::P50kEdit,
            // tiktoken maps gpt2 onto the r50k vocabulary
            Tokenizer::R50kBase | Tokenizer::Gpt2 => Encoding::R50kBase,
        }
    }

    pub fn bpe(self) -> &'static tiktoken_rs::CoreBPE {
        match self {
            Encoding::O200kHarmony => tiktoken_rs::o200k_harmony_singleton(),
            Encoding::O200kBase => tiktoken_rs::o200k_base_singleton(),
            Encoding::Cl100kBase => tiktoken_rs::cl100k_base_singleton(),
            Encoding::P50kBase => tiktoken_rs::p50k_base_singleton(),
            Encoding::P50kEdit => tiktoken_rs::p50k_edit_singleton(),
            Encoding::R50kBase => tiktoken_rs::r50k_base_singleton(),
        }
    }
}

/// Where the tokenizer comes from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenizerSource {
    Tiktoken(Encoding),
    Hub {
        repo: String,
        revision: Option<String>,
    },
    Local(PathBuf),
}

/// Resolve `--model` per §3.1: tiktoken model name, then tiktoken encoding
/// name, then (if it contains `/`) an HF Hub repo id. No aliases, no fuzz.
pub fn resolve_model(name: &str, revision: Option<&str>) -> Result<TokenizerSource> {
    let tiktoken = tiktoken_rs::tokenizer::get_tokenizer(name)
        .map(Encoding::from_tiktoken)
        .or_else(|| Encoding::from_name(name));
    if let Some(encoding) = tiktoken {
        if revision.is_some() {
            bail!("--revision only applies to Hugging Face models, not '{name}'");
        }
        return Ok(TokenizerSource::Tiktoken(encoding));
    }
    if name.contains('/') {
        return Ok(TokenizerSource::Hub {
            repo: name.to_string(),
            revision: revision.map(str::to_string),
        });
    }
    bail!(
        "unknown model '{name}'\n\
         accepted: a tiktoken model (gpt-4o), a tiktoken encoding (o200k_base),\n\
         or a Hugging Face repo id (meta-llama/Meta-Llama-3-8B)"
    );
}

/// Fetch `tokenizer.json` for an HF Hub repo, cache-first (§7). Returns the
/// local path. Never fetches anything but `tokenizer.json`.
pub fn fetch_hub_tokenizer(repo: &str, revision: Option<&str>) -> Result<PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    use hf_hub::{Repo, RepoType};

    let api = ApiBuilder::from_env()
        .with_progress(false)
        .build()
        .map_err(|e| anyhow!("could not initialize Hugging Face Hub client: {e}"))?;
    let revision = revision.unwrap_or("main");
    let api_repo = api.repo(Repo::with_revision(
        repo.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    api_repo
        .get("tokenizer.json")
        .map_err(|e| hub_fetch_error(repo, &e))
}

/// Map a Hub fetch failure to a setup-error message; 401/403 get the
/// HF_TOKEN + license hint (§7).
fn hub_fetch_error(repo: &str, err: &ApiError) -> anyhow::Error {
    if is_auth_error(err) {
        anyhow!(
            "{repo}: access denied ({err})\n\
             hint: check that the repo id is spelled correctly; for gated models,\n\
             set HF_TOKEN (or log in to the Hugging Face CLI) and accept the model\n\
             license on huggingface.co"
        )
    } else {
        anyhow!("failed to fetch tokenizer.json for '{repo}': {err}")
    }
}

fn is_auth_error(err: &ApiError) -> bool {
    match err {
        ApiError::RequestError(inner) => matches!(**inner, ureq::Error::StatusCode(401 | 403)),
        ApiError::TooManyRetries(inner) => is_auth_error(inner),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiktoken_model_names_resolve_to_encodings() {
        for (model, encoding) in [
            ("gpt-4o", Encoding::O200kBase),
            ("o1", Encoding::O200kBase),
            ("gpt-4", Encoding::Cl100kBase),
            ("gpt-3.5-turbo", Encoding::Cl100kBase),
            ("text-davinci-003", Encoding::P50kBase),
            ("gpt2", Encoding::R50kBase),
        ] {
            assert_eq!(
                resolve_model(model, None).unwrap(),
                TokenizerSource::Tiktoken(encoding),
                "model {model}"
            );
        }
    }

    #[test]
    fn tiktoken_encoding_names_resolve_directly() {
        for name in [
            "o200k_base",
            "o200k_harmony",
            "cl100k_base",
            "p50k_base",
            "p50k_edit",
            "r50k_base",
        ] {
            match resolve_model(name, None).unwrap() {
                TokenizerSource::Tiktoken(e) => assert_eq!(e.name(), name),
                other => panic!("{name} resolved to {other:?}"),
            }
        }
    }

    #[test]
    fn slash_means_hub_repo() {
        assert_eq!(
            resolve_model("meta-llama/Llama-3.1-8B", None).unwrap(),
            TokenizerSource::Hub {
                repo: "meta-llama/Llama-3.1-8B".to_string(),
                revision: None,
            }
        );
        assert_eq!(
            resolve_model("Qwen/Qwen2-7B", Some("abc123")).unwrap(),
            TokenizerSource::Hub {
                repo: "Qwen/Qwen2-7B".to_string(),
                revision: Some("abc123".to_string()),
            }
        );
    }

    #[test]
    fn unknown_name_is_an_error_listing_accepted_forms() {
        let err = resolve_model("llama3", None).unwrap_err().to_string();
        assert!(err.contains("unknown model 'llama3'"), "{err}");
        assert!(err.contains("tiktoken model (gpt-4o)"), "{err}");
        assert!(err.contains("Hugging Face repo id"), "{err}");
    }

    #[test]
    fn revision_with_tiktoken_model_is_an_error() {
        let err = resolve_model("gpt-4o", Some("main"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("--revision"), "{err}");
    }

    #[test]
    fn auth_failures_get_a_token_hint_without_any_network() {
        let auth = ApiError::RequestError(Box::new(ureq::Error::StatusCode(401)));
        assert!(is_auth_error(&auth));
        let retried = ApiError::TooManyRetries(Box::new(ApiError::RequestError(Box::new(
            ureq::Error::StatusCode(403),
        ))));
        assert!(is_auth_error(&retried));
        let not_found = ApiError::RequestError(Box::new(ureq::Error::StatusCode(404)));
        assert!(!is_auth_error(&not_found));

        let msg = hub_fetch_error("meta-llama/Llama-3.1-8B", &auth).to_string();
        assert!(msg.contains("HF_TOKEN"), "{msg}");
        assert!(msg.contains("license"), "{msg}");
        let msg = hub_fetch_error("foo/bar", &not_found).to_string();
        assert!(!msg.contains("HF_TOKEN"), "{msg}");
    }
}
