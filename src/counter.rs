//! The `Counter` trait (§8.2) and its tiktoken / Hugging Face implementations.
//!
//! Counting semantics (§4): the mode is fixed at construction. Default
//! (raw-text) mode adds nothing — `encode_ordinary` for tiktoken,
//! `add_special_tokens = false` for HF — so an empty file is 0 tokens for
//! every model. `--special-tokens` switches to each tokenizer's native
//! behavior.

use std::path::Path;

use anyhow::{Result, anyhow};

use crate::resolve::Encoding;

pub trait Counter {
    /// Token count of `text`. The counting mode (§4: raw-text vs
    /// --special-tokens) is fixed when the Counter is constructed.
    fn count(&self, text: &str) -> usize;
}

pub struct TiktokenCounter {
    bpe: &'static tiktoken_rs::CoreBPE,
    special_tokens: bool,
}

impl TiktokenCounter {
    pub fn new(encoding: Encoding, special_tokens: bool) -> Self {
        Self {
            bpe: encoding.bpe(),
            special_tokens,
        }
    }
}

impl Counter for TiktokenCounter {
    fn count(&self, text: &str) -> usize {
        if self.special_tokens {
            self.bpe.encode_with_special_tokens(text).len()
        } else {
            self.bpe.encode_ordinary(text).len()
        }
    }
}

pub struct HfCounter {
    tokenizer: tokenizers::Tokenizer,
    special_tokens: bool,
}

impl HfCounter {
    pub fn from_file(path: &Path, special_tokens: bool) -> Result<Self> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow!("{}: not a usable tokenizer.json: {e}", path.display()))?;
        Ok(Self {
            tokenizer,
            special_tokens,
        })
    }
}

impl Counter for HfCounter {
    fn count(&self, text: &str) -> usize {
        // `add_special_tokens` controls only post-processor templates (e.g.
        // BOS); registered special tokens appearing literally in the input
        // are extracted as single tokens in both modes (§4.1).
        self.tokenizer
            .encode(text, self.special_tokens)
            .expect("tokenizer failed to encode text")
            .get_ids()
            .len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    #[test]
    fn empty_input_is_zero_in_default_mode_for_both_families() {
        assert_eq!(
            TiktokenCounter::new(Encoding::O200kBase, false).count(""),
            0
        );
        assert_eq!(
            TiktokenCounter::new(Encoding::Cl100kBase, false).count(""),
            0
        );
        let hf = HfCounter::from_file(&fixture("bos.json"), false).unwrap();
        assert_eq!(hf.count(""), 0);
    }

    #[test]
    fn tiktoken_default_mode_treats_literal_special_tokens_as_plain_text() {
        let counter = TiktokenCounter::new(Encoding::Cl100kBase, false);
        let n = counter.count("<|endoftext|>");
        assert!(n > 1, "expected several plain-text tokens, got {n}");
    }

    #[test]
    fn tiktoken_special_mode_recognizes_literal_special_tokens() {
        let counter = TiktokenCounter::new(Encoding::Cl100kBase, true);
        assert_eq!(counter.count("<|endoftext|>"), 1);
        // ...but still adds nothing: empty input stays 0.
        assert_eq!(counter.count(""), 0);
    }

    #[test]
    fn hf_extracts_registered_special_tokens_even_in_default_mode() {
        // Pins the documented `tokenizers` crate behavior (§4.1): the
        // add_special_tokens flag does not stop extraction of registered
        // special tokens found literally in the input.
        let counter = HfCounter::from_file(&fixture("special.json"), false).unwrap();
        assert_eq!(counter.count("hello <|test|> world"), 3);
        assert_eq!(counter.count("<|test|>"), 1);
    }

    #[test]
    fn hf_special_mode_counts_post_processor_templates() {
        // bos.json has a TemplateProcessing post-processor that prepends <s>.
        let with = HfCounter::from_file(&fixture("bos.json"), true).unwrap();
        let without = HfCounter::from_file(&fixture("bos.json"), false).unwrap();
        assert_eq!(with.count(""), 1, "empty input counts the BOS");
        assert_eq!(without.count(""), 0);
        assert_eq!(with.count("hello world"), 3);
        assert_eq!(without.count("hello world"), 2);
    }

    #[test]
    fn known_strings_have_stable_counts() {
        // Hardcoded expected counts for embedded encodings (§8.3).
        let o200k = TiktokenCounter::new(Encoding::O200kBase, false);
        let cl100k = TiktokenCounter::new(Encoding::Cl100kBase, false);
        for (text, o200k_count, cl100k_count) in [
            ("hello world", 2, 2),
            ("Hello, world!", 4, 4),
            ("The quick brown fox jumps over the lazy dog.", 10, 10),
            ("你好，世界", 3, 6),
        ] {
            assert_eq!(o200k.count(text), o200k_count, "o200k_base: {text:?}");
            assert_eq!(cl100k.count(text), cl100k_count, "cl100k_base: {text:?}");
        }
    }
}
