//! Integration tests (§8.3). The whole suite must pass with no network
//! access: HF behavior is exercised only through `--tokenizer` fixtures, and
//! tiktoken vocabularies are embedded in the binary.

use std::fs;
use std::path::{Path, PathBuf};

use assert_cmd::Command;
use predicates::prelude::*;

fn tokens() -> Command {
    Command::cargo_bin("tokens").unwrap()
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn write(dir: &Path, name: &str, contents: &[u8]) -> PathBuf {
    let path = dir.join(name);
    fs::write(&path, contents).unwrap();
    path
}

#[test]
fn single_file_prints_bare_count() {
    let dir = tempfile::tempdir().unwrap();
    let a = write(dir.path(), "a.txt", b"hello world");
    tokens().arg(&a).assert().success().stdout("2\n").stderr("");
}

#[test]
fn multiple_files_print_wc_style_lines_and_total() {
    let dir = tempfile::tempdir().unwrap();
    let a = write(dir.path(), "a.txt", b"hello world"); // 2 tokens
    let b = write(dir.path(), "b.txt", "hello world ".repeat(60).as_bytes()); // 121 tokens
    let expected = format!("  2 {}\n121 {}\n123 total\n", a.display(), b.display());
    tokens()
        .arg(&a)
        .arg(&b)
        .assert()
        .success()
        .stdout(expected)
        .stderr("");
}

#[test]
fn stdin_via_dash() {
    tokens()
        .arg("-")
        .write_stdin("hello world")
        .assert()
        .success()
        .stdout("2\n");
}

#[test]
fn stdin_via_pipe_with_no_paths() {
    tokens()
        .write_stdin("hello world")
        .assert()
        .success()
        .stdout("2\n");
}

#[test]
fn stdin_is_shown_as_dash_in_multi_file_output() {
    let dir = tempfile::tempdir().unwrap();
    let a = write(dir.path(), "a.txt", b"hello world");
    tokens()
        .arg(&a)
        .arg("-")
        .write_stdin("hello")
        .assert()
        .success()
        .stdout(format!("2 {}\n1 -\n3 total\n", a.display()));
}

#[test]
fn empty_input_counts_zero() {
    tokens().write_stdin("").assert().success().stdout("0\n");
}

#[test]
fn json_output_shape() {
    let dir = tempfile::tempdir().unwrap();
    let a = write(dir.path(), "a.txt", b"hello world");
    let output = tokens().arg("--json").arg(&a).assert().success();
    let doc: serde_json::Value = serde_json::from_slice(&output.get_output().stdout).unwrap();
    assert_eq!(doc["model"], "o200k_base");
    assert_eq!(doc["special_tokens"], false);
    assert_eq!(doc["total"], 2);
    let files = doc["files"].as_array().unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0]["path"], a.display().to_string());
    assert_eq!(files[0]["tokens"], 2);
    assert!(doc.get("revision").is_none());
}

#[test]
fn json_records_special_tokens_mode_and_per_file_errors() {
    let dir = tempfile::tempdir().unwrap();
    let a = write(dir.path(), "a.txt", b"hello world");
    let missing = dir.path().join("missing.txt");
    let output = tokens()
        .arg("--json")
        .arg("--special-tokens")
        .arg(&a)
        .arg(&missing)
        .assert()
        .code(1);
    let doc: serde_json::Value = serde_json::from_slice(&output.get_output().stdout).unwrap();
    assert_eq!(doc["special_tokens"], true);
    assert_eq!(doc["total"], 2, "total sums successful files only");
    let files = doc["files"].as_array().unwrap();
    assert_eq!(files[0]["tokens"], 2);
    assert!(files[1].get("tokens").is_none());
    assert!(files[1]["error"].is_string());
}

#[test]
fn missing_file_reports_stderr_and_exits_1_but_others_count() {
    let dir = tempfile::tempdir().unwrap();
    let a = write(dir.path(), "a.txt", b"hello world");
    let missing = dir.path().join("missing.txt");
    tokens()
        .arg(&missing)
        .arg(&a)
        .assert()
        .code(1)
        .stdout(predicate::str::contains("2 ").and(predicate::str::contains("total")))
        .stderr(predicate::str::contains(format!(
            "tokens: {}: ",
            missing.display()
        )));
}

#[test]
fn invalid_utf8_is_an_error_not_a_lossy_decode() {
    let dir = tempfile::tempdir().unwrap();
    let bad = write(dir.path(), "bad.bin", &[0xff, 0xfe, 0x80, 0x00]);
    tokens()
        .arg(&bad)
        .assert()
        .code(1)
        .stdout("")
        .stderr(predicate::str::contains("not valid UTF-8 text"));
}

#[test]
fn unknown_model_is_a_setup_error_exit_2() {
    tokens()
        .arg("-m")
        .arg("llama3")
        .write_stdin("hello")
        .assert()
        .code(2)
        .stdout("")
        .stderr(
            predicate::str::contains("tokens: unknown model 'llama3'")
                .and(predicate::str::contains("accepted:")),
        );
}

#[test]
fn model_and_tokenizer_flags_conflict() {
    tokens()
        .arg("-m")
        .arg("gpt-4o")
        .arg("--tokenizer")
        .arg(fixture("bos.json"))
        .write_stdin("hello")
        .assert()
        .code(2);
}

#[test]
fn revision_and_tokenizer_flags_conflict() {
    tokens()
        .arg("--revision")
        .arg("abc")
        .arg("--tokenizer")
        .arg(fixture("bos.json"))
        .write_stdin("hello")
        .assert()
        .code(2);
}

#[test]
fn revision_with_tiktoken_model_is_a_setup_error() {
    tokens()
        .arg("-m")
        .arg("gpt-4o")
        .arg("--revision")
        .arg("abc")
        .write_stdin("hello")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("--revision"));
}

#[test]
fn unreadable_tokenizer_file_is_a_setup_error() {
    tokens()
        .arg("--tokenizer")
        .arg("/nonexistent/tokenizer.json")
        .write_stdin("hello")
        .assert()
        .code(2)
        .stdout("")
        .stderr(predicate::str::contains("tokens: "));
}

#[test]
fn local_tokenizer_counts_and_reports_path_as_model() {
    let bos = fixture("bos.json");
    let output = tokens()
        .arg("--tokenizer")
        .arg(&bos)
        .arg("--json")
        .write_stdin("hello world")
        .assert()
        .success();
    let doc: serde_json::Value = serde_json::from_slice(&output.get_output().stdout).unwrap();
    assert_eq!(doc["model"], bos.display().to_string());
    assert_eq!(doc["total"], 2);
}

#[test]
fn special_tokens_flag_counts_hf_post_processor_templates() {
    // Empty input: 0 by default, 1 (the BOS) with --special-tokens.
    tokens()
        .arg("--tokenizer")
        .arg(fixture("bos.json"))
        .write_stdin("")
        .assert()
        .success()
        .stdout("0\n");
    tokens()
        .arg("--tokenizer")
        .arg(fixture("bos.json"))
        .arg("--special-tokens")
        .write_stdin("")
        .assert()
        .success()
        .stdout("1\n");
}

#[test]
fn tiktoken_special_tokens_flag_recognizes_literal_specials() {
    tokens()
        .arg("-m")
        .arg("cl100k_base")
        .write_stdin("<|endoftext|>")
        .assert()
        .success()
        .stdout("7\n");
    tokens()
        .arg("-m")
        .arg("cl100k_base")
        .arg("--special-tokens")
        .write_stdin("<|endoftext|>")
        .assert()
        .success()
        .stdout("1\n");
}
