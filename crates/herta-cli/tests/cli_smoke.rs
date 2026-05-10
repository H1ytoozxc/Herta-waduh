//! CLI smoke tests.
//!
//! These tests invoke the compiled binary via `assert_cmd` to confirm the
//! top-level commands work end-to-end with default configuration and no
//! external dependencies.

use assert_cmd::Command;
use predicates::prelude::PredicateBooleanExt;
use predicates::str;

#[test]
fn help_prints_usage() {
    let mut cmd = Command::cargo_bin("herta").unwrap();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(str::contains("voice assistant"));
}

#[test]
fn version_prints() {
    let mut cmd = Command::cargo_bin("herta").unwrap();
    cmd.arg("--version").assert().success();
}

#[test]
fn show_config_yaml_is_default() {
    let mut cmd = Command::cargo_bin("herta").unwrap();
    cmd.env("HERTA_LLM_PROVIDER", "ollama")
        .arg("--no-server")
        .arg("show-config")
        .assert()
        .success()
        .stdout(str::contains("llm_provider:"));
}

#[test]
fn show_config_redacts_secrets() {
    let mut cmd = Command::cargo_bin("herta").unwrap();
    cmd.env("HERTA_DEEPSEEK__API_KEY", "super-secret-key-xyz")
        .arg("--no-server")
        .arg("show-config")
        .assert()
        .success()
        .stdout(str::contains("redacted"))
        .stdout(str::contains("super-secret-key-xyz").not());
}
