//! Production entrypoint for The Herta Voice Assistant.
//!
//! Responsibilities:
//!
//! 1. Parse CLI arguments.
//! 2. Load configuration (env + file + defaults).
//! 3. Initialize tracing / metrics.
//! 4. Dispatch to the requested subcommand.

use clap::Parser;
use herta_cli::app::Cli;

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let code = match herta_cli::commands::run(cli).await {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err:#}");
            1
        }
    };
    std::process::exit(code);
}
