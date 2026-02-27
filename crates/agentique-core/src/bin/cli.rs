use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use agentique_core::tools::{FileReadTool, FileWriteTool, ListFilesTool, ReplGetTool, ReplSetTool};
use agentique_core::{AgentLoop, SessionStore, ToolRouter};
use llm_provider::{OpenAiProvider, RetryProvider};
use observability::BudgetTracker;
use ripple_engine::ReplSession;
use tokio::sync::Mutex;

const SYSTEM_PROMPT: &str = r#"You are Agentique, a helpful coding and research assistant.

You have access to:
- File system tools (file_read, file_write, list_files) for reading, writing, and listing files.
- REPL session tools (repl_set, repl_get) for storing and retrieving named variables across turns.
  Use 'final_' prefix on variable names to mark terminal outputs (e.g., 'final_answer').

Always explain what you're doing and present results clearly."#;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    observability::init_tracing();

    let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
        anyhow::anyhow!("OPENAI_API_KEY environment variable is required")
    })?;

    let model = std::env::var("AGENTIQUE_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    let budget_dollars: f64 = std::env::var("AGENTIQUE_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);

    let home_dir = std::env::var("AGENTIQUE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs_home().join(".agentique")
        });

    let openai = OpenAiProvider::new(api_key).with_model(&model);
    let provider = RetryProvider::with_defaults(Box::new(openai));
    let budget = Arc::new(BudgetTracker::with_dollar_ceiling(budget_dollars));
    let repl_session = Arc::new(Mutex::new(ReplSession::new()));

    let mut router = ToolRouter::new();
    router.register(Box::new(FileReadTool));
    router.register(Box::new(FileWriteTool));
    router.register(Box::new(ListFilesTool));
    router.register(Box::new(ReplSetTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplGetTool::new(Arc::clone(&repl_session))));

    let session_store = SessionStore::new(&home_dir).await?;
    let session_id = session_store.session_id();

    let mut agent = AgentLoop::new(
        Box::new(provider),
        router,
        SYSTEM_PROMPT,
        &model,
        30,
        budget,
        repl_session,
    )
    .with_session_store(session_store);

    println!("Agentique Console (model: {model}, budget: ${budget_dollars:.2})");
    println!("Session: {session_id}");
    println!("Type your message and press Enter. Type 'quit' to exit.\n");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("> ");
        stdout.flush()?;

        let mut input = String::new();
        stdin.lock().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }
        if input == "quit" || input == "exit" {
            println!(
                "\nSession cost: ${:.4}. Goodbye!",
                agent.spent_dollars()
            );
            break;
        }

        match agent.process(input).await {
            Ok(response) => {
                println!("\n{response}");
                println!("[cost so far: ${:.4}]\n", agent.spent_dollars());
            }
            Err(err) => {
                eprintln!("\nError: {err}\n");
            }
        }
    }

    Ok(())
}

fn dirs_home() -> PathBuf {
    std::env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}
