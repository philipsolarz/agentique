use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;

use agentique_core::tools::{
    FileReadTool, FileWriteTool, ListFilesTool, ReplChunksTool, ReplGetTool, ReplLenTool,
    ReplLoadFileTool, ReplSearchTool, ReplSetTool, ReplSliceTool,
};
use agentique_core::{AgentLoop, AgentStreamEvent, SessionStore, ToolRouter};
use llm_provider::{AnthropicProvider, CompletionProvider, OpenAiProvider, RetryProvider};
use observability::BudgetTracker;
use ripple_engine::ReplSession;
use tokio::sync::Mutex;

const SYSTEM_PROMPT: &str = r#"You are Agentique, a helpful coding and research assistant.

You have access to:
- File system tools (file_read, file_write, list_files) for working with files.
- REPL session tools for managing variables:
  - repl_set / repl_get: Store and retrieve named variables.
  - repl_load_file: Load a file into the REPL as a symbolic variable (you see metadata, not content).
  - repl_slice: Extract a character range from a variable.
  - repl_search: Regex search within a variable, returns matches with line numbers.
  - repl_chunks: Split a large variable into smaller chunks for processing.
  - repl_len: Get size and token info about a variable.

For large files or codebases, use repl_load_file to load them, then use repl_search/repl_slice/repl_chunks
to work with specific parts. This keeps large content out of the conversation context.
Use 'final_' prefix on variable names to mark terminal outputs (e.g., 'final_answer').

Always explain what you're doing and present results clearly."#;

/// Create a provider based on the model name and available API keys.
/// Models starting with "claude" or "anthropic/" use Anthropic; everything else uses OpenAI.
fn create_provider(model: &str) -> anyhow::Result<Box<dyn CompletionProvider>> {
    let is_anthropic = model.starts_with("claude") || model.starts_with("anthropic/");

    if is_anthropic {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            anyhow::anyhow!(
                "ANTHROPIC_API_KEY environment variable is required for model '{model}'"
            )
        })?;
        let provider = AnthropicProvider::new(api_key).with_model(model);
        Ok(Box::new(RetryProvider::with_defaults(Box::new(provider))))
    } else {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            anyhow::anyhow!("OPENAI_API_KEY environment variable is required for model '{model}'")
        })?;
        let provider = OpenAiProvider::new(api_key).with_model(model);
        Ok(Box::new(RetryProvider::with_defaults(Box::new(provider))))
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    observability::init_tracing();

    let model = std::env::var("AGENTIQUE_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    let budget_dollars: f64 = std::env::var("AGENTIQUE_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);

    let home_dir = std::env::var("AGENTIQUE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs_home().join(".agentique"));

    let provider = create_provider(&model)?;
    let budget = Arc::new(BudgetTracker::with_dollar_ceiling(budget_dollars));
    let repl_session = Arc::new(Mutex::new(ReplSession::new()));

    let mut router = ToolRouter::new();
    router.register(Box::new(FileReadTool));
    router.register(Box::new(FileWriteTool));
    router.register(Box::new(ListFilesTool));
    router.register(Box::new(ReplSetTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplGetTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplLoadFileTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplSliceTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplSearchTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplChunksTool::new(Arc::clone(&repl_session))));
    router.register(Box::new(ReplLenTool::new(Arc::clone(&repl_session))));

    let session_store = SessionStore::new(&home_dir).await?;
    let session_id = session_store.session_id();

    let mut agent = AgentLoop::new(
        provider,
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

        match agent
            .process_streaming(input, |event| match &event {
                AgentStreamEvent::Token(token) => {
                    print!("{token}");
                    let _ = io::stdout().flush();
                }
                AgentStreamEvent::ToolCallStart(name) => {
                    print!("\n[calling {name}...]");
                    let _ = io::stdout().flush();
                }
                AgentStreamEvent::ToolCallEnd(name) => {
                    println!(" [{name} done]");
                }
                AgentStreamEvent::Done(_) => {
                    println!();
                }
            })
            .await
        {
            Ok(response) => {
                println!("[cost so far: ${:.4}]\n", agent.spent_dollars());
                let _ = response;
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
