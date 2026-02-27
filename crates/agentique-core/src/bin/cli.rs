use std::io::{self, BufRead, Write};
use std::sync::Arc;

use agentique_core::tools::{FileReadTool, FileWriteTool, ListFilesTool};
use agentique_core::{AgentLoop, ToolRouter};
use llm_provider::OpenAiProvider;
use observability::BudgetTracker;

const SYSTEM_PROMPT: &str = r#"You are Agentique, a helpful coding and research assistant.
You have access to file system tools. Use them when the user asks you to read, write, or list files.
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

    let provider = OpenAiProvider::new(api_key).with_model(&model);
    let budget = Arc::new(BudgetTracker::with_dollar_ceiling(budget_dollars));

    let mut router = ToolRouter::new();
    router.register(Box::new(FileReadTool));
    router.register(Box::new(FileWriteTool));
    router.register(Box::new(ListFilesTool));

    let mut agent = AgentLoop::new(
        Box::new(provider),
        router,
        SYSTEM_PROMPT,
        &model,
        30,
        budget,
    );

    println!("Agentique Console (model: {model}, budget: ${budget_dollars:.2})");
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
