use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use agentique_core::{AgentStreamEvent, SessionBuilder};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    observability::init_tracing();

    let model = std::env::var("AGENTIQUE_MODEL").unwrap_or_else(|_| "gpt-4o".to_string());
    let budget_dollars: f64 = std::env::var("AGENTIQUE_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);

    let api_key = if model.starts_with("claude") || model.starts_with("anthropic/") {
        std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            anyhow::anyhow!("ANTHROPIC_API_KEY environment variable is required for model '{model}'")
        })?
    } else {
        std::env::var("OPENAI_API_KEY").map_err(|_| {
            anyhow::anyhow!("OPENAI_API_KEY environment variable is required for model '{model}'")
        })?
    };

    let home_dir = std::env::var("AGENTIQUE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| dirs_home().join(".agentique"));

    let built = SessionBuilder::new(&model, &api_key)
        .budget(budget_dollars)
        .data_dir(&home_dir)
        .build()
        .await?;

    let mut agent = built.agent;
    let session_id = &built.session_id;

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
