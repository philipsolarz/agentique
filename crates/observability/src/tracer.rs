use tracing_subscriber::{fmt, EnvFilter};

/// Initialize the tracing subscriber with sensible defaults.
/// Respects the `RUST_LOG` env var; defaults to `info`.
pub fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    fmt().with_env_filter(filter).with_target(true).init();
}
