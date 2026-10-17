//! MCP owns worker processes; workers own physics and the robot runtimes.
mod archive;
mod clock;
mod export;
mod queries;
mod rosz;
mod server;
mod verification;
mod visual;
mod worker;

pub fn run() -> color_eyre::Result<()> {
    let arguments: Vec<_> = std::env::args().collect();
    if arguments
        .get(1)
        .is_some_and(|argument| argument == "--worker")
    {
        return worker::run(
            arguments
                .get(2)
                .ok_or_else(|| color_eyre::eyre::eyre!("Missing worker socket"))?,
        );
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(server::run())
}
