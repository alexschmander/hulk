#[cfg(target_os = "linux")]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

#[cfg(unix)]
fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    let mut args = std::env::args_os().skip(1);
    let options = args
        .next()
        .map(
            |path| -> color_eyre::Result<simulate::bhuman::BenchmarkOptions> {
                Ok(serde_json::from_slice(&std::fs::read(path)?)?)
            },
        )
        .transpose()?
        .unwrap_or_default();
    color_eyre::eyre::ensure!(
        args.next().is_none(),
        "Usage: bhuman-benchmark [CONFIGURATION.json]"
    );
    println!(
        "{}",
        serde_json::to_string_pretty(&simulate::bhuman::run_benchmark(options)?)?
    );
    Ok(())
}

#[cfg(not(unix))]
fn main() {
    eprintln!("The B-Human benchmark requires Unix sockets.");
    std::process::exit(1);
}
