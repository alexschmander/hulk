#[cfg(unix)]
fn main() -> color_eyre::Result<()> {
    use std::{path::PathBuf, time::Duration};
    color_eyre::install()?;
    let mut args = std::env::args_os().skip(1);
    let first = args.next().ok_or_else(|| {
        color_eyre::eyre::eyre!(
            "Usage: bhuman-motion-smoke [--soccer] WORKER BHUMAN_SOURCE [FRAMES]"
        )
    })?;
    let soccer = first == "--soccer";
    let executable = PathBuf::from(if soccer { args.next() } else { Some(first) }.ok_or_else(
        || color_eyre::eyre::eyre!("Usage: bhuman-motion-smoke WORKER BHUMAN_SOURCE [FRAMES]"),
    )?);
    let source = PathBuf::from(
        args.next()
            .ok_or_else(|| color_eyre::eyre::eyre!("Missing B-Human source checkout"))?,
    );
    let frames = args
        .next()
        .map(|value| value.to_string_lossy().parse::<u64>())
        .transpose()?
        .unwrap_or(if soccer { 3000 } else { 1500 });
    color_eyre::eyre::ensure!(args.next().is_none(), "Unexpected argument");
    let report = simulate::bhuman::run_smoke(
        simulate::bhuman::Configuration {
            executable,
            source,
            player: if soccer { 2 } else { 1 },
            team: 5,
            cognition: soccer,
            active: true,
            field: if soccer {
                Some(simulate::FieldConfiguration::load(
                    &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../etc/parameters"),
                    "incheon_small",
                )?)
            } else {
                None
            },
            timeout: Duration::from_secs(30),
        },
        frames,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(not(unix))]
fn main() {
    eprintln!("The B-Human adapter currently requires Unix sockets.");
    std::process::exit(1);
}
