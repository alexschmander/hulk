fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    simulate::mcp::run()
}
