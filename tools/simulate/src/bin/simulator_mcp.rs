#[cfg(target_os = "linux")]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn main() -> color_eyre::Result<()> {
    color_eyre::install()?;
    simulate::mcp::run()
}
