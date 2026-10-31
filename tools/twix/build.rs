use std::{env, path::PathBuf};

fn main() {
    for name in [
        "MUJOCO_STATIC_LINK_DIR",
        "MUJOCO_DYNAMIC_LINK_DIR",
        "MUJOCO_DOWNLOAD_DIR",
        "PKG_CONFIG_PATH",
        "MUJOCO_NO_PKG_CONFIG",
    ] {
        println!("cargo::rerun-if-env-changed={name}");
    }
    if env::var_os("CARGO_FEATURE_SIMULATOR").is_none()
        || env::var_os("MUJOCO_STATIC_LINK_DIR").is_some()
        || !matches!(
            env::var("CARGO_CFG_TARGET_OS").as_deref(),
            Ok("linux" | "macos")
        )
    {
        return;
    }

    // Match mujoco-rs's library selection. The installed executable needs this
    // search path before main runs, so setting LD_LIBRARY_PATH in Rust is too late.
    let directories = if let Some(directory) = env::var_os("MUJOCO_DYNAMIC_LINK_DIR") {
        vec![PathBuf::from(directory)]
    } else if let Ok(library) = pkg_config::Config::new()
        .exactly_version("3.9.0")
        .cargo_metadata(false)
        .probe("mujoco")
    {
        library.link_paths
    } else if let Some(directory) = env::var_os("MUJOCO_DOWNLOAD_DIR") {
        vec![PathBuf::from(directory).join("mujoco-3.9.0/lib")]
    } else {
        Vec::new()
    };
    for directory in directories {
        println!("cargo::rustc-link-arg=-Wl,-rpath,{}", directory.display());
    }
}
