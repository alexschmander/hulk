#!/usr/bin/env bash
# Build the upstream GUI's unmodified runtime and test real UDP in isolated networks.
set -euo pipefail
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export HSL_CHECKOUT=$(realpath "${1:?Usage: roundtrip.sh /path/to/HSL-GameController}")
export HSL_EXCHANGE=$(mktemp -d)
trap 'cp -r "$HSL_EXCHANGE/logs" "$root/target/hsl-roundtrip/last-logs" 2>/dev/null || true; rm -rf "$HSL_EXCHANGE"' EXIT
export MUJOCO_DOWNLOAD_DIR="${MUJOCO_DOWNLOAD_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/mujoco-rs}"
export LD_LIBRARY_PATH="$MUJOCO_DOWNLOAD_DIR/mujoco-3.9.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 - "$root" <<'PY'
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
tmp = pathlib.Path(os.environ['HSL_EXCHANGE'])
gc = pathlib.Path(os.environ['HSL_CHECKOUT'])
(tmp / 'Cargo.toml').write_text('''[package]
name = "hsl-roundtrip"
version = "0.0.0"
edition = "2021"
[[bin]]
name = "hsl-roundtrip"
path = ''' + json.dumps(str(root / 'tools/simulate/tests/hsl/driver.rs')) + '''
[dependencies]
anyhow = "1"
clap = { version = "4", features = ["derive"] }
serde_json = "1"
tokio = { version = "1", features = ["full"] }
''' + '\n'.join(f'{name} = {{path = {json.dumps(str(gc / name))}}}' for name in ['game_controller_core', 'game_controller_runtime']))
PY
cargo build --manifest-path "$HSL_EXCHANGE/Cargo.toml" --target-dir "$root/target/hsl-roundtrip"
cargo test --manifest-path "$root/Cargo.toml" -p simulate --lib --no-run --message-format=json > "$HSL_EXCHANGE/build.json"
robot=$(python3 - "$HSL_EXCHANGE/build.json" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    item=json.loads(line)
    if item.get('reason') == 'compiler-artifact' and item.get('executable') and item['target']['name'] == 'simulate':
        print(item['executable'])
PY
)
# All ip commands execute inside a new user + network namespace, never on the host.
unshare --user --map-root-user --net bash -s -- "$robot" "$root/target/hsl-roundtrip/debug/hsl-roundtrip" <<'SH'
set -euo pipefail
unshare --net sleep 90 &
gc_net=$!
trap 'kill "$gc_net" ${gc_pid:-} 2>/dev/null || true' EXIT
ip link set lo up
ip link add robot type veth peer name gc
ip link set gc netns "$gc_net"
ip addr add 10.0.0.1/16 brd + dev robot
ip link set robot up
nsenter -t "$gc_net" -n ip link set lo up
nsenter -t "$gc_net" -n ip addr add 10.0.0.2/16 brd + dev gc
nsenter -t "$gc_net" -n ip link set gc up
nsenter -t "$gc_net" -n "$2" &
gc_pid=$!
"$1" upstream_hsl_game_controller_roundtrip --ignored --nocapture
wait "$gc_pid"
SH
