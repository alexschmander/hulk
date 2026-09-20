#!/usr/bin/env python3
"""Run isolated head gain experiments through the ordinary upload/test helper."""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent


def gain(value):
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
        raise argparse.ArgumentTypeError("gain must be a nonnegative decimal number")
    if not math.isfinite(float(value)) or float(value) > 3.4028234e38:
        raise argparse.ArgumentTypeError("gain must fit in a finite f32")
    return value


def seconds(value):
    number = int(value)
    if not 1 <= number <= 600:
        raise argparse.ArgumentTypeError("duration must be 1..600 seconds")
    return number


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("robot", help="robot IP or hostname")
    parser.add_argument("--joint", choices=["yaw", "pitch"], default="yaw")
    parser.add_argument("--kp", nargs="+", type=gain, default=["10", "12"])
    parser.add_argument("--kd", nargs="+", type=gain, default=["1.2", "1.4"])
    parser.add_argument("--hold-seconds", type=seconds, default=15)
    parser.add_argument("--scan-seconds", type=seconds, default=30)
    parser.add_argument("--dry-run", action="store_true", help="print plan without contacting robot")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9.-]*", args.robot):
        parser.error("invalid robot address")
    return args


def plan(args, campaign_id):
    steps = []
    for kd in args.kd:
        for kp in args.kp:
            gains = {"yaw_kp": "10", "yaw_kd": "1.2", "pitch_kp": "10", "pitch_kd": "1.2"}
            gains[f"{args.joint}_kp"] = kp
            gains[f"{args.joint}_kd"] = kd
            # Same waypoints as the production LookAround path used by `scan`.
            for label, pattern, duration, yaw, pitch in [
                ("center", "hold", args.hold_seconds, "0", "0.7"),
                ("left", "hold", args.hold_seconds, "0.95", "0.5"),
                ("right", "hold", args.hold_seconds, "-0.95", "0.5"),
                ("scan", "scan", args.scan_seconds, "0", "0.7"),
            ]:
                run_id = f"{campaign_id}-{len(steps) + 1:03d}-{label}"
                command = [str(ROOT / "scripts/head-only-test"), "run", args.robot,
                           pattern, str(duration), yaw, pitch, "--run-id", run_id]
                for key, value in gains.items():
                    command.extend(["--" + key.replace("_", "-"), value])
                steps.append({"run_id": run_id, "pattern": pattern, "waypoint": label,
                              "seconds": duration, "yaw": float(yaw), "pitch": float(pitch),
                              "gains": {key: float(value) for key, value in gains.items()},
                              "command": command, "status": "planned"})
    return steps


def run_process(command):
    # Separate process group lets Ctrl-C reach the complete active SSH/run chain.
    with subprocess.Popen(command, start_new_session=True) as process:
        try:
            return process.wait()
        except KeyboardInterrupt:
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise


def save(path, manifest):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def read_result(step, destination):
    result_path = destination / step["run_id"] / "result.json"
    if result_path.exists():
        step["result"] = json.loads(result_path.read_text())
    result = step.get("result", {})
    if not isinstance(result, dict):
        return False
    complete = result.get("stop_reason") == "completed" and all(
        key in result and result[key] is None
        for key in ["test_error", "damping_error", "recording_error"]
    )
    return complete


def main(argv=None):
    args = arguments(argv)
    campaign_id = datetime.now(timezone.utc).strftime("campaign-%Y%m%dT%H%M%S%f")
    steps = plan(args, campaign_id)
    active_seconds = sum(step["seconds"] for step in steps)
    print(f"{args.joint} campaign: {len(steps)} runs, {active_seconds / 60:.1f} minutes active "
          "plus startup, transfers, and 2 seconds damping between runs.", flush=True)
    for step in steps:
        print(shlex.join(step["command"]), flush=True)
    if args.dry_run:
        return 0

    kit = ROOT / "target/head-only-kit"
    if not (kit / "build-id").exists():
        raise RuntimeError("Run scripts/head-only-test prepare first")
    build_id = (kit / "build-id").read_text().strip()
    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}", build_id):
        raise RuntimeError("Invalid prepared build ID")
    if (kit / "run").read_bytes() != (ROOT / "scripts/run-head-only-on-robot").read_bytes():
        raise RuntimeError("Test runner changed; run scripts/head-only-test prepare first")

    destination = ROOT / "logs/head-only" / args.robot / build_id
    manifest_path = destination / "campaigns" / f"{campaign_id}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"campaign_id": campaign_id, "build_id": build_id, "robot": args.robot,
                "joint": args.joint, "status": "running", "steps": steps}
    save(manifest_path, manifest)
    helper = str(ROOT / "scripts/head-only-test")
    print("Robot must be supported; leave head untouched. Ctrl-C stops the campaign and requests damping.", flush=True)
    exit_code = 0
    current = None
    try:
        for index, current in enumerate(steps):
            if (kit / "build-id").read_text().strip() != build_id:
                raise RuntimeError("Prepared kit changed during campaign; refusing a mixed-build test")
            current["status"] = "running"
            save(manifest_path, manifest)
            print(f"Run {index + 1}/{len(steps)}: {current['run_id']} {current['gains']}", flush=True)
            status = run_process(current["command"])
            current["exit_code"] = status
            current["status"] = "executed" if status == 0 else "failed"
            save(manifest_path, manifest)
            if status != 0:
                manifest["status"] = "failed"
                exit_code = 1
                break
            # A signal can stop the binary cleanly with exit 0. Check its result
            # before starting another run, so `stop` cannot silently rearm it.
            if (kit / "build-id").read_text().strip() != build_id:
                raise RuntimeError("Prepared kit changed during campaign")
            current["fetch_exit_code"] = run_process([helper, "fetch", args.robot])
            if current["fetch_exit_code"] != 0 or not read_result(current, destination):
                current["status"] = "incomplete"
                manifest["status"] = "incomplete"
                exit_code = 1
                break
            current["status"] = "completed"
            save(manifest_path, manifest)
            if index + 1 < len(steps):
                time.sleep(2)
        else:
            manifest["status"] = "completed"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        if current is not None and current["status"] == "running":
            current["status"] = "interrupted"
        exit_code = 130
        manifest["stop_exit_code"] = run_process([helper, "stop", args.robot])
    except (OSError, RuntimeError, ValueError) as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        if current is not None and current["status"] in {"running", "executed"}:
            current["status"] = "incomplete"
        exit_code = 1
    finally:
        save(manifest_path, manifest)

    # Successful runs are already downloaded. Recover partial recordings after a stop.
    try:
        if (kit / "build-id").read_text().strip() != build_id:
            raise RuntimeError(f"Kit changed: retrieve robot directory head-only-tests/{build_id}/logs manually")
        if manifest["status"] != "completed":
            manifest["fetch_exit_code"] = run_process([helper, "fetch", args.robot])
            if manifest["fetch_exit_code"] != 0:
                raise RuntimeError("Recording download failed; retry scripts/head-only-test fetch")
        for step in steps:
            if step["status"] != "planned":
                read_result(step, destination)
    except (OSError, RuntimeError, ValueError, KeyboardInterrupt) as error:
        manifest["fetch_error"] = str(error) or "Download interrupted"
        exit_code = exit_code or 1
        if manifest["status"] == "completed":
            manifest["status"] = "incomplete"
    finally:
        save(manifest_path, manifest)
        print(f"Campaign {manifest['status']}. Manifest: {manifest_path}", flush=True)
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
