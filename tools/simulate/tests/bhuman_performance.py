#!/usr/bin/env python3
"""Paired native-opponent throughput measurements, with full recording by default.

Build bhuman-benchmark with --features bhuman, save the baseline binary and native
worker, then run --case before=BINARY,WORKER --case after=BINARY,WORKER --source
BHUMAN_SOURCE --output RESULTS.json. Run on an otherwise idle machine. Each case
uses five HULKs and five B-Humans, Localization, a fixed warmup and frame count.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import threading
import time


def process_tree(pid):
    result = {}
    pending = [pid]
    while pending:
        current = pending.pop()
        fields = Path(f'/proc/{current}/stat').read_text().rsplit(')', 1)[1].split()
        result[current] = {
            'cpu': (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK'),
            'threads': int(fields[17]),
            'rss': int(fields[21]) * os.sysconf('SC_PAGE_SIZE'),
        }
        for task in Path(f'/proc/{current}/task').iterdir():
            try:
                pending.extend(int(child) for child in (task / 'children').read_text().split())
            except FileNotFoundError:
                # Tokio can retire an idle thread while procfs is being sampled.
                continue
    return result


def pressure():
    return int(Path('/proc/pressure/cpu').read_text().splitlines()[0].rsplit('total=', 1)[1])


def validate(report):
    assert not report['recording_errors'], report['recording_errors']
    assert len(report['robots']) == 2 * report['options']['players']
    for robot in report['robots']:
        assert robot['clock_matches'] and robot['failure'] is None, robot
        if not report['options']['playing']:
            assert robot['upright'], robot
    errors = []
    yaw_errors = []
    for entry in report['localization']:
        estimate = entry['estimate']
        assert estimate and estimate['robot_to_field'] and 0 <= entry['age_seconds'] < 0.1, entry
        robot = next(robot for robot in report['robots'] if robot['id'] == entry['id'])
        x, y, z = robot['position']
        position = estimate['robot_to_field']['pose']['translation']
        error = math.dist(position, [x, -z, y])
        errors.append(error)
        assert error < 0.15, (entry['id'], error)
        if 'rotation_xyzw' in robot:
            x, y, z, w = robot['rotation_xyzw']
            truth = [x, -z, y, w]
            def yaw(q):
                x, y, z, w = q
                return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            angle = yaw(estimate['robot_to_field']['pose']['rotation']) - yaw(truth)
            yaw_error = abs(math.atan2(math.sin(angle), math.cos(angle)))
            yaw_errors.append(yaw_error)
            assert yaw_error < 0.15, (entry['id'], yaw_error)
    report['maximum_localization_position_error_m'] = max(errors, default=None)
    report['maximum_localization_yaw_error_rad'] = max(yaw_errors, default=None)
    for recording in report['recordings']:
        finished = recording['finished']
        assert finished['error'] is None and finished['dropped_samples'] == 0, recording
        assert finished['dropped_logs'] == 0, recording


def run_case(case, repeat, args):
    label, binary, worker = case
    with tempfile.TemporaryDirectory(prefix='bhuman-perf-') as temporary:
        directory = Path(temporary)
        config = {
            'worker': str(worker), 'source': str(args.source),
            'profile': args.profile, 'recording': not args.no_recording,
            'players': args.players, 'warmup_frames': args.warmup_frames,
            'frames': args.frames, 'playing': args.playing, 'location': 'hsl_middle',
        }
        configuration = directory / 'configuration.json'
        configuration.write_text(json.dumps(config))
        env = os.environ.copy()
        env['SIMULATOR_RECORDING_ROOT'] = str(directory / 'recordings')
        ort = args.source / 'Util/onnxruntime/lib/Linux/libonnxruntime.so.1.22.0'
        env['ORT_DYLIB_PATH'] = str(ort)
        env['LD_PRELOAD'] = str(ort)
        with (directory / 'report.json').open('w') as output:
            process = subprocess.Popen([str(binary), str(configuration)], stdout=output,
                                       stderr=subprocess.PIPE, text=True, env=env)
            logs = []
            before = after = None
            samples = []
            stopped = threading.Event()

            def monitor():
                while not stopped.wait(0.25):
                    try:
                        samples.append((process_tree(process.pid), pressure(), time.monotonic()))
                    except (FileNotFoundError, ProcessLookupError):
                        return

            sampler = threading.Thread(target=monitor)
            sampler.start()
            try:
                for line in process.stderr:
                    logs.append(line)
                    if line.startswith('BENCHMARK measuring'):
                        before = process_tree(process.pid), pressure(), time.monotonic()
                    elif line.startswith('BENCHMARK measured'):
                        after = process_tree(process.pid), pressure(), time.monotonic()
                code = process.wait()
            finally:
                stopped.set()
                sampler.join()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        log = args.output.with_name(f'{args.output.stem}-{label}-{repeat}.log')
        log.write_text(''.join(logs))
        assert code == 0, f'{label} exited {code}: {log}'
        report = json.loads((directory / 'report.json').read_text())
        args.output.with_name(f'{args.output.stem}-{label}-{repeat}-raw.json').write_text(json.dumps(report, indent=2))
        validate(report)
        if before and not after:
            # Older baseline binaries lack the end marker. Retain the last CPU
            # sample inside the measured interval, excluding shutdown work.
            measured = [sample for sample in samples
                        if before[2] < sample[2] <= before[2] + report['wall_seconds']]
            after = measured[-1] if measured else None
        if before and after:
            cpu = sum(value['cpu'] - before[0].get(pid, {'cpu': 0})['cpu']
                      for pid, value in after[0].items())
            fraction = min(1.0, (after[2] - before[2]) / report['wall_seconds'])
            report['cpu_seconds_per_simulation_second'] = cpu / (report['simulation_seconds'] * fraction)
            report['threads'] = sum(value['threads'] for value in after[0].values())
            report['rss_bytes'] = sum(value['rss'] for value in after[0].values())
            report['host_cpu_stall_fraction'] = (after[1] - before[1]) / 1e6 / (after[2] - before[2])
        report.update(label=label, repeat=repeat)
        print(f'{label} repeat={repeat} speed={report["speed"]:.3f}x '
              f'cpu/sim={report.get("cpu_seconds_per_simulation_second", float("nan")):.3f}', flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', required=True, metavar='LABEL=BINARY,WORKER')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--players', type=int, choices=range(1, 6), default=5)
    parser.add_argument('--profile', default='localization')
    parser.add_argument('--frames', type=int, default=15000)
    parser.add_argument('--warmup-frames', type=int, default=2500)
    parser.add_argument('--no-recording', action='store_true')
    parser.add_argument('--playing', action='store_true')
    parser.add_argument('--require-speedup', nargs=3, metavar=('BEFORE', 'AFTER', 'FACTOR'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cases = []
    for case in args.case:
        label, paths = case.split('=', 1)
        binary, worker = (Path(path).resolve(strict=True) for path in paths.split(','))
        cases.append((label, binary, worker))
    manifest = {label: {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in [('simulator', binary), ('worker', worker)]}
                for label, binary, worker in cases}
    results = []
    for repeat in range(args.repeats):
        for case in cases if repeat % 2 == 0 else reversed(cases):
            results.append(run_case(case, repeat, args))
            args.output.write_text(json.dumps({'binaries': manifest, 'results': results}, indent=2))
    medians = {}
    for label, _, _ in cases:
        speeds = [row['speed'] for row in results if row['label'] == label]
        medians[label] = statistics.median(speeds)
        print(f'{label} median={medians[label]:.3f}x', flush=True)
    if args.require_speedup:
        before, after, factor = args.require_speedup
        speedup = medians[after] / medians[before]
        print(f'{before} -> {after}: {speedup:.3f}x throughput', flush=True)
        assert speedup >= float(factor), f'{speedup:.3f}x < {factor}x target'


if __name__ == '__main__':
    main()
