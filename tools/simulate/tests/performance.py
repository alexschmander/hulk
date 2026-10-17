#!/usr/bin/env python3
"""Compare headless simulator binaries. Run with uv --no-project --with jsonschema python.

Example:
  performance.py --binary before=/path/to/before --binary after=/path/to/after \
    --output logs/simulator-performance/results.json

Runs every preset at 1v1, 2v2, 3v3 and 5v5 with full recording, a fixed simulation
warm-up, and consecutive wall-time sampling windows. Robots remain in Initial.
Temporary recordings are checked after shutdown, then deleted to bound disk use.
Use --recording LABEL=off to compare MCAP disabled against LABEL=on on the same binary.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

from mcp_smoke import Client, ROOT

PROFILES = ('motion_behavior', 'filtering', 'body_state_odometry', 'localization')


def process_stats(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return {
        'cpu_seconds': (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK'),
        'rss_bytes': int(fields[21]) * os.sysconf('SC_PAGE_SIZE'),
        'threads': int(fields[17]),
    }


def worker_pid(server):
    children = {
        int(child)
        for task in Path(f'/proc/{server}/task').iterdir()
        for child in (task / 'children').read_text().split()
    }
    assert len(children) == 1, children
    return children.pop()


def cpu_pressure_total():
    line = Path('/proc/pressure/cpu').read_text().splitlines()[0]
    return int(line.rsplit('total=', 1)[1])


def run_case(binary, label, profile, players, repeat, args):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='simulator-performance-') as temporary:
        directory = Path(temporary)
        recording_enabled = args.recording.get(label, 'on') == 'on'
        configuration = {}
        if label in args.recording:
            parameter_root = directory / 'parameters'
            parameter_root.mkdir()
            (parameter_root / 'base').symlink_to(ROOT / 'etc/parameters/base', target_is_directory=True)
            location = parameter_root / 'location/hsl_middle'
            location.mkdir(parents=True)
            for source in (ROOT / 'etc/parameters/location/hsl_middle').iterdir():
                if source.name != 'simulator_recording.json5':
                    (location / source.name).symlink_to(source, target_is_directory=source.is_dir())
            (location / 'simulator_recording.json5').write_text(json.dumps({
                'mcap_enabled': recording_enabled, 'topics': ['**'],
                'queue_capacity': 8192, 'capture_level': 3,
            }))
            configuration['parameter_root'] = str(parameter_root)
        client = Client(str(binary), directory)
        try:
            client.call('server_info')
            session = client.complete('simulator_start', configuration={
                'location': 'hsl_middle',
                'profile': profile,
                'referee': 'external',
                'competition': 'large',
                'robot_count': players,
                'opponent_count': players,
                'rendering': 'disabled',
                'recording_root': str(directory / 'recordings'),
                **configuration,
            })
            sid = session['session_id']
            pid = worker_pid(client.process.pid)
            startup = time.monotonic() - started
            print(f'START {label} {profile} {players}v{players} '
                  f'repeat={repeat} startup={startup:.2f}s', flush=True)
            client.complete('simulator_control', session_id=sid,
                            command={'kind': 'set_speed', 'factor': 4})
            client.complete('simulator_control', session_id=sid, command={
                'kind': 'advance', 'duration_ns': str(round(args.warmup * 1e9)),
                'wall_timeout_ms': 120000,
            })
            client.complete('simulator_control', session_id=sid, command={'kind': 'resume'})
            previous = client.call('simulator_status', session_id=sid)
            before = process_stats(pid)
            pressure = cpu_pressure_total()
            wall_start = time.monotonic()
            samples = []
            for index in range(args.samples):
                time.sleep(args.seconds)
                current = client.call('simulator_status', session_id=sid)
                after = process_stats(pid)
                current_pressure = cpu_pressure_total()
                wall_end = time.monotonic()
                assert current['state'] == 'running', current
                simulated = (int(current['simulation_time_ns']) -
                             int(previous['simulation_time_ns'])) / 1e9
                assert simulated > 0, current
                wall = wall_end - wall_start
                cpu = after['cpu_seconds'] - before['cpu_seconds']
                sample = {
                    'wall_seconds': wall,
                    'simulation_seconds': simulated,
                    'speed': simulated / wall,
                    'cpu_seconds': cpu,
                    'cpu_cores': cpu / wall,
                    'cpu_seconds_per_simulation_second': cpu / simulated,
                    'rss_bytes': after['rss_bytes'],
                    'threads': after['threads'],
                    'host_cpu_stall_fraction': (current_pressure - pressure) / 1e6 / wall,
                }
                samples.append(sample)
                print(f'SAMPLE {label} {profile} {players}v{players} '
                      f'{index + 1}/{args.samples} speed={sample["speed"]:.3f}x '
                      f'cpu/sim={cpu / simulated:.3f}', flush=True)
                previous, before, wall_start = current, after, wall_end
                pressure = current_pressure
            client.complete('simulator_control', session_id=sid, command={'kind': 'stop'})
        finally:
            client.close()
            shutil.copyfile(directory / 'stderr.log', args.output.parent /
                            f'{label}-{profile}-{players}v{players}-{repeat}.stderr.log')

        finished = [json.loads(p.read_text()) for p in directory.rglob('finished.json')]
        assert len(finished) == 2 * players, finished
        assert not any(row['error'] for row in finished), finished
        if label in args.recording:
            assert all(row['mcap_enabled'] == recording_enabled for row in finished), finished
        recordings = list(directory.rglob('*.mcap'))
        if not recording_enabled:
            assert not recordings, recordings
        result = {
            'label': label, 'profile': profile, 'players_per_team': players,
            'repeat': repeat, 'startup_seconds': startup, 'samples': samples,
            'median_speed': statistics.median(row['speed'] for row in samples),
            'median_cpu_per_simulation_second': statistics.median(
                row['cpu_seconds_per_simulation_second'] for row in samples),
            'mcap_enabled': recording_enabled,
            'recording_files': len(recordings),
            'recording_bytes': sum(p.stat().st_size for p in recordings),
            'log_bytes': sum(p.stat().st_size for p in directory.rglob('logs.jsonl')),
            'dropped_samples': sum(row['dropped_samples'] for row in finished),
            'dropped_logs': sum(row['dropped_logs'] for row in finished),
        }
        print('RESULT ' + json.dumps({k: v for k, v in result.items() if k != 'samples'}),
              flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', action='append', required=True, metavar='LABEL=PATH')
    parser.add_argument('--recording', action='append', default=[], metavar='LABEL=on|off',
                        help='Override MCAP recording for a variant; robot logs remain enabled')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES))
    parser.add_argument('--players', nargs='+', type=int, choices=(1, 2, 3, 5),
                        default=[1, 2, 3, 5])
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--seconds', type=float, default=3)
    parser.add_argument('--warmup', type=float, default=5,
                        help='Simulation seconds to advance before sampling')
    args = parser.parse_args()
    assert args.repeats > 0 and args.samples > 0 and 0 < args.seconds <= 60
    assert args.warmup > 0
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    binaries = {}
    for option in args.binary:
        label, binary = option.split('=', 1)
        assert label and all(c.isalnum() or c in '_-' for c in label), label
        assert label not in binaries, label
        binaries[label] = Path(binary).resolve(strict=True)
    recording = {}
    for option in args.recording:
        label, mode = option.split('=', 1)
        assert label in binaries and label not in recording and mode in ('on', 'off'), option
        recording[label] = mode
    args.recording = recording
    metadata = {
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                            text=True).strip(),
        'binaries': {label: {'path': str(binary), 'sha256': hashlib.sha256(
            binary.read_bytes()).hexdigest()} for label, binary in binaries.items()},
        'configuration': {'location': 'hsl_middle', 'referee': 'external',
                          'rendering': 'disabled', 'recording': 'all topics', 'speed': 4,
                          'robot_state': 'Initial', 'warmup_simulation_seconds': args.warmup,
                          'sample_wall_seconds': args.seconds, 'samples': args.samples},
    }
    if recording:
        metadata['configuration']['recording'] = {
            label: recording.get(label, 'on') for label in binaries
        }
    results = []
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        assert previous['metadata']['binaries'] == metadata['binaries']
        assert previous['metadata']['configuration'] == metadata['configuration']
        results = previous['results']
    completed = {(r['label'], r['profile'], r['players_per_team'], r['repeat']) for r in results}
    for repeat in range(args.repeats):
        for profile in args.profiles:
            for players in args.players:
                variants = list(binaries.items())
                if (repeat + PROFILES.index(profile) + (1, 2, 3, 5).index(players)) % 2:
                    variants.reverse()
                for label, binary in variants:
                    if (label, profile, players, repeat) in completed:
                        continue
                    result = run_case(binary, label, profile, players, repeat, args)
                    results.append(result)
                    temporary = args.output.with_suffix('.json.tmp')
                    temporary.write_text(json.dumps({'metadata': metadata, 'results': results},
                                                    indent=2) + '\n')
                    temporary.replace(args.output)


if __name__ == '__main__':
    main()
