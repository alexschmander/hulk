#!/usr/bin/env python3
"""Exercise the real stdio MCP server. Run with uv --with jsonschema (see README)."""
import argparse
import base64
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import tempfile
import threading
import time
import uuid

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[3]
CONTRACT = json.loads((ROOT / 'docs/tooling/simulator_mcp_contract.schema.json').read_text())
TOOLS = {t['name']: t for t in CONTRACT['x-tools']}


def validate(schema, value):
    Draft202012Validator({'$defs': CONTRACT['$defs'], **schema}).validate(value)


class Client:
    def __init__(self, binary, directory):
        self.stderr = open(directory / 'stderr.log', 'w')
        env = {**os.environ, 'SIMULATOR_MCP_ARTIFACTS': str(directory / 'artifacts')}
        self.process = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, env=env, cwd=ROOT)
        self.messages = queue.Queue()
        self.sequence = 0
        threading.Thread(target=self._read, daemon=True).start()
        self.rpc('initialize', {'protocolVersion': '2025-11-25', 'capabilities': {},
                               'clientInfo': {'name': 'simulator-mcp-smoke', 'version': '1'}})
        self.send({'jsonrpc': '2.0', 'method': 'notifications/initialized'})

    def _read(self):
        try:
            for line in self.process.stdout:
                self.messages.put(json.loads(line))
        except Exception as error:
            self.messages.put(error)
        self.messages.put(RuntimeError('MCP server exited'))

    def send(self, value):
        self.process.stdin.write(json.dumps(value) + '\n')
        self.process.stdin.flush()

    def rpc(self, method, params):
        self.sequence += 1
        self.send({'jsonrpc': '2.0', 'id': self.sequence, 'method': method, 'params': params})
        while True:
            response = self.messages.get(timeout=200)
            if isinstance(response, Exception):
                raise response
            if response.get('id') == self.sequence:
                assert 'error' not in response, response
                return response['result']

    def call(self, tool, **arguments):
        result = self.rpc('tools/call', {'name': tool, 'arguments': arguments})['structuredContent']
        validate(TOOLS[tool]['output_schema'], result)
        assert 'error' not in result, result
        return result

    def begin(self, name, **arguments):
        return self.call(name, request_id=str(uuid.uuid4()), **arguments)

    def terminal(self, operation):
        deadline = time.monotonic() + 180
        while operation['state'] in ('queued', 'running'):
            assert time.monotonic() < deadline, operation
            operation = self.call('operation_status', operation_id=operation['operation_id'], wait_ms=1000)
        if 'result' in operation:
            validate(TOOLS[operation['tool']]['completion_schema'], operation['result'])
        return operation

    def finish(self, operation):
        operation = self.terminal(operation)
        assert operation['state'] == 'succeeded', operation
        return operation['result']

    def complete(self, name, **arguments):
        return self.finish(self.begin(name, **arguments))

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=35)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.stderr.close()


def smoke(client, directory, render):
    assert len(client.rpc('tools/list', {})['tools']) == 22
    info = client.call('server_info')
    assert 'active_recording_fallback' in info['capabilities']
    start_arguments = {'request_id': 'start-once', 'configuration': {
        'location': 'hsl_middle', 'profile': 'motion_behavior', 'referee': 'automatic',
        'recording_root': str(directory / 'recordings')}}
    operation = client.call('simulator_start', **start_arguments)
    assert client.call('simulator_start', **start_arguments)['operation_id'] == operation['operation_id']
    session = client.finish(operation)
    assert session['state'] == 'paused' and session['simulation_time_ns'] == '0', session
    sid = session['session_id']
    target = session['targets'][0]['target_id']
    print('Started paused headless session', flush=True)
    graph = client.call('rosz_graph', target_id=target, limit=1000)
    assert graph['endpoints']
    (directory / 'graph.json').write_text(json.dumps(graph, indent=2))
    page = client.call('rosz_graph', target_id=target, limit=1)
    assert page['next_cursor']
    next_args = {'target_id': target, 'limit': 1, 'cursor': page['next_cursor']}
    assert client.call('rosz_graph', **next_args) == client.call('rosz_graph', **next_args)
    source = {'topic': 'primary_state'}
    schema = client.call('rosz_schema', target_id=target, kind='topic', name=source['topic'])
    assert schema['type_info']['schema_hash']
    observation = client.begin('rosz_observe', target_id=target, sources=[source])
    result = client.complete('simulator_control', session_id=sid, command={'kind': 'advance', 'steps': 100})
    assert result['advanced_steps'] == 100 and result['session']['state'] == 'paused', result
    assert int(result['session']['simulation_time_ns']) == 100 * int(session['physics_step_ns'])
    for _ in range(25):
        latest = client.call('rosz_read', target_id=target, sources=[source], query={'kind': 'latest'})
        assert latest['samples'], latest
        assert latest['samples'][0]['provenance'] == ['observation']
    end = str(int(result['session']['simulation_time_ns']) + 1)
    query = {'target_id': target, 'sources': [source], 'query': {
        'kind': 'history', 'range': {'clock': 'simulation_receipt', 'start_ns': '0', 'end_ns': end}}}
    history = client.call('rosz_read', **query)
    assert history['samples'], history
    archived = client.finish(client.call('rosz_read', **query, allow_recording_fallback=True))
    assert archived['checkpoint']['recorders'][0]['tail_status'] == 'complete_to_barrier'
    assert any(set(row['provenance']) == {'observation', 'recording'} for row in archived['samples']), archived
    assert len({row['publication_id'] for row in archived['samples']}) == len(archived['samples'])
    print('Verified exact steps, repeated live reads, and active recording deduplication', flush=True)
    condition = {'match': 'all', 'predicates': [{'target_id': target, 'source': source, 'operator': 'exists'}]}
    assert client.complete('rosz_wait', condition=condition, wall_timeout_ms=100)['matched']
    scene = client.call('scene_read', session_id=sid)
    pose = {'frame': 'world', 'position_m': [0, 1, .2], 'orientation_xyzw': [0, 0, 0, 1]}
    edited = client.complete('scene_edit', session_id=sid, expected_scene_revision=scene['scene_revision'],
                             edit={'kind': 'add_ball', 'pose': pose, 'linear_velocity_m_s': [1, 0, 0]})
    assert edited['scene_revision'] == scene['scene_revision'] + 1
    client.complete('simulator_input', session_id=sid, input={'kind': 'whistle'})
    client.call('logs_read', target_id=target, min_level='info', limit=2)
    parameters = client.call('rosz_parameter_read', target_id=target, node='behavior_node')
    assert parameters['layers'] and isinstance(parameters['values'], dict)
    name, value = next(iter(parameters['values'].items()))
    # The last layer is the simulator's private temporary override directory.
    layer = parameters['layers'][-1]
    assert str(ROOT) not in layer, layer
    client.complete('rosz_parameter_write', target_id=target, node='behavior_node', action={
        'kind': 'set', 'expected_revision': parameters['revision'],
        'writes': [{'path': name, 'value': value, 'layer': layer}]})
    service_name = 'behavior_node/parameter/get_type_info'
    service = client.call('rosz_schema', target_id=target, kind='service', name=service_name)
    reply = client.complete('rosz_call', target_id=target, service=service_name,
                            schema_hash=service['type_info']['schema_hash'], request={}, wall_timeout_ms=2000)
    assert reply['response']['success'] is True
    client.complete('rosz_publish', target_id=target, topic=source['topic'],
                    schema_hash=schema['type_info']['schema_hash'], payload=latest['samples'][0]['value'],
                    delivery={'kind': 'once'})
    print('Verified parameter transaction, dynamic service call, and dynamic publication', flush=True)
    nodes = [node['name'] for node in graph['nodes']]
    print('Discovered parameter nodes:', nodes[:10], flush=True)
    if render:
        view = client.complete('view_render', view={'kind': 'scene', 'session_id': sid,
                                                     'camera': {'kind': 'preset', 'name': 'top'}},
                                size={'width': 640, 'height': 480})
        resource = client.rpc('resources/read', {'uri': view['artifact']['uri']})['contents'][0]
        png = base64.b64decode(resource['blob'])
        assert png.startswith(b'\x89PNG\r\n\x1a\n')
        (directory / 'scene.png').write_bytes(png)
        plot = client.complete('view_render', view={'kind': 'plot', 'target_id': target,
                                                   'sources': [source], 'range': query['query']['range']})
        assert plot['artifact']['mime_type'] == 'image/png'
        clip = client.begin('video_capture', session_id=sid, duration_ns='200000000', fps=10,
                            wall_timeout_ms=10000, camera={'kind': 'preset', 'name': 'field'},
                            size={'width': 320, 'height': 240})
        deadline = time.monotonic() + 5
        while True:
            armed = client.call('operation_status', operation_id=clip['operation_id'], wait_ms=0)
            if armed.get('stage') == 'armed':
                break
            assert time.monotonic() < deadline, armed
            time.sleep(.01)
        still = client.call('simulator_status', session_id=sid)
        assert still['state'] == 'paused'
        client.complete('simulator_control', session_id=sid, command={'kind': 'advance', 'duration_ns': '220000000'})
        video = client.finish(clip)
        assert video['frame_count'] > 0 and video['artifact']['mime_type'] == 'image/gif'
        print('Verified scene image, plot, and video without implicit advancement', flush=True)
    advancing = client.begin('simulator_control', session_id=sid,
                             command={'kind': 'advance', 'steps': 100000, 'wall_timeout_ms': 30000})
    time.sleep(.05)
    client.begin('operation_cancel', operation_id=advancing['operation_id'])
    cancelled = client.terminal(advancing)
    assert cancelled['state'] == 'cancelled', cancelled
    assert client.call('simulator_status', session_id=sid)['state'] == 'paused'
    controller = client.begin('simulator_input', session_id=sid, input={
        'kind': 'controller', 'target_id': target, 'wall_duration_ms': 30000,
        'value': {'connected': True, 'device_name': 'MCP test', 'axes': [], 'buttons': []}})
    time.sleep(.05)
    client.begin('operation_cancel', operation_id=controller['operation_id'])
    assert client.terminal(controller)['state'] == 'cancelled'
    # Adding a robot refreshes the parent's target catalog.
    scene = client.call('scene_read', session_id=sid)
    client.complete('scene_edit', session_id=sid, expected_scene_revision=scene['scene_revision'],
                    edit={'kind': 'add_robot', 'team': 'hulks'})
    targets = client.call('simulator_status', session_id=sid)['targets']
    assert len(targets) == 2
    client.call('rosz_graph', target_id=next(t['target_id'] for t in targets if t['target_id'] != target))
    print('Verified cancellation, paused cleanup, and new robot discovery', flush=True)
    stopped = client.complete('simulator_control', session_id=sid, command={'kind': 'stop'})
    assert stopped['session']['state'] == 'stopped'
    retained = client.call('rosz_read', **query)['samples']
    assert all(row in retained for row in history['samples'])
    assert client.finish(client.call('rosz_read', **query, allow_recording_fallback=True))['samples']
    print('Verified stopped-session history and cleanup', flush=True)
    restarted = client.complete('simulator_start', configuration=start_arguments['configuration'])
    assert restarted['session_id'] != sid
    # Linux worker ownership is explicit: kill only the process spawned by this MCP server.
    worker_pid = None
    for task in Path(f'/proc/{client.process.pid}/task').iterdir():
        for candidate in (task / 'children').read_text().split():
            command = Path(f'/proc/{candidate}/cmdline').read_bytes()
            if b'--worker' in command:
                worker_pid = int(candidate)
    assert worker_pid is not None
    os.kill(worker_pid, signal.SIGKILL)
    crashed = client.call('simulator_status', session_id=restarted['session_id'])
    assert crashed['state'] == 'failed' and not crashed['ready'], crashed
    assert client.call('rosz_read', **query)['samples']  # Earlier completed session stays queryable.
    client.begin('rosz_unobserve', observation_id=observation['observation_id'])
    print('Verified worker-crash reporting and parent survival', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default=str(ROOT / 'target/debug/simulator-mcp'))
    parser.add_argument('--render', action='store_true', help='Also require working headless EGL rendering')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    directory = args.output or Path(tempfile.mkdtemp(prefix='simulator-mcp-smoke-'))
    directory.mkdir(parents=True, exist_ok=True)
    client = Client(args.binary, directory)
    try:
        smoke(client, directory, args.render)
    finally:
        client.close()
        print('Test artifacts:', directory, flush=True)


if __name__ == '__main__':
    main()
