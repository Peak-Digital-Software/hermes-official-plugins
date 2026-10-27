"""Advisor server-tool blocks survive the relay's capture and the native turn parser.

An advisor turn carries a ``server_tool_use`` block named ``advisor`` and a fully formed
``advisor_tool_result`` block in the assistant message. ``Capture`` understands only the ordinary
text/thinking/tool_use deltas, and ``directsdk`` projects only ``tool_use`` blocks for Hermes, so
both must ignore the server-tool blocks while retaining them in the native turn unchanged. Shapes
follow the advisor tool documentation:
https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import directsdk
from admission import Capture

ADVISOR_ID = 'srvtoolu_01AdvisorFixture'
TOOL_ID = 'toolu_01WritePlan'
TEXT_BEFORE = 'Let me consult the advisor on this.'
TEXT_AFTER = 'The advisor suggests a channel-based design. Writing the plan now.'
ENCRYPTED = 'EqQBCkYIBRgCIiQ3YTAwMjY1Mi1mZjM5LTQ1NGUtODgxNC1kNjNjNTk1ZWI3Y...'
TOOL_INPUT = {'path': 'plan.md', 'content': '# plan'}
TOOLS = [{'type': 'function', 'function': {
    'name': 'write_file', 'description': 'write a file',
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'},
                                                     'content': {'type': 'string'}}}}}]

USAGE = {'input_tokens': 21, 'cache_read_input_tokens': 4, 'cache_creation_input_tokens': 0,
         'output_tokens': 1}
ITERATIONS = [
    {'type': 'message', 'input_tokens': 21, 'cache_read_input_tokens': 4,
     'cache_creation_input_tokens': 0, 'output_tokens': 11},
    {'type': 'advisor_message', 'model': 'claude-opus-5-5', 'input_tokens': 823,
     'cache_read_input_tokens': 0, 'cache_creation_input_tokens': 0, 'output_tokens': 89},
]


def advisor_events(stop_reason='tool_use', advisor_result=True, tool_use=True):
    """SSE events for an advisor turn; the result arrives whole, in one content_block_start."""
    events = [
        {'type': 'message_start', 'message': {'id': 'msg_advisor', 'role': 'assistant', 'model': 'sonnet',
                                              'content': [], 'usage': dict(USAGE)}},
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': TEXT_BEFORE}},
        {'type': 'content_block_stop', 'index': 0},
        {'type': 'content_block_start', 'index': 1, 'content_block': {
            'type': 'server_tool_use', 'id': ADVISOR_ID, 'name': 'advisor', 'input': {}}},
        {'type': 'content_block_delta', 'index': 1, 'delta': {'type': 'input_json_delta', 'partial_json': '{}'}},
        {'type': 'content_block_stop', 'index': 1},
    ]
    if advisor_result:
        events += [
            {'type': 'content_block_start', 'index': 2, 'content_block': {
                'type': 'advisor_tool_result', 'tool_use_id': ADVISOR_ID,
                'content': {'type': 'advisor_redacted_result', 'encrypted_content': ENCRYPTED}}},
            {'type': 'content_block_stop', 'index': 2},
            {'type': 'content_block_start', 'index': 3, 'content_block': {'type': 'text', 'text': ''}},
            {'type': 'content_block_delta', 'index': 3, 'delta': {'type': 'text_delta', 'text': TEXT_AFTER}},
            {'type': 'content_block_stop', 'index': 3},
        ]
    if tool_use:
        index = 4 if advisor_result else 2
        raw = json.dumps(TOOL_INPUT, separators=(',', ':'))
        events += [
            {'type': 'content_block_start', 'index': index, 'content_block': {
                'type': 'tool_use', 'id': TOOL_ID, 'name': 'mcp__hermes__write_file', 'input': {}}},
            {'type': 'content_block_delta', 'index': index,
             'delta': {'type': 'input_json_delta', 'partial_json': raw[:12]}},
            {'type': 'content_block_delta', 'index': index,
             'delta': {'type': 'input_json_delta', 'partial_json': raw[12:]}},
            {'type': 'content_block_stop', 'index': index},
        ]
    events += [
        {'type': 'message_delta', 'delta': {'stop_reason': stop_reason},
         'usage': {'output_tokens': 37, 'iterations': ITERATIONS}},
        {'type': 'message_stop'},
    ]
    return events


def sse(events):
    return b''.join(b'event: ' + e['type'].encode() + b'\ndata: ' + json.dumps(e).encode() + b'\n\n'
                    for e in events)


def expected_blocks(advisor_result=True, tool_use=True):
    blocks = [{'type': 'text', 'text': TEXT_BEFORE},
              {'type': 'server_tool_use', 'id': ADVISOR_ID, 'name': 'advisor', 'input': {}}]
    if advisor_result:
        blocks += [
            {'type': 'advisor_tool_result', 'tool_use_id': ADVISOR_ID,
             'content': {'type': 'advisor_redacted_result', 'encrypted_content': ENCRYPTED}},
            {'type': 'text', 'text': TEXT_AFTER},
        ]
    if tool_use:
        blocks.append({'type': 'tool_use', 'id': TOOL_ID, 'name': 'mcp__hermes__write_file',
                       'input': TOOL_INPUT})
    return blocks


@pytest.mark.parametrize('chunk_size', [None, 7, 1])
def test_capture_keeps_advisor_blocks_in_order(chunk_size):
    raw = sse(advisor_events())
    capture = Capture()
    if chunk_size is None:
        capture.feed(raw)
    else:
        for start in range(0, len(raw), chunk_size):
            capture.feed(raw[start:start + chunk_size])
    assert capture.complete
    assert capture.message['content'] == expected_blocks()
    assert capture.message['stop_reason'] == 'tool_use'
    assert capture.message['content'][-1]['input'] == TOOL_INPUT
    assert capture.message['usage']['iterations'] == ITERATIONS


def test_capture_completes_on_a_paused_advisor_turn():
    # Per the docs a paused turn carries the pending server_tool_use; this fixture also keeps the
    # complete advisor result so the test proves the stop reason alone never wedges the relay.
    raw = sse(advisor_events(stop_reason='pause_turn'))
    capture = Capture()
    capture.feed(raw)
    assert capture.complete
    assert capture.message['stop_reason'] == 'pause_turn'
    assert capture.message['content'] == expected_blocks()


FAKE_NATIVE = r'''
import json, os, sys, urllib.error, urllib.request

rows = []
for line in sys.stdin:
    frame = json.loads(line)
    rows.append(frame)
    if frame.get('shouldQuery') is False:
        print(json.dumps({'type': 'result', 'num_turns': 0, 'is_error': False}), flush=True)
        continue
    break

def request(messages):
    payload = json.dumps({'model': 'sonnet', 'messages': messages}).encode()
    request = urllib.request.Request(os.environ['ANTHROPIC_BASE_URL'] + '/v1/messages', data=payload,
                                     headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=5).read().decode()

def assemble(body):
    message, arguments = None, {}
    for line in body.splitlines():
        if not line.startswith('data: '):
            continue
        event = json.loads(line[6:])
        kind = event['type']
        if kind == 'message_start':
            message = event['message']
        elif kind == 'content_block_start':
            message['content'].append(event['content_block'])
        elif kind == 'content_block_delta':
            delta, block = event['delta'], message['content'][event['index']]
            if delta['type'] == 'text_delta':
                block['text'] = block.get('text', '') + delta['text']
            elif delta['type'] == 'thinking_delta':
                block['thinking'] = block.get('thinking', '') + delta['thinking']
            elif delta['type'] == 'signature_delta':
                block['signature'] = block.get('signature', '') + delta['signature']
            elif delta['type'] == 'input_json_delta':
                arguments[event['index']] = arguments.get(event['index'], '') + delta['partial_json']
        elif kind == 'content_block_stop':
            if event['index'] in arguments:
                raw = arguments.pop(event['index'])
                message['content'][event['index']]['input'] = json.loads(raw) if raw.strip() else {}
        elif kind == 'message_delta':
            message.update(event.get('delta', {}))
            message['usage'].update(event.get('usage', {}))
    return message

messages = [row['message'] for row in rows]
first = assemble(request(messages))
if os.environ.get('RESUME_PAUSE') and first.get('stop_reason') == 'pause_turn':
    # Native's documented resume: append the assistant turn unchanged and ask again.
    try:
        request(messages + [{'role': 'assistant', 'content': first['content']}])
    except urllib.error.HTTPError as error:
        error.read()
print(json.dumps({'type': 'assistant', 'message': first}), flush=True)
print(json.dumps({'type': 'stream_event', 'event': {'type': 'message_stop'}}), flush=True)
print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False,
                  'usage': {'input_tokens': 1, 'output_tokens': 1}}), flush=True)
'''


def start_peer(events):
    received = []

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            data = sse(events)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    return peer, thread, received


def stop_peer(peer, thread):
    peer.shutdown()
    thread.join()
    peer.server_close()


def make_client(tmp_path, peer, resume=False):
    script = tmp_path / 'native.py'
    script.write_text(FAKE_NATIVE, encoding='utf-8')
    env = {'PATH': os.defpath, 'HOME': str(tmp_path),
           'ANTHROPIC_BASE_URL': f'http://127.0.0.1:{peer.server_port}'}
    if resume:
        env['RESUME_PAUSE'] = '1'
    return directsdk.Client(command=[sys.executable, str(script)], env=env)


def test_advisor_turn_reaches_hermes_with_text_and_its_tool_call(tmp_path):
    peer, thread, received = start_peer(advisor_events())
    client = make_client(tmp_path, peer)
    try:
        result = client.create(model='sonnet', messages=[{'role': 'user', 'content': 'Consult and write.'}],
                               tools=TOOLS)
        choice = result.choices[0]
        assert choice.message.content == TEXT_BEFORE + TEXT_AFTER
        assert choice.finish_reason == 'tool_calls'
        assert len(choice.message.tool_calls) == 1
        call = choice.message.tool_calls[0]
        assert call.id == TOOL_ID and call.function.name == 'write_file'
        assert json.loads(call.function.arguments) == TOOL_INPUT
        assert result.usage.native_admission.upstream_requests == 1
        assert result.usage.native_admission.blocked_requests == 0
        # The server-tool blocks stay in the carried native turn, so a replay keeps them.
        assert choice.message.reasoning_details[0]['messages'][0]['content'] == expected_blocks()
        assert len(received) == 1
    finally:
        client.close()
        stop_peer(peer, thread)


def test_paused_advisor_turn_is_one_admitted_request_and_a_resume_is_denied(tmp_path):
    """A pause_turn response carries a pending server_tool_use and ends the native turn.

    directsdk has no pause_turn resume path: it treats the first complete capture as the turn, so the
    text before the advisor call reaches Hermes with finish_reason ``stop`` and no exception. When the
    fake native follows the documented resume (append the assistant turn unchanged and ask again), the
    relay has already consumed its one admission: the second request is answered 400
    HERMES_MODEL_ADMISSION_CONSUMED, counted in ``blocked_requests``, and never forwarded upstream.
    That is the current behaviour this test pins; resuming a paused advisor turn would need a
    production change in directsdk.
    """
    pause = advisor_events(stop_reason='pause_turn', advisor_result=False, tool_use=False)
    peer, thread, received = start_peer(pause)
    client = make_client(tmp_path, peer, resume=True)
    try:
        result = client.create(model='sonnet', messages=[{'role': 'user', 'content': 'Consult.'}], tools=TOOLS)
        choice = result.choices[0]
        assert choice.message.content == TEXT_BEFORE
        assert choice.message.tool_calls is None
        assert choice.finish_reason == 'stop'
        assert result.usage.native_admission.upstream_requests == 1
        assert result.usage.native_admission.blocked_requests == 1
        # The denied resume never reached the fake upstream: only the first request was served.
        assert len(received) == 1
    finally:
        client.close()
        stop_peer(peer, thread)
