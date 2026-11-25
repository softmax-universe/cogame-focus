"""Exercise complete native server/player games and preserve all test artifacts."""

import base64
import hashlib
import json
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def make_provider(fail_ply=None, fail_mode="model_rejection"):
    class Provider(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            raw = self.rfile.read(int(self.headers["Content-Length"]))
            body = json.loads(raw)
            user = body["messages"][0]["content"]
            ply = int(re.search(r"Ply (\d+) of", user)[1])
            slot = self.headers["X-Coworld-Player-Slot"]
            assert "PRIVATE_OPERATOR_" + slot in user
            assert "PRIVATE_OPERATOR_" + str(1 - int(slot)) not in user
            retry = "Your previous reply was invalid" in user
            legal = user.split("YOUR LEGAL OPTIONS:\n", 1)[1].split("\n\n", 1)[0]
            move = re.search(r"([a-h][1-8]) \(h\d+\): (\d+)([NSEW])", legal)
            action = {"say": "hello"}
            if move:
                action["move"] = {"from": move[1], "count": int(move[2]), "dir": move[3]}
            else:
                assert "DROP:" in legal
                action["place"] = "c1"
            if fail_ply is not None and ply == fail_ply:
                if fail_mode == "provider":
                    err = json.dumps({"error": "upstream unavailable"}).encode()
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                    return
                if fail_mode == "timeout":
                    time.sleep(2.5)
                    return
                if not retry:
                    text = "I think I will play " + json.dumps(action)
                else:
                    bad = dict(action)
                    if "move" in bad:
                        bad_move = dict(bad["move"])
                        bad_move["direction"] = bad_move.pop("dir")
                        bad["move"] = bad_move
                    else:
                        bad["place"] = {"to": bad["place"]}
                    text = json.dumps(bad)
            elif ply == 1 and not retry:
                text = "invalid fixture reply"
            else:
                if ply == 3:
                    action["say"] = "s" * 300
                text = json.dumps(action)
                if ply == 2:
                    text = f"```json\n{text}\n```"
                elif ply == 4:
                    text = f"```\n{text}\n```"
            response = json.dumps(
                {
                    "model": body["model"],
                    "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": text}],
                    "usage": None if ply == 4 else {"input_tokens": 100, "output_tokens": 20},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.send_header("X-Softmax-Llm-Call-Id", str(uuid.uuid5(uuid.NAMESPACE_URL, raw.decode())))
            self.end_headers()
            self.wfile.write(response)

    return Provider


class SimpleWebSocket:
    """Minimal RFC 6455 text WebSocket client using only the Python stdlib."""

    def __init__(self, host, port, path):
        self.sock = socket.create_connection((host, port), timeout=10)
        key = base64.b64encode(os.urandom(16)).decode()
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self.sock.sendall(req.encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise EOFError("WebSocket handshake closed")
            buf += chunk
        header, rest = buf.split(b"\r\n\r\n", 1)
        assert header.startswith(b"HTTP/1.1 101"), header.decode(errors="replace")
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()
        ).decode()
        assert expected.encode() in header
        self.buf = rest

    def _read_exact(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(4096)
            if not chunk:
                return None
            self.buf += chunk
        out = self.buf[:n]
        self.buf = self.buf[n:]
        return out

    def send_json(self, obj):
        payload = json.dumps(obj).encode()
        mask = os.urandom(4)
        header = bytearray([0x81])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", length))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", length))
        header.extend(mask)
        masked = bytes(b ^ mask[i & 3] for i, b in enumerate(payload))
        self.sock.sendall(bytes(header) + masked)

    def recv_json(self):
        while True:
            head = self._read_exact(2)
            if head is None:
                return None
            opcode = head[0] & 0x0F
            length = head[1] & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(8))[0]
            data = self._read_exact(length)
            if data is None:
                return None
            if opcode == 0x8:
                return None
            if opcode == 0x1:
                return json.loads(data.decode())

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


def run(
    game,
    player,
    output,
    variant,
    capture,
    seed,
    model_enabled=True,
    fail_ply=None,
    fail_mode="model_rejection",
    scripted=False,
):
    tag = f"{variant['id']}-seed{seed}-capture{capture}"
    if not model_enabled:
        tag += "-scripted" if scripted else "-offline"
    if fail_ply is not None:
        tag += f"-{fail_mode}-fail{fail_ply}"
    root = output / tag
    root.mkdir(mode=0o700)
    config = dict(variant["game_config"])
    seats = config["num_agents"]
    config.update(
        seed=seed,
        turnDelayMs=0,
        llmTimeoutSeconds=1 if fail_mode == "timeout" else 2,
        player_connect_timeout_seconds=5,
        episodeTimeoutSeconds=120,
        shutdownGraceSeconds=1,
        tokens=[f"fixture-{i}" for i in range(seats)],
    )
    (root / "config.json").write_text(json.dumps(config))
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    env = dict(os.environ)
    for key in (
        "COWORLD_LLM_ENDPOINT",
        "COWORLD_LLM_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY_URI",
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
        "AWS_BEARER_TOKEN_BEDROCK",
    ):
        env.pop(key, None)
    episode = f"focus-native-capture-fixture-{tag}"
    env.update(
        COGAME_HOST="127.0.0.1",
        COGAME_PORT=str(port),
        COGAME_CONFIG_URI=(root / "config.json").as_uri(),
        COGAME_RESULTS_URI=(root / "results.json").as_uri(),
        COGAME_SAVE_REPLAY_URI=(root / "replay.json").as_uri(),
        COGAME_SAVE_TRAJECTORY_URI=(root / "trajectory.jsonl").as_uri(),
        COWORLD_EPISODE_ID=episode,
        COWORLD_GAME_VERSION="identity-fixture",
        COWORLD_SOURCE_REVISION="a" * 40,
    )
    if not capture:
        env.pop("COGAME_SAVE_TRAJECTORY_URI")
    provider = ThreadingHTTPServer(("127.0.0.1", 0), make_provider(fail_ply=fail_ply, fail_mode=fail_mode))
    thread = threading.Thread(target=provider.serve_forever, daemon=True)
    thread.start()
    if model_enabled:
        env["COWORLD_LLM_ENDPOINT"] = f"http://127.0.0.1:{provider.server_port}"
        env["COWORLD_LLM_MODEL"] = "native-capture-fixture"
    children = []
    streams = []
    expect_failure = fail_ply is not None or (not model_enabled and not scripted)
    try:
        log = (root / "game.log").open("wb")
        streams.append(log)
        server = subprocess.Popen([str(game)], env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(server)
        for _ in range(100):
            assert server.poll() is None, "game exited before readiness; see retained game.log"
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                    assert response.status == 200
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.05)
        else:
            raise TimeoutError("native game did not become ready")
        for seat in range(seats):
            log = (root / f"player-{seat}.log").open("wb")
            streams.append(log)
            player_env = dict(env, PLAYER_PROMPT=f"PRIVATE_OPERATOR_{seat}", COWORLD_TIMEOUT_SECONDS="120")
            if scripted:
                player_env["PLAYER_SCRIPTED"] = "1"
            player_env["COWORLD_PLAYER_WS_URL"] = f"ws://127.0.0.1:{port}/player?slot={seat}&token=fixture-{seat}"
            children.append(subprocess.Popen([str(player)], env=player_env, stdout=log, stderr=subprocess.STDOUT))
        rc = server.wait(timeout=120)
        for child in children[1:]:
            assert child.wait(timeout=5) == 0
        replay = json.loads((root / "replay.json").read_text())
        if expect_failure:
            assert rc != 0, f"expected nonzero exit on failure ({tag})"
            assert not (root / "results.json").exists(), "failed episode must not publish official results"
            expected_cause = fail_mode if fail_ply is not None else "missing"
            expected_ply = (fail_ply - 1) if fail_ply is not None else 0
            assert replay["results"] is None
            assert replay["failure"]["status"] == "failed"
            assert replay["failure"]["cause"] == expected_cause
            assert replay["failure"]["ply"] == expected_ply
            assert not any(ev["kind"] == "end" for ev in replay["events"])
            if capture:
                trace = root / "trajectory.jsonl"
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                assert trace.stat().st_mode & 0o077 == 0
                assert rows[-1]["status"] == "failed"
                assert rows[-1]["outcome"]["cause"] == expected_cause
                assert rows[-1]["outcome"]["plies"] == expected_ply
                assert rows[-1]["participant_outcomes"] == {}
                assert len(rows) == expected_ply + 2
                last = rows[-2]
                expected_action_status = "rejected" if expected_cause == "model_rejection" else "missing"
                assert last["action_status"] == expected_action_status
                assert last["executed_action"] is None
                assert last["selected_attempt_id"] is None
                assert last["terminal"] is True
                if expected_cause == "missing":
                    assert last["attempts"] == []
                else:
                    assert len(last["attempts"]) == 2
                    assert all(not a["accepted"] for a in last["attempts"])
                    if expected_cause == "provider":
                        assert all(a["http_status"] == 503 for a in last["attempts"])
            return None, replay
        assert rc == 0
        results = json.loads((root / "results.json").read_text())
        assert results["reason"] in ("cap", "no-moves")
        assert 1 <= results["plies"] <= config["maxPlies"]
        assert abs(sum(results["scores"]) - 1.0) < 1e-9
        if capture:
            trace = root / "trajectory.jsonl"
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            assert trace.stat().st_mode & 0o077 == 0
            assert rows[-1]["status"] == "completed"
            assert rows[-1]["outcome"] == results
            if scripted:
                assert len(rows) == results["plies"] + 1
                for row in rows[:-1]:
                    assert row["action_status"] == "accepted"
                    assert len(row["attempts"]) == 1
                    assert row["attempts"][0]["origin"] == "teacher"
            else:
                assert len(rows) == results["plies"] + 1
                for row in rows[:-1]:
                    attempts = row["attempts"]
                    ply = row["observation"]["view"]["ply"] + 1
                    if ply == 1:
                        assert len(attempts) == 2 and not attempts[0]["accepted"]
                        assert attempts[1]["accepted"]
                        assert row["selected_attempt_id"] == attempts[1]["attempt_id"]
                    elif ply == 3:
                        assert row["action_status"] == "rejected"
                        assert row["selected_attempt_id"] is None
                        assert len(row["executed_action"]["say"]) <= 160
                    else:
                        assert row["action_status"] == "accepted"
                    if ply == 4:
                        assert attempts[0]["input_tokens"] is None
                        assert attempts[0]["output_tokens"] is None
                    for attempt in attempts:
                        assert attempt["prompt"][0]["content"] == attempt["request"]["system"]
                        assert attempt["platform_call_id"] is not None
                        assert attempt["behavior_logprobs"] is None
        return results, replay
    finally:
        provider.shutdown()
        provider.server_close()
        thread.join(timeout=5)
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        for stream in streams:
            stream.close()


def run_external_failure_test(game, player, output, variant, mode):
    """Verify external policy legal move plus illegal-action, disconnect, and timeout failures."""
    root = output / f"external-{mode}"
    root.mkdir(mode=0o700)
    config = dict(variant["game_config"])
    config.update(
        seed=42,
        turnDelayMs=0,
        llmTimeoutSeconds=1 if mode == "timeout" else 3,
        player_connect_timeout_seconds=5,
        episodeTimeoutSeconds=120,
        tokens=["fixture-0", "fixture-1"],
    )
    (root / "config.json").write_text(json.dumps(config))
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    env = dict(os.environ)
    for key in (
        "COWORLD_LLM_ENDPOINT",
        "COWORLD_LLM_MODEL",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_API_KEY_URI",
        "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
        "AWS_BEARER_TOKEN_BEDROCK",
    ):
        env.pop(key, None)
    env.update(
        COGAME_HOST="127.0.0.1",
        COGAME_PORT=str(port),
        COGAME_CONFIG_URI=(root / "config.json").as_uri(),
        COGAME_RESULTS_URI=(root / "results.json").as_uri(),
        COGAME_SAVE_REPLAY_URI=(root / "replay.json").as_uri(),
        COGAME_SAVE_TRAJECTORY_URI=(root / "trajectory.jsonl").as_uri(),
        COWORLD_EPISODE_ID=f"focus-external-{mode}",
        COWORLD_GAME_VERSION="identity-fixture",
        COWORLD_SOURCE_REVISION="a" * 40,
    )
    children = []
    streams = []
    ws = None
    try:
        log = (root / "game.log").open("wb")
        streams.append(log)
        server = subprocess.Popen([str(game)], env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append(server)
        for _ in range(100):
            assert server.poll() is None
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                    assert response.status == 200
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.05)
        else:
            raise TimeoutError("game did not become ready")
        # Seat 1 is genuine scripted tactician; seat 0 is our external policy (seed 42 -> seat 0 moves first).
        p1_log = (root / "player-1.log").open("wb")
        streams.append(p1_log)
        p1_env = dict(
            env,
            PLAYER_SCRIPTED="1",
            COWORLD_TIMEOUT_SECONDS="120",
            COWORLD_PLAYER_WS_URL=f"ws://127.0.0.1:{port}/player?slot=1&token=fixture-1",
        )
        children.append(subprocess.Popen([str(player)], env=p1_env, stdout=p1_log, stderr=subprocess.STDOUT))

        ws = SimpleWebSocket("127.0.0.1", port, "/player?slot=0&token=fixture-0")
        welcome = ws.recv_json()
        assert welcome["type"] == "welcome" and welcome["slot"] == 0
        ws.send_json({"type": "register", "control": "external"})
        turns = 0
        while True:
            msg = ws.recv_json()
            if msg is None or msg.get("type") in ("final", "error"):
                break
            if msg.get("type") == "observation":
                turns += 1
                if turns == 1:
                    # First turn: send a genuine legal move.
                    ws.send_json({"type": "action", "id": msg["id"], "move": msg["observation"]["legalMoves"][0]})
                elif mode == "illegal":
                    ws.send_json({"type": "action", "id": msg["id"], "move": "a1 x1 N -> a2"})
                elif mode == "disconnect":
                    ws.close()
                    ws = None
                    break
                elif mode == "timeout":
                    # Deliberately do not reply; wait for server error/close.
                    pass
        assert server.wait(timeout=15) != 0
        assert children[1].wait(timeout=5) == 0
        assert not (root / "results.json").exists(), "failed external episode must not publish official results"
        replay = json.loads((root / "replay.json").read_text())
        expected_cause = {"illegal": "model_rejection", "disconnect": "disconnected", "timeout": "timeout"}[mode]
        assert replay["results"] is None
        assert replay["failure"]["status"] == "failed"
        assert replay["failure"]["cause"] == expected_cause
        assert replay["failure"]["ply"] == 2
        assert not any(ev["kind"] == "end" for ev in replay["events"])
        rows = [json.loads(line) for line in (root / "trajectory.jsonl").read_text().splitlines()]
        assert rows[-1]["status"] == "failed"
        assert rows[-1]["outcome"]["cause"] == expected_cause
        assert rows[-1]["participant_outcomes"] == {}
        expected_status = "rejected" if mode == "illegal" else "missing"
        assert rows[-2]["action_status"] == expected_status
        assert rows[-2]["executed_action"] is None
        assert rows[-2]["terminal"] is True
    finally:
        if ws is not None:
            ws.close()
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
        for stream in streams:
            stream.close()


def main():
    game, player, destination = map(lambda value: Path(value).resolve(), sys.argv[1:])
    destination.mkdir(mode=0o700, exist_ok=False)
    manifest = json.loads((Path(__file__).resolve().parents[1] / "coworld_manifest_template.json").read_text())
    variant = manifest["variants"][0]
    for seed in (42, 43, 44):
        captured = run(game, player, destination, variant, True, seed)
        ordinary = run(game, player, destination, variant, False, seed)
        assert captured == ordinary, "capture changed the native outcome or replay"
    captured_fail = run(game, player, destination, variant, True, 42, fail_ply=2, fail_mode="model_rejection")
    ordinary_fail = run(game, player, destination, variant, False, 42, fail_ply=2, fail_mode="model_rejection")
    assert captured_fail == ordinary_fail, "capture changed model-rejection failure replay"
    captured_off = run(game, player, destination, variant, True, 42, model_enabled=False)
    ordinary_off = run(game, player, destination, variant, False, 42, model_enabled=False)
    assert captured_off == ordinary_off, "capture changed missing-credentials failure replay"
    run(game, player, destination, variant, True, 42, fail_ply=2, fail_mode="provider")
    run(game, player, destination, variant, True, 42, fail_ply=2, fail_mode="timeout")
    run(game, player, destination, variant, True, 42, model_enabled=False, scripted=True)
    run_external_failure_test(game, player, destination, variant, "illegal")
    run_external_failure_test(game, player, destination, variant, "disconnect")
    run_external_failure_test(game, player, destination, variant, "timeout")
    families = {}
    for seed in (42, 43, 44):
        trace = destination / f"standard-seed{seed}-captureTrue" / "trajectory.jsonl"
        families[seed] = json.loads(trace.read_text().splitlines()[-1])["seed_family"]
    assert families[42] == families[44] and families[42] != families[43]
    print(
        "native capture, fenced JSON, retries, explicit failed episodes "
        "(model_rejection, missing, provider, timeout, external illegal/disconnect/timeout), "
        "genuine scripted play, and family grouping PASS"
    )


if __name__ == "__main__":
    main()
