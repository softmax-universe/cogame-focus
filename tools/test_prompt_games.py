"""Complete published Focus games with matched wordings and actual stack moves."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = Path(sys.argv[1]).resolve()
WORDINGS = (
    "canonical",
    "plain",
    "numbered",
    "questions",
    "reordered",
    "heldout-briefing",
    "heldout-reference",
)


def play(seed, wording="canonical", prompt_seed=0, speech=False):
    command = [
        str(BRIDGE),
        str(ROOT / "coworld_manifest_template.json"),
        "standard",
    ]
    command.append("text")
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    ) as process:
        assert process.stdin is not None and process.stdout is not None

        def request(value):
            process.stdin.write(json.dumps(value) + "\n")
            process.stdin.flush()
            return json.loads(process.stdout.readline())

        observation = request(
            {
                "kind": "reset",
                "players": 2,
                "seed": str(seed),
                "promptVariant": wording,
                "promptSeed": prompt_seed,
                "split": "test",
            }
        )
        trace, systems = [], {}
        drops = 0
        while observation["kind"] == "decision":
            index, seat = observation["decision_id"], observation["seat"]
            assert index == len(trace)
            view, messages = observation["semantic_view"], observation["messages"]
            systems.setdefault(seat, messages[0]["content"])
            assert systems[seat] == messages[0]["content"]
            metadata = view.pop("prompt_template")
            assert metadata["bank"] == "focus-rules-v2"
            assert metadata["variant"] == (
                WORDINGS[prompt_seed] if wording == "mixed" else wording
            )
            for message in messages:
                assert (
                    metadata[message["role"] + "_sha256"].lower()
                    == hashlib.sha256(message["content"].encode()).hexdigest()
                )
            teacher = json.loads(request({"kind": "teacher"})["response"])
            assert isinstance(teacher["move"], str)
            if speech:
                teacher["say"] = "  PUBLIC-TALK-" + "x" * 200 + "  "
            else:
                teacher.pop("say", None)
            if teacher["move"].startswith("drop @ "):
                drops += 1
            raw_response = json.dumps(teacher)
            if speech and index % 3 == 1:
                raw_response = f"```json\n{raw_response}\n```"
            elif speech and index % 3 == 2:
                raw_response = f"```\n{raw_response}\n```"
            result = request(
                {"kind": "step", "decision_id": index, "response": raw_response}
            )
            assert result["kind"] == "accepted"
            action = result["action"]
            assert isinstance(action["move"], str)
            if speech:
                assert (
                    action["say"].startswith("PUBLIC-TALK-")
                    and len(action["say"].encode()) <= 160
                )
            else:
                assert "say" not in action
            trace.append((view, messages[1]["content"]))
            observation = result["observation"]
        assert 1 <= len(trace) <= 120 and sum(observation["scores"].values()) == 1
        process.stdin.close()
        assert process.wait(timeout=5) == 0
        return trace, observation, systems, drops


for seed in (17, 946001):
    baseline = play(seed)
    distinct = []
    for wording, prompt_seed in [(wording, 0) for wording in WORDINGS] + [("mixed", 4)]:
        result = play(seed, wording, prompt_seed)
        assert result[:2] == baseline[:2]
        if wording == "canonical":
            assert result[2] == baseline[2]
        if wording == "mixed":
            assert result[2] == distinct[prompt_seed]
        else:
            distinct.append(result[2])
    assert all(len({systems[seat] for systems in distinct}) == 7 for seat in range(2))
    print(seed, "9 complete matched wording games PASS", flush=True)
spoken = play(17, speech=True)
assert spoken[3] > 0
print(
    "19 complete text games; speech, fences, and strict move/drop grammar PASS",
    flush=True,
)
for bad_response in (
    'Sure: {"move":"c7 x1 E -> d7"}',
    '{"move":{"from":"c7","count":1,"dir":"E"}}',
    '{"place":"a4"}',
    '{"move":"c7 x1 E -> d7","place":"a4"}',
    '{"move":"c7 x1 E -> d7"} {"move":"b7 x1 E -> c7"}',
    '{"move":"c7 x1 E → d7"}',
    '{"move":["c7 x1 E -> d7","b7 x1 E -> c7"]}',
    '{"move":"a1 x1 N -> a2"}',
):
    reset_line = json.dumps(
        {
            "kind": "reset",
            "players": 2,
            "seed": "17",
            "promptVariant": "canonical",
            "promptSeed": 0,
            "split": "test",
        }
    )
    step_line = json.dumps(
        {"kind": "step", "decision_id": 0, "response": bad_response}
    )
    rejected = subprocess.run(
        [str(BRIDGE), str(ROOT / "coworld_manifest_template.json"), "standard", "text"],
        input=reset_line + "\n" + step_line + "\n",
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert rejected.returncode != 0, f"expected bridge to reject {bad_response!r}"
print("Bridge strict rejection checks PASS", flush=True)
for wording in WORDINGS[-2:]:
    result = subprocess.run(
        [str(BRIDGE), str(ROOT / "coworld_manifest_template.json"), "standard", "text"],
        input=json.dumps(
            {
                "kind": "reset",
                "players": 2,
                "seed": "17",
                "promptVariant": wording,
                "split": "train",
            }
        )
        + "\n",
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert (
        result.returncode != 0
        and "held-out wording cannot be used for training" in result.stderr
    )
print("Held-out training gate PASS", flush=True)
