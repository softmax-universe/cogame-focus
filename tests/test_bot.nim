## The scripted baseline must play whole games without ever proposing an
## illegal move — it is a fieldable policy (PLAYER_SCRIPTED=1). Model replies
## must parse strictly with no compatibility shims or silent fallbacks.

import std/[json, monotimes, os, times, unittest]
import focus/[llm, sim, prompts]

proc fixture(seed: int, maxPlies = 160): GameConfig =
  result = defaultGameConfig()
  result.seed = seed
  result.maxPlies = maxPlies
  result.sampled = true
  for index in 0 ..< 2:
    result.players.add(PlayerConfig(name: "P" & $(index + 1)))
    result.tokens.add("t" & $index)

suite "scripted baseline":
  test "plays full games legally and fast":
    for seed in [1, 7, 42, 1234]:
      let config = fixture(seed)
      let client = newLlmClient(config)
      var sim = initSim(config)
      let started = getMonoTime()
      var plies = 0
      while not sim.done:
        let seat = sim.turn
        let decision = client.scriptedAction(sim, seat)
        ## The bot's move must be legal as-is: applyMove raises on
        ## anything else and would fail this test.
        sim.recordSay(seat, decision.say)
        sim.applyMove(seat, decision.move)
        inc plies
        check plies <= config.maxPlies
      let elapsed = (getMonoTime() - started).inMilliseconds
      echo "seed ", seed, ": ", plies, " plies, ", sim.reason, ", winner ",
        sim.winner, ", ", elapsed, " ms"
      ## Budget: well under a second per ply on average.
      check elapsed < plies * 1000
      let results = sim.resultsJson()
      var total = 0.0
      for score in results["scores"]:
        total += score.getFloat()
      check abs(total - 1.0) < 1e-9
      var pieces = 0
      for cell in 0 ..< Cells:
        pieces += sim.board[cell].len
      pieces += sim.reserve[0] + sim.reserve[1] +
        sim.captured[0] + sim.captured[1]
      check pieces == 36

  test "decide raises for prompt seat without credentials, plays scripted seat":
    let keys = [
      "COWORLD_LLM_ENDPOINT",
      "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
      "AWS_BEARER_TOKEN_BEDROCK",
      "ANTHROPIC_API_KEY",
      "ANTHROPIC_API_KEY_URI"
    ]
    var saved: seq[(string, string, bool)]
    for key in keys:
      saved.add((key, getEnv(key), existsEnv(key)))
      delEnv(key)
    defer:
      for (key, value, existed) in saved:
        if existed:
          putEnv(key, value)
        else:
          delEnv(key)
    let config = fixture(3, maxPlies = 4)
    let client = newLlmClient(config)
    var sim = initSim(config)
    expect FocusError:
      discard client.decide(sim, sim.turn, "attack the centre",
        scripted = false, selection = PromptSelection())
    check client.lastFailureCause == "missing"
    let decision = client.decide(sim, sim.turn, "",
      scripted = true, selection = PromptSelection())
    check client.lastFailureCause == ""
    sim.applyMove(sim.turn, decision.move)

  test "extractJsonObject accepts bare or single-fenced JSON objects only":
    let bare = extractJsonObject("""{"say":"hi","move":{"from":"c7","count":1,"dir":"E"}}""")
    check bare["move"]["from"].getStr() == "c7"
    let fencedJson = extractJsonObject("```json\n{\"place\":\"a4\"}\n```")
    check fencedJson["place"].getStr() == "a4"
    let fencedPlain = extractJsonObject("  ```\n{\"place\":\"a4\"}\n```  ")
    check fencedPlain["place"].getStr() == "a4"
    let fencedCrlf = extractJsonObject("```JSON\r\n{\"place\":\"a4\"}\r\n```")
    check fencedCrlf["place"].getStr() == "a4"
    for bad in [
      "",
      "Sure, here is my move: {\"place\":\"a4\"}",
      "{\"place\":\"a4\"} good luck",
      "Here:\n```json\n{\"place\":\"a4\"}\n```",
      "```json\n{\"place\":\"a4\"}\n```\nThanks",
      "{\"place\":\"a4\"} {\"place\":\"b4\"}",
      "```json\n{\"place\":\"a4\"}\n```\n```json\n{\"place\":\"b4\"}\n```",
      "```python\n{\"place\":\"a4\"}\n```",
      "```json {\"place\":\"a4\"}```",
      "[1, 2, 3]"
    ]:
      expect FocusError:
        discard extractJsonObject(bad)

  test "model replies parse strictly into moves and drops":
    let config = fixture(0)
    var sim = initSim(config)
    let moved = parseDecision(sim, 0, parseJson(
      """{"say":"hi","move":{"from":"c7","count":1,"dir":"E"}}"""))
    check moved.move.kind == mkMove
    check moved.move.fromCell == parseCell("c7")
    check moved.move.toCell == parseCell("d7")
    check moved.say == "hi"
    sim.applyMove(0, moved.move)
    let noSay = parseDecision(sim, 1, parseJson(
      """{"move":{"from":"b6","count":1,"dir":"N"}}"""))
    check noSay.say == ""
    check noSay.move.dir == dirN
    let dropped = parseDecision(sim, 1, parseJson(
      """{"say":"","place":"a4"}"""))
    check dropped.move.kind == mkPlace
    check dropped.move.toCell == parseCell("a4")
    for bad in [
      """{"say":"?"}""",
      """{"move":{"from":"a1","count":1,"dir":"N"}}""",
      """{"move":{"place":"h5"}}""",
      """{"place":{"to":"a4"}}""",
      """{"move":{"from":"b6","count":"1","dir":"N"}}""",
      """{"move":{"from":"b6","count":1,"direction":"N"}}""",
      """{"move":{"from":"b6","count":1,"dir":"north"}}""",
      """{"move":{"from":"B6","count":1,"dir":"N"}}""",
      """{"move":{"from":"b6","dir":"N"}}""",
      """{"move":{"from":"b6","count":0,"dir":"N"}}""",
      """{"move":{"from":"b6","count":1,"dir":"N"},"place":"a4"}""",
      """{"move":{"from":"b6","count":1,"dir":"N","extra":1}}""",
      """{"move":{"from":"b6","count":1,"dir":"N"},"extra":1}""",
      """{"say":123,"place":"a4"}"""
    ]:
      expect FocusError:
        discard parseDecision(sim, 1, parseJson(bad))
