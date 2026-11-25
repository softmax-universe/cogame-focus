## The scripted baseline must play whole games without ever proposing an
## illegal move — it is a fieldable policy (PLAYER_SCRIPTED=1). Model replies
## must parse strictly with no compatibility shims or silent fallbacks.

import std/[json, monotimes, os, strutils, times, unittest]
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

  test "bedrock transport pins a single configured model ID":
    let keys = [
      "COWORLD_LLM_ENDPOINT",
      "AWS_ENDPOINT_URL_BEDROCK_RUNTIME",
      "AWS_BEARER_TOKEN_BEDROCK",
      "BEDROCK_MODEL",
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
    putEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME", "http://127.0.0.1:9999")
    putEnv("AWS_BEARER_TOKEN_BEDROCK", "test-token")
    putEnv("BEDROCK_MODEL", "pinned-bedrock-test-model")
    let pinnedClient = newLlmClient(fixture(3, maxPlies = 4))
    check pinnedClient.model == "pinned-bedrock-test-model"
    check pinnedClient.bedrockUrl() ==
      "http://127.0.0.1:9999/model/pinned-bedrock-test-model/invoke"
    delEnv("BEDROCK_MODEL")
    let defaultClient = newLlmClient(fixture(3, maxPlies = 4))
    check defaultClient.model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    check defaultClient.bedrockUrl() ==
      "http://127.0.0.1:9999/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/invoke"

  test "the final move line or a standalone fence identifies the selected move":
    let bare = extractJsonObject("""{"say":"hi","move":"c7 x1 E -> d7"}""")
    check bare["move"].getStr() == "c7 x1 E -> d7"
    let finalLine = extractJsonObject("Board analysis first.\n\n{\"move\":\"e6 x1 N -> e7\"}")
    check finalLine["move"].getStr() == "e6 x1 N -> e7"
    let fencedJson = extractJsonObject("```json\n{\"move\":\"drop @ a4\"}\n```")
    check fencedJson["move"].getStr() == "drop @ a4"
    let fencedPlain = extractJsonObject("  ```\n{\"move\":\"drop @ a4\"}\n```  ")
    check fencedPlain["move"].getStr() == "drop @ a4"
    let fencedCrlf = extractJsonObject("```JSON\r\n{\"move\":\"drop @ a4\"}\r\n```")
    check fencedCrlf["move"].getStr() == "drop @ a4"
    for bad in [
      "",
      "Sure, here is my move: {\"move\":\"drop @ a4\"}",
      "{\"move\":\"drop @ a4\"} good luck",
      "Here:\n```json\n{\"move\":\"drop @ a4\"}\n```",
      "```json\n{\"move\":\"drop @ a4\"}\n```\nThanks",
      "{\"move\":\"drop @ a4\"} {\"move\":\"drop @ b4\"}",
      "```json\n{\"move\":\"drop @ a4\"}\n```\n```json\n{\"move\":\"drop @ b4\"}\n```",
      "```python\n{\"move\":\"drop @ a4\"}\n```",
      "```json {\"move\":\"drop @ a4\"}```",
      "[1, 2, 3]"
    ]:
      expect FocusError:
        discard extractJsonObject(bad)

  test "model replies parse strictly into canonical moveText IDs across move and drop phases":
    let config = fixture(0)
    var sim = initSim(config)
    let moved = parseDecision(sim, 0, parseJson(
      """{"say":"hi","move":"c7 x1 E -> d7"}"""))
    check moved.move.kind == mkMove
    check moved.move.fromCell == parseCell("c7")
    check moved.move.toCell == parseCell("d7")
    check moved.say == "hi"
    check decisionJson(moved) == %*{"move": "c7 x1 E -> d7", "say": "hi"}
    sim.applyMove(0, moved.move)
    let noSay = parseDecision(sim, 1, parseJson(
      """{"move":"b6 x1 N -> b7"}"""))
    check noSay.say == ""
    check noSay.move.dir == dirN
    check decisionJson(noSay) == %*{"move": "b6 x1 N -> b7"}
    sim.applyMove(1, noSay.move)
    ## Multi-piece stack carry (d7 has height 2 controlled by seat 0).
    let stacked = parseDecision(sim, 0, parseJson(
      """{"move":"d7 x2 E -> f7"}"""))
    check stacked.move.kind == mkMove
    check stacked.move.count == 2
    check stacked.move.toCell == parseCell("f7")
    sim.applyMove(0, stacked.move)
    ## Drop phase: reserve drop only legal when seat has reserve > 0.
    expect FocusError:
      discard parseDecision(sim, 1, parseJson("""{"move":"drop @ a4"}"""))
    sim.reserve[1] = 1
    let dropped = parseDecision(sim, 1, parseJson(
      """{"say":"","move":"drop @ a4"}"""))
    check dropped.move.kind == mkPlace
    check dropped.move.toCell == parseCell("a4")
    check decisionJson(dropped) == %*{"move": "drop @ a4"}
    for bad in [
      """{"say":"?"}""",
      """{"move":"a1 x1 N -> a2"}""",
      """{"move":"f7 x1 W -> e7"}""",
      """{"move":"b6 x1 N → b7"}""",
      """{"move":"b6 x1 N -> b7 "}""",
      """{"move":"B6 x1 N -> b7"}""",
      """{"move":"DROP @ a4"}""",
      """{"move":{"from":"b6","count":1,"dir":"N"}}""",
      """{"place":"a4"}""",
      """{"move":"drop @ a4","place":"a4"}""",
      """{"move":["drop @ a4","drop @ b4"]}""",
      """{"moves":["drop @ a4"]}""",
      """{"move":"drop @ a4","extra":1}""",
      """{"say":123,"move":"drop @ a4"}"""
    ]:
      expect FocusError:
        discard parseDecision(sim, 1, parseJson(bad))

  test "prompt lists exact legal moveText IDs and builds example from current legal move":
    let config = fixture(0)
    var sim = initSim(config)
    let legalMove = moveText(sim.legalMoves(0)[0])
    let sys = systemPrompt(sim, 0, PromptSelection())
    let usr = userPrompt(sim, 0, "Play strong Focus.", "Ply 1 of at most 160.")
    check "Bluff, needle, and mislead" notin sys
    check $(%*{"move": legalMove}) in usr
    check legalMove in usr
    ## Drop-only position: example must be a legal drop ID.
    for cell in 0 ..< Cells:
      sim.board[cell] = @[]
    sim.board[parseCell("d4")] = @[1]
    sim.reserve[0] = 1
    let dropLegal = moveText(sim.legalMoves(0)[0])
    check dropLegal.startsWith("drop @ ")
    let dropUsr = userPrompt(sim, 0, "", "Ply 1 of at most 160.")
    check $(%*{"move": dropLegal}) in dropUsr
    check dropLegal in dropUsr
