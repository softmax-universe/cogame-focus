## Claude-backed decision making for Focus. Each seat's policy is just a
## prompt: the game server composes the board plus that seat's prompt and
## asks Claude what the cog says and which legal move it makes.
##
## Credentials, in order of preference:
##   COWORLD_LLM_ENDPOINT            - hosted sidecar
##   Bedrock bearer token            - local play
##   ANTHROPIC_API_KEY               - the key itself
##   ANTHROPIC_API_KEY_URI           - a URI holding the key
## Prompt seats require working model credentials and a valid legal JSON reply;
## rejected, missing, or provider-failed decisions raise FocusError and fail
## the episode without awarding scores. The built-in scripted bot is a separate
## fieldable policy: a player that registers as scripted plays it deliberately,
## LLM or not.

import
  std/[json, options, os, random, strutils, times],
  bitworld/runtime,
  curly,
  sim, prompts, training

const
  AnthropicUrl = "https://api.anthropic.com/v1/messages"
  AnthropicVersion = "2023-06-01"
  BedrockAnthropicVersion = "bedrock-2023-05-31"
  DefaultBedrockModel = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
  ## What the viewer's speech bubble can actually show (~4 wrapped lines).
  MaxSayLen = 160

type
  Decision* = object
    say*: string
    move*: Move

  LlmTransport = enum
    ltNone, ltSidecar, ltBedrock, ltAnthropic

  LlmClient* = ref object
    nativeAttempts*: seq[DecisionAttempt]
    lastFailureCause*: string
    curl: Curly
    transport: LlmTransport
    apiKey: string          ## anthropic transport
    sidecarEndpoint: string
    bedrockEndpoint: string ## local Bedrock transport
    bedrockToken: string
    model*: string
    maxOutputTokens: int
    timeoutSeconds: int
    disabled: bool    ## true once credentials are known-unavailable
    rand: Rand

proc resolveApiKey(): string =
  result = getEnv("ANTHROPIC_API_KEY").strip()
  if result.len > 0:
    return
  let uri = getEnv("ANTHROPIC_API_KEY_URI").strip()
  if uri.len == 0:
    return ""
  try:
    result = readCogameUri(uri, "ANTHROPIC_API_KEY_URI").strip()
  except CatchableError as error:
    echo "focus llm: failed to fetch ANTHROPIC_API_KEY_URI: ", error.msg
    result = ""

proc bedrockModelId*(): string =
  ## Single pinned Bedrock inference-profile id: BEDROCK_MODEL if configured,
  ## otherwise the default Haiku profile. Never switches to alternate models.
  let pinned = getEnv("BEDROCK_MODEL").strip()
  if pinned.len > 0: pinned
  else: DefaultBedrockModel

proc bedrockUrl*(client: LlmClient): string =
  client.bedrockEndpoint & "/model/" & client.model & "/invoke"

proc newLlmClient*(config: GameConfig): LlmClient =
  result = LlmClient(
    model: config.model,
    maxOutputTokens: config.maxOutputTokens,
    timeoutSeconds: config.llmTimeoutSeconds,
    rand: initRand(config.seed xor 0x5EED)
  )
  let sidecarEndpoint = getEnv("COWORLD_LLM_ENDPOINT").strip()
  if sidecarEndpoint.len > 0:
    result.transport = ltSidecar
    result.sidecarEndpoint = sidecarEndpoint.strip(chars = {'/'}, leading = false)
    let sidecarModel = getEnv("COWORLD_LLM_MODEL").strip()
    result.model =
      if sidecarModel.len > 0: sidecarModel
      else: "anthropic/claude-haiku-4.5"
    result.curl = newCurly()
    echo "focus llm: sidecar transport, model ", result.model
    return
  let bedrockEndpoint = getEnv("AWS_ENDPOINT_URL_BEDROCK_RUNTIME").strip()
  let bedrockToken = getEnv("AWS_BEARER_TOKEN_BEDROCK").strip()
  if bedrockEndpoint.len > 0 or bedrockToken.len > 0:
    let region = getEnv("AWS_REGION",
      getEnv("AWS_DEFAULT_REGION", "us-west-2"))
    let endpoint =
      if bedrockEndpoint.len > 0: bedrockEndpoint
      else: "https://bedrock-runtime." & region & ".amazonaws.com"
    result.transport = ltBedrock
    result.bedrockEndpoint = endpoint.strip(chars = {'/'}, leading = false)
    result.model = bedrockModelId()
    result.bedrockToken = bedrockToken
    result.curl = newCurly()
    echo "focus llm: bedrock transport, url ", result.bedrockUrl
    return
  result.apiKey = resolveApiKey()
  if result.apiKey.len > 0:
    result.transport = ltAnthropic
    result.curl = newCurly()
    echo "focus llm: anthropic transport, model ", result.model
  else:
    result.transport = ltNone
    result.disabled = true
    echo "focus llm: no LLM credentials; prompt seats will fail"

proc newScriptedClient*(config: GameConfig): LlmClient =
  ## Offline training needs the same seeded minimax policy without transport.
  LlmClient(rand: initRand(config.seed xor 0x5EED))

# ---- Scripted baseline ------------------------------------------------------

const
  CaptureLines = [
    "Off the bottom you go.",
    "One less to worry about.",
    "Mind the gap — that piece is gone.",
    ""
  ]
  StackLines = [
    "Consolidating.",
    "Building the tower.",
    "Higher ground.",
    ""
  ]
  TakeLines = [
    "That stack is mine now.",
    "Thanks for holding it for me.",
    "Under new management.",
    ""
  ]
  DropLines = [
    "Reinforcements from the bench.",
    "Fresh piece, fresh problem for you.",
    ""
  ]

proc pick(client: LlmClient, lines: openArray[string]): string =
  lines[client.rand.rand(lines.high)]

proc evaluate(sim: Sim, seat: int): int =
  ## Static score from `seat`'s side, in centi-pieces. Material (pieces in
  ## controlled stacks + reserve) dominates; captures are permanent and
  ## count extra; controlled stacks are mobility.
  let other = 1 - seat
  if sim.done:
    if sim.winner == seat: return 100_000
    if sim.winner == other: return -100_000
    return 0
  result = 100 * (sim.material(seat) - sim.material(other)) +
    200 * (sim.captured[seat] - sim.captured[other]) +
    50 * (sim.controlledStacks(seat) - sim.controlledStacks(other))

proc scriptedAction*(client: LlmClient, sim: Sim, seat: int): Decision =
  ## Rule-based baseline: depth-2 minimax (my move, their best reply)
  ## on material / captures / mobility, seeded random tie-breaks. Always
  ## returns a legal move.
  let moves = sim.legalMoves(seat)
  if moves.len == 0:
    raise newException(FocusError, "no legal move for the scripted bot")
  var bestScore = int.low
  var bestMoves: seq[Move]
  ## Search on a copy without the event log: every probe copies the Sim,
  ## and the log is the only part that grows over a game.
  var root = sim
  root.events = @[]
  for move in moves:
    var next = root
    next.applyMove(seat, move)
    ## Opponent's best reply (one ply), then static evaluation.
    var score: int
    if next.done:
      score = next.evaluate(seat)
    else:
      score = int.high
      let opponent = next.turn
      for reply in next.legalMoves(opponent):
        var after = next
        after.applyMove(opponent, reply)
        let s = after.evaluate(seat)
        if s < score:
          score = s
          if score < bestScore:
            break   # this move cannot beat the incumbent
    if score > bestScore:
      bestScore = score
      bestMoves = @[move]
    elif score == bestScore:
      bestMoves.add(move)
  let chosen = bestMoves[client.rand.rand(bestMoves.high)]
  ## A line to match the beat.
  var probe = root
  probe.applyMove(seat, chosen)
  var event = probe.events[0]
  for candidate in probe.events:
    if candidate.kind in {evMove, evPlace}:
      event = candidate
  result.move = chosen
  if chosen.kind == mkPlace:
    result.say = client.pick(DropLines)
  elif event.captured > 0:
    result.say = client.pick(CaptureLines)
  elif event.stackAfter.len > chosen.count and
      event.stackAfter[event.stackAfter.len - chosen.count - 1] != seat:
    result.say = client.pick(TakeLines)
  else:
    result.say = client.pick(StackLines)

# ---- Prompt building --------------------------------------------------------

proc seatName(sim: Sim, seat: int): string =
  sim.names[seat]

proc pieceLetter(sim: Sim, owner: int): string =
  ## Pieces are named by their seat alias's initial.
  $sim.names[owner][0]

proc stackText(sim: Sim, stack: seq[int]): string =
  for owner in stack:
    result.add(sim.pieceLetter(owner))

proc renderBoard(sim: Sim, viewer: int): string =
  ## Every occupied square with its stack bottom->top, grouped by
  ## controller. Letters are the aliases' initials.
  var mine: seq[string]
  var theirs: seq[string]
  for cell in 0 ..< Cells:
    if sim.board[cell].len == 0:
      continue
    let line = cellName(cell) & ": " & sim.stackText(sim.board[cell]) &
      " (h" & $sim.board[cell].len & ")"
    if sim.controls(viewer, cell):
      mine.add(line)
    else:
      theirs.add(line)
  result.add("Stacks YOU control (bottom->top, top piece last): " &
    (if mine.len == 0: "none" else: mine.join("; ")) & "\n")
  result.add("Stacks " & sim.seatName(1 - viewer) & " controls: " &
    (if theirs.len == 0: "none" else: theirs.join("; ")) & "\n")

proc renderHistory(sim: Sim): string =
  ## The public record, in reading order (last 24 events keep prompts
  ## short; the board is the state that matters).
  var lines: seq[string]
  for event in sim.events:
    case event.kind
    of evStart:
      lines.add(event.text & " moves first.")
    of evSay:
      lines.add(sim.seatName(event.seat) & " says: \"" & event.text & "\"")
    of evMove:
      var line = sim.seatName(event.seat) & " moves " & $event.count &
        " from " & cellName(event.fromCell) & " " & $event.dir & " to " &
        cellName(event.toCell) & " (now " & sim.stackText(event.stackAfter) &
        ")"
      if event.captured > 0:
        line.add(", capturing " & $event.captured)
      if event.reserved > 0:
        line.add(", banking " & $event.reserved & " to reserve")
      lines.add(line & ".")
    of evPlace:
      var line = sim.seatName(event.seat) & " drops a reserve piece on " &
        cellName(event.toCell) & " (now " & sim.stackText(event.stackAfter) &
        ")"
      if event.captured > 0:
        line.add(", capturing " & $event.captured)
      if event.reserved > 0:
        line.add(", banking " & $event.reserved & " to reserve")
      lines.add(line & ".")
    of evEnd:
      lines.add("The game is over.")
  if lines.len == 0:
    return "(nothing has happened yet)"
  if lines.len > 24:
    lines = @["(earlier moves omitted)"] & lines[^24 .. ^1]
  lines.join("\n")

proc renderLegalMoves*(sim: Sim, seat: int): string =
  ## Exact canonical moveText IDs for every legal move, grouped cleanly so
  ## every comma-separated item is a copyable legal move ID.
  var lines: seq[string]
  for cell in 0 ..< Cells:
    if not sim.controls(seat, cell):
      continue
    var options: seq[string]
    let height = sim.board[cell].len
    for count in 1 .. height:
      for dir in Direction:
        let dest = step(cell, dir, count)
        if dest >= 0:
          options.add(moveText(Move(kind: mkMove, fromCell: cell,
            toCell: dest, count: count, dir: dir)))
    if options.len > 0:
      lines.add(options.join(", "))
  if sim.reserve[seat] > 0:
    var drops: seq[string]
    for cell in 0 ..< Cells:
      if validCell(cell):
        drops.add(moveText(Move(kind: mkPlace, fromCell: -1, toCell: cell)))
        if drops.len == 8:
          lines.add(drops.join(", "))
          drops.setLen(0)
    if drops.len > 0:
      lines.add(drops.join(", "))
  lines.join("\n")

const BoardRules = "Rules:\n- 8x8 board with the three squares in each corner missing (a1 b1 a2, g1 h1\n  h2, a8 b8 a7, g8 h8 h7 are NOT playable). Files a-h, ranks 1-8; N is\n  toward rank 8, E toward file h.\n"
const CanonicalPlayRules = "- Pieces stack. Whoever owns the TOP piece controls the stack.\n- On your turn, MOVE: take the top k pieces (1..height) of a stack you\n  control and move them exactly k squares in one direction (N/S/E/W),\n  jumping over anything, landing on top of whatever is there. The pieces\n  you carry may include enemy pieces underneath your top piece.\n- If a stack ends taller than 5, pieces come off the BOTTOM: your own go\n  to your reserve, enemy pieces are captured for good.\n- Or DROP: instead of moving, place one reserve piece on top of any\n  playable square.\n- A player who cannot move (controls no stack, no reserve) LOSES. At the\n  ply cap the side with more material (pieces in stacks it controls plus\n  reserve) wins.\n- Optional public \"say\" text may be included inside the JSON reply.\n"
const RulesText* = BoardRules & CanonicalPlayRules

proc systemPrompt*(sim: Sim, seat: int, selection: PromptSelection): string =
  let playRules = case selection.selectedVariant
    of pvCanonical: CanonicalPlayRules
    of pvPlain: "Stacks belong to whoever owns their top piece. To MOVE, take the top k pieces of a controlled stack (1 through its height) and travel exactly k squares N, S, E, or W. Jump over intervening pieces and land on top of the destination; enemy pieces below your top piece may travel with you. Trim any stack taller than five from its bottom: return your own pieces to reserve and permanently capture enemy pieces. Alternatively DROP one reserve piece onto any playable square. Having no controlled stack and no reserve means no move and a loss. At the ply cap, more material wins: pieces in controlled stacks plus reserve. An optional public \"say\" field may be included in the JSON reply.\n"
    of pvNumbered: "1. Control a stack by owning its top piece.\n2. MOVE the top k pieces, 1..height, from a controlled stack exactly k squares N/S/E/W. Jump intervening pieces and land on the destination stack; carried lower pieces may belong to the enemy.\n3. Trim stacks above five from the bottom: own pieces to reserve, enemy pieces permanently captured.\n4. Instead of moving, DROP one reserve piece on any playable square.\n5. No controlled stacks and no reserve means no move and a loss. At the ply cap, compare controlled-stack pieces plus reserve; more wins.\n6. Optional public \"say\" text may be placed in the JSON reply.\n"
    of pvQuestions: "Who controls a stack? The owner of its top piece. How do you MOVE? Take its top k pieces, 1..height, and jump exactly k squares N/S/E/W onto the destination; lower enemy pieces travel too. What if the stack exceeds five? Remove bottom pieces: yours enter reserve, enemies are captured permanently. What is the alternative? DROP one reserve piece on any playable square. How do you lose? No controlled stack and no reserve leaves no move. How does the ply cap settle? More pieces in controlled stacks plus reserve wins. Can you speak? Optionally via the JSON \"say\" field.\n"
    of pvReordered: "Win by leaving the opponent unable to move: no controlled stack and no reserve. At the ply cap, more material (pieces in controlled stacks plus reserve) wins. A stack is controlled by its top piece's owner. Either DROP one reserve piece onto any playable square or MOVE the top k pieces of a controlled stack exactly k squares N/S/E/W, with 1 <= k <= height. Jump over intervening pieces, land on top, and carry lower enemy pieces along. Remove excess above five from the bottom: own pieces return to reserve, enemy pieces are captured permanently. Optional public speech belongs only in the JSON \"say\" field.\n"
    of pvHeldoutBriefing: "Control follows the top piece of each stack. On a move, carry the top k pieces of one you control exactly k squares along a cardinal direction, for any k from one to its height. You jump intervening pieces and land above the destination stack; enemy pieces below your top piece can accompany you. A stack over five sheds bottom pieces, sending yours to reserve and capturing enemy pieces permanently. You may instead drop one reserve piece on any playable square. Losing all controlled stacks and reserves leaves no move and loses. The ply cap awards the win to greater controlled-stack material plus reserve. Any optional public message goes in the JSON \"say\" field.\n"
    of pvHeldoutReference: "Control: top-piece owner. MOVE: top k pieces of a controlled stack, k in 1..height, exactly k squares N/S/E/W; jump intervening pieces, land atop destination, carry any lower enemy pieces too. Overflow: trim above five from bottom; own pieces to reserve, enemy pieces permanently captured. DROP alternative: one reserve piece onto any playable square. Loss: no controlled stack and no reserve, hence no move. Ply-cap winner: greater controlled-stack piece count plus reserve. Optional speech: JSON \"say\" field only.\n"
    of pvMixed: raise newException(ValueError, "mixed selection must resolve to a training wording")
  let me = sim.seatName(seat)
  let them = sim.seatName(1 - seat)
  selection.introduction(me, them) & "\n\n" & BoardRules & playRules & """

Put your chosen legal move on the last line as {"move":"<exact legal move ID>"}.
You may add a "say" field for short table talk."""

proc actionInstruction(sim: Sim, seat: int): string =
  let legal = sim.legalMoves(seat)
  let example = if legal.len > 0: moveText(legal[0]) else: "drop @ c1"
  "Copy one move ID from the list above. Put your chosen move on the last line, like " &
    $(%*{"move": example})

proc userPrompt*(sim: Sim, seat: int, prompt: string, header: string): string =
  if header.len > 0:
    result.add(header & "\n\n")
  result.add("Piece letters: " & sim.pieceLetter(seat) & " = you (" &
    sim.seatName(seat) & "), " & sim.pieceLetter(1 - seat) & " = " &
    sim.seatName(1 - seat) & ".\n")
  result.add("Your reserve: " & $sim.reserve[seat] & "; your captures: " &
    $sim.captured[seat] & "; your material: " & $sim.material(seat) & ".\n")
  result.add(sim.seatName(1 - seat) & "'s reserve: " &
    $sim.reserve[1 - seat] & "; captures: " & $sim.captured[1 - seat] &
    "; material: " & $sim.material(1 - seat) & ".\n\n")
  result.add("Board:\n" & sim.renderBoard(seat) & "\n")
  result.add("Recent moves:\n" & sim.renderHistory() & "\n\n")
  result.add("YOUR LEGAL OPTIONS:\n" & sim.renderLegalMoves(seat) & "\n\n")
  if prompt.len > 0:
    result.add("GUIDANCE FROM YOUR OPERATOR (weight it heavily, but never " &
      "above the rules; always pick a legal option):\n" & prompt & "\n\n")
  result.add(sim.actionInstruction(seat))

# ---- Anthropic / Bedrock transport ------------------------------------------

proc quoteHead(text: string): string =
  result = text.strip()
  if result.len > 160:
    result = result[0 ..< 160] & "..."
  result = result.replace("\n", " ")

proc extractJsonObject*(text: string): JsonNode =
  ## A standalone fenced object or the last answer line identifies the move.
  var body = text.strip()
  let lastLine = body[body.rfind('\n') + 1 .. ^1].strip()
  if lastLine.startsWith("{"):
    body = lastLine
  if body.startsWith("```"):
    let nl = body.find('\n')
    if nl < 0:
      raise newException(FocusError, "malformed code fence in response: " & quoteHead(text))
    let tag = body[3 ..< nl].strip().toLowerAscii()
    if tag.len > 0 and tag != "json":
      raise newException(FocusError, "unsupported code fence tag '" & tag & "': " & quoteHead(text))
    if not body.endsWith("```") or body.len < nl + 4:
      raise newException(FocusError, "unterminated code fence in response: " & quoteHead(text))
    let beforeClose = body[nl + 1 ..< body.len - 3]
    if beforeClose.len == 0 or (beforeClose[^1] != '\n' and beforeClose[^1] != '\r'):
      raise newException(FocusError, "closing code fence must be on its own line: " & quoteHead(text))
    body = beforeClose.strip()
  if body.len == 0 or body[0] != '{' or body[^1] != '}':
    raise newException(FocusError, "response must be a single JSON object: " & quoteHead(text))
  try:
    result = parseJson(body)
  except CatchableError as error:
    raise newException(FocusError, "invalid JSON object (" & error.msg & "): " & quoteHead(text))
  if result.kind != JObject:
    raise newException(FocusError, "response must be a JSON object: " & quoteHead(text))

type
  CompletionResult = object
    text: string
    stopReason: string
    outputTokens: Option[int]
    blockTypes: seq[string]

proc formatCompletionMeta(meta: CompletionResult): string =
  let stop = if meta.stopReason.len > 0: meta.stopReason else: "unknown"
  let tokens =
    if meta.outputTokens.isSome: $meta.outputTokens.get()
    else: "unknown"
  "stop_reason=" & stop & ", output_tokens=" & tokens &
    ", blocks=[" & meta.blockTypes.join(",") & "]"

proc completeText(client: LlmClient, system, user: string, slot: int,
    policy: string, ply: int): CompletionResult =
  var body = %*{
    "max_tokens": client.maxOutputTokens,
    "system": system,
    "messages": [{"role": "user", "content": user}]
  }
  var headers: HttpHeaders
  if client.transport == ltSidecar and slot >= 0:
    headers["X-Coworld-Player-Slot"] = $slot
  headers["content-type"] = "application/json"
  var url: string
  if client.transport == ltBedrock:
    body["anthropic_version"] = %BedrockAnthropicVersion
    if client.bedrockToken.len > 0:
      headers["authorization"] = "Bearer " & client.bedrockToken
    url = client.bedrockUrl()
  elif client.transport == ltSidecar:
    body["model"] = %client.model
    headers["anthropic-version"] = AnthropicVersion
    url = client.sidecarEndpoint & "/v1/messages"
  else:
    body["model"] = %client.model
    ## Only the Claude 5 / Opus tiers accept an effort setting; Haiku 4.5
    ## rejects the whole request with a 400 if it is present.
    if "haiku" notin client.model and "4-5" notin client.model:
      body["output_config"] = %*{"effort": "low"}
    headers["x-api-key"] = client.apiKey
    headers["anthropic-version"] = AnthropicVersion
    url = AnthropicUrl
  let capture = getEnv(CogameSaveTrajectoryUriEnv).len > 0
  if capture: client.nativeAttempts.add(nativeAttempt(policy, ply, slot, 0, $body))
  let started = epochTime()
  var response: Response
  try:
    response = client.curl.post(url, headers, $body, client.timeoutSeconds)
  except CatchableError as error:
    if capture:
      client.nativeAttempts[^1].latencyMs = some(float(int((epochTime() - started) * 1000)))
    let lower = error.msg.toLowerAscii()
    client.lastFailureCause =
      if "timeout" in lower or "timed out" in lower: "timeout"
      else: "provider"
    raise newException(FocusError, "llm transport failed: " & error.msg)
  if capture:
    client.nativeAttempts[^1].receiveNativeResponse(response, "", int((epochTime() - started) * 1000))
  if response.code == 401 or response.code == 403:
    client.lastFailureCause = "provider"
    let detail = response.body[0 .. min(response.body.high, 400)]
    client.disabled = true
    raise newException(FocusError,
      "llm auth failed (" & $response.code & ") at " & url & ": " & detail)
  if response.code == 429:
    client.lastFailureCause = "provider"
    let detail = response.body[0 .. min(response.body.high, 300)]
    raise newException(FocusError, "llm throttled (429): " & detail)
  if response.code < 200 or response.code >= 300:
    client.lastFailureCause = "provider"
    raise newException(FocusError, "anthropic error " & $response.code &
      ": " & response.body[0 .. min(response.body.high, 300)])
  var payload: JsonNode
  try:
    payload = parseJson(response.body)
  except CatchableError as error:
    client.lastFailureCause = "provider"
    raise newException(FocusError, "invalid provider JSON: " & error.msg)
  if payload.kind != JObject or not payload.hasKey("content") or payload["content"].kind != JArray:
    client.lastFailureCause = "provider"
    raise newException(FocusError, "invalid provider payload: missing content array")
  result.stopReason = payload{"stop_reason"}.getStr()
  let usage = payload{"usage"}
  if not usage.isNil and usage.kind == JObject and
      usage.hasKey("output_tokens") and usage["output_tokens"].kind == JInt:
    result.outputTokens = some(usage["output_tokens"].getInt())
  for contentBlock in payload["content"]:
    let blockType = contentBlock{"type"}.getStr("unknown")
    result.blockTypes.add(blockType)
    if blockType == "text":
      result.text.add(contentBlock{"text"}.getStr())
  if result.stopReason == "refusal":
    client.lastFailureCause = "model_rejection"
    raise newException(FocusError, "anthropic refusal (" &
      formatCompletionMeta(result) & ")")
  if result.stopReason == "max_tokens" and '{' notin result.text:
    client.lastFailureCause = "model_rejection"
    raise newException(FocusError, "reply cut off at max_tokens before " &
      "any JSON (" & formatCompletionMeta(result) & "): " &
      quoteHead(result.text))

proc cleanSay*(text: string): string =
  result = text.strip()
  if result.len <= MaxSayLen:
    return
  ## A model that overshoots the stated cap gets cut at a word boundary
  ## with the cut marked.
  result = result[0 ..< MaxSayLen - 3]
  while result.len > 0 and (result[^1].ord and 0xC0) == 0x80:
    result.setLen(result.len - 1)
  let space = result.rfind(' ')
  if space > MaxSayLen div 2:
    result.setLen(space)
  result.add("…")

proc parseMoveId*(sim: Sim, seat: int, moveId: string): Move =
  ## Matches `moveId` against the exact canonical moveText IDs for `seat`'s
  ## legal moves. No fuzzy matching, no alternate wire formats.
  for move in sim.legalMoves(seat):
    if moveText(move) == moveId:
      return move
  raise newException(FocusError, "illegal move: " & moveId)

proc parseDecision*(sim: Sim, seat: int, payload: JsonNode): Decision =
  ## Maps {"move": "<exact legal moveText ID>", "say": "..." (optional)} onto
  ## a legal Move for `seat`.
  if payload.isNil or payload.kind != JObject:
    raise newException(FocusError, "response must be a JSON object")
  for key in payload.keys:
    if key notin ["move", "say"]:
      raise newException(FocusError, "unexpected field in response: " & key)
  if payload.hasKey("say") and payload["say"].kind != JNull:
    if payload["say"].kind != JString:
      raise newException(FocusError, "say must be a string")
    result.say = cleanSay(payload["say"].getStr())
  if not payload.hasKey("move") or payload["move"].kind != JString:
    raise newException(FocusError, "move must be a legal move ID string")
  result.move = sim.parseMoveId(seat, payload["move"].getStr())

proc decisionJson*(decision: Decision): JsonNode =
  result = %*{"move": moveText(decision.move)}
  if decision.say.len > 0:
    result["say"] = %decision.say

proc decide*(
  client: LlmClient,
  sim: Sim,
  seat: int,
  prompt: string,
  scripted: bool,
  selection: PromptSelection,
  header = ""
): Decision =
  ## One decision for one seat. Scripted seats play the minimax baseline;
  ## prompt seats require a valid legal model reply and raise FocusError on
  ## failure so the server fails the episode instead of substituting a move.
  client.lastFailureCause = ""
  let capture = getEnv(CogameSaveTrajectoryUriEnv).len > 0
  if capture:
    client.nativeAttempts.setLen(0)
  if scripted:
    return client.scriptedAction(sim, seat)
  if client.disabled:
    client.lastFailureCause = if client.transport == ltNone: "missing" else: "provider"
    raise newException(FocusError,
      if client.transport == ltNone: "no LLM credentials configured"
      else: "LLM transport disabled")
  let system = systemPrompt(sim, seat, selection)
  let user = userPrompt(sim, seat, prompt, header)
  var completion: CompletionResult
  var hasCompletion = false
  try:
    completion = client.completeText(system, user, seat,
      sim.config.players[seat].name, sim.ply)
    hasCompletion = true
    client.lastFailureCause = "model_rejection"
    let payload = extractJsonObject(completion.text)
    let decision = parseDecision(sim, seat, payload)
    var probe = sim
    probe.applyMove(seat, decision.move)
    client.lastFailureCause = ""
    if capture:
      client.nativeAttempts[^1].parsedAction = decision.decisionJson()
      let rawSay =
        if payload.hasKey("say") and payload["say"].kind == JString:
          payload["say"].getStr()
        else:
          ""
      client.nativeAttempts[^1].accepted = rawSay == decision.say
      if not client.nativeAttempts[^1].accepted:
        client.nativeAttempts[^1].rejectionReason = some("native parser normalized speech")
    return decision
  except CatchableError as error:
    if client.lastFailureCause.len == 0:
      client.lastFailureCause = "provider"
    let detail =
      if hasCompletion and client.lastFailureCause == "model_rejection":
        error.msg & " (" & formatCompletionMeta(completion) & ")"
      else:
        error.msg
    if capture and client.nativeAttempts.len > 0:
      client.nativeAttempts[^1].accepted = false
      client.nativeAttempts[^1].rejectionReason = some(detail)
    echo "focus llm: seat ", seat, " failed: ", detail
    raise newException(FocusError, "model decision failed: " & detail)
