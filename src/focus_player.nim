## Focus player: prompt or scripted baseline.
##
##
## PLAYER_SCRIPTED=1 registers the seat as the built-in minimax baseline
## instead: the server plays it deterministically, no LLM.
##
## To field your own policy, reuse this image and set PLAYER_PROMPT:
##   coworld upload-policy <focus-image> --name my-focus \
##     --run /bin/focus-player --secret-env PLAYER_PROMPT="<your strategy>"

import
  std/[json, options, os, strutils],
  whisky

const DefaultPrompt =
  "Play strong Focus: capture enemy pieces by overflowing tall stacks, " &
  "control multiple stacks to preserve mobility, and bank reserve pieces " &
  "for key drops."

when isMainModule:
  let url = getEnv("COWORLD_PLAYER_WS_URL")
  if url.len == 0:
    quit("COWORLD_PLAYER_WS_URL is not set", 1)
  var prompt = getEnv("PLAYER_PROMPT")
  if prompt.len == 0:
    prompt = DefaultPrompt
  let scripted = getEnv("PLAYER_SCRIPTED").strip() in ["1", "true", "yes"]

  proc promptFrame(): string =
    $ %*{"type": "prompt", "prompt": prompt, "scripted": scripted}

  echo "focus player: connecting to game"
  let socket = newWebSocket(url)
  socket.send(promptFrame())
  echo "focus player: prompt delivered (", prompt.len, " chars",
    (if scripted: ", scripted" else: ""), ")"

  while true:
    let received = socket.receiveMessage()
    if received.isNone:
      echo "focus player: connection closed, exiting"
      break
    let message = received.get()
    if message.kind != TextMessage:
      continue
    try:
      let payload = parseJson(message.data)
      case payload{"type"}.getStr()
      of "welcome":
        echo "focus player: seated at slot ",
          payload{"slot"}.getInt(), " as ", payload{"name"}.getStr()
        ## Re-deliver the prompt after the welcome, in case the first send
        ## raced the server's slot registration.
        socket.send(promptFrame())
      of "final":
        echo "focus player: final scores ", payload{"scores"}
        break
      of "error":
        echo "focus player: episode failed (", payload{"cause"}.getStr(), "): ",
          payload{"detail"}.getStr()
        break
      else:
        discard
    except CatchableError as error:
      echo "focus player: ignoring bad frame: ", error.msg
  socket.close()
