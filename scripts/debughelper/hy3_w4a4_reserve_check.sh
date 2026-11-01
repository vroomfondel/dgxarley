#!/usr/bin/env bash
# ============================================================================
# Hy3-NVFP4-W4A4 clean-reserve coherence check (2026-07-10)
# Run AFTER `ansible-playbook k8s_dgx.yml --tags sglang` has deployed
# vroomfondel/Hy3-NVFP4-W4A4 and the head is Ready.
#
# Purpose: decide the open question after the flashinfer JIT-cache clear —
#   COHERENT output  -> the matrix NaN was a stale/old-image artifact; W4A4 is fine.
#   garbage / NaN     -> "all fixes present, still NaN" is a real residual
#                        -> next: per-layer activation dump to localise the first inf.
#
# Garbage signatures to watch for (from the profile / TESTLOGS):
#   - "!!!!" single-char repetition        (triton attn path)
#   - empty / 500 / NaN device-assert      (flashinfer path)
#   - multilingual token salad             (marlin path)
# ============================================================================
set -uo pipefail
EP="${EP:-https://sglang.dgx.example.com}"
MODEL="${MODEL:-vroomfondel/Hy3-NVFP4-W4A4}"
MAXTOK="${MAXTOK:-256}"

echo ">>> endpoint=$EP  model=$MODEL"
echo ">>> 1) waiting for /v1/models to return 200 (head Ready) ..."
for i in $(seq 1 120); do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$EP/v1/models" 2>/dev/null)
  [ "$code" = "200" ] && { echo "    ready (HTTP 200) after ~$((i*10))s"; break; }
  printf '    [%3ds] HTTP %s\r' "$((i*10))" "$code"; sleep 10
done
[ "$code" != "200" ] && { echo "    NEVER became ready — check: kubectl --context=ht@dgxarley get pods -n sglang"; exit 1; }

ask () {  # $1 = label, $2 = prompt
  echo; echo "=================== $1 ==================="
  resp=$(curl -s "$EP/v1/chat/completions" -H 'Content-Type: application/json' -d "$(python3 -c '
import json,sys
print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],
                  "max_tokens":int(sys.argv[3]),"temperature":0.7,"top_p":1.0,"top_k":-1}))
' "$MODEL" "$2" "$MAXTOK")")
  echo "$resp" | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    if "choices" in d:
        m=d["choices"][0]["message"]
        out=(m.get("content") or "")+ (("\n[reasoning] "+m["reasoning_content"]) if m.get("reasoning_content") else "")
        print(out.strip()[:1200] or "<EMPTY OUTPUT>")
        print("\n  finish_reason:",d["choices"][0].get("finish_reason"))
    else:
        print("ERROR RESPONSE:",json.dumps(d)[:600])
except Exception as e:
    print("PARSE FAIL:",e,"\nraw:",sys.stdin.read()[:600])
'
}

ask "A) German coherence"   "Erkläre in drei Sätzen, warum der Himmel blau ist."
ask "B) Reasoning / math"   "Ein Zug fährt 240 km in 3 Stunden, dann 150 km in 1,5 Stunden. Wie hoch ist die Durchschnittsgeschwindigkeit über die Gesamtstrecke? Denke Schritt für Schritt."
ask "C) English + code"     "Write a Python function that returns the nth Fibonacci number iteratively. One short sentence, then the code."

echo; echo ">>> VERDICT GUIDE:"
echo "    coherent German + correct ~86.7 km/h + valid fib() -> W4A4 SERVES (matrix NaN was stale/old-image). DONE."
echo "    '!!!!' / empty / salad / 500 -> residual real -> per-layer activation dump next."
