"""[dgxarley] flashinfer_backend.py: fehlendes `uniform_q_len`-Argument in
SGLangs fast_prefill_plan nachreichen (ABI-Bruch flashinfer >= 0.6.16).

DER BEFUND (2026-07-28, Image 0.5.16-dev-sm121, flashinfer 0.6.16rc3):
Spekulatives Decoding (NEXTN/EAGLE) killt den Scheduler beim ersten Request:

    File .../sglang/srt/speculative/eagle_worker_v2.py, in _draft_extend_for_decode
    File .../eagle_draft_extend_cuda_graph_runner.py, in execute
    File .../sglang/srt/layers/attention/flashinfer_backend.py:297, in fast_prefill_plan
        self._plan_info = self._cached_module.plan(*args)
    TypeError: Mismatched number of arguments when calling: `plan(...)`.
               Expected 20 but got 19 arguments

Nach aussen sah das wie ein Haenger aus: der Scheduler stirbt, der HTTP-Warmup
wartet danach 600 s in den Timeout ("No live scheduler processes found") und der
Server meldet nur "Initialization failed. warmup error". Normales Serving ist
NICHT betroffen (GSM8K 19/20 auf demselben Image gemessen) — nur der
Draft-Extend-CUDA-Graph, also Spekulation.

URSACHE: `fast_prefill_plan` ist SGLangs Abkuerzung, die flashinfers JIT-Modul
`plan()` DIREKT aufruft und dabei die Argumentliste des fa2-Backends von Hand
nachbaut, statt flashinfers Python-Wrapper zu benutzen. flashinfer 0.6.16 haengt
dieser Liste ein viertes Tail-Argument an (flashinfer/prefill.py):

    args.append(fixed_split_size or -1)   # fixed_split_size
    args.append(disable_split_kv)         # disable_split_kv
    args.append(0)                        # num_colocated_ctas
    args.append(0)                        # uniform_q_len   <-- NEU in 0.6.16

SGLang uebergibt weiterhin 19 Argumente, in v0.5.15.post1 UND v0.5.16 (beide
gezaehlt), weil upstream flashinfer 0.6.14 pinnt. Der Bruch entsteht also
ausschliesslich durch unseren FLASHINFER-Override im Recipe, nicht durch den
SGLang-Bump — er wuerde das Produktionsimage genauso treffen, sobald dort
flashinfer >= 0.6.16 landet.

FIX: das fehlende `0,  # uniform_q_len` anhaengen. Der Default ist auch
flashinfer-seitig 0, das Verhalten bleibt damit unveraendert.

UPSTREAM-FIX seit v0.5.18 (2026-08-28): SGLang hat seinen eigenen flashinfer-Pin
auf 0.6.17 gezogen und reicht das Argument in fast_prefill_plan jetzt selbst
durch. Auf so einem Image ist der Anker (args-Literal OHNE uniform_q_len)
zwangslaeufig weg -- das ist KEIN Drift, sondern erledigt. Das Gate unten prueft
das explizit, damit der Acceptance-Gate-Report eine Arbeitsliste bleibt und nicht
in Rauschen untergeht. Der Patch bleibt fuer Instanzen, die noch auf
<= v0.5.17-Images gepinnt sind; loeschbar, sobald keine mehr laeuft.

GATE: nur anwenden, wenn das INSTALLIERTE flashinfer den Parameter ueberhaupt
kennt. Auf flashinfer <= 0.6.15.post1 (kein `uniform_q_len` in prefill.py) wuerde
das 20. Argument den Aufruf spiegelbildlich brechen. Eine ConfigMap bedient
Instanzen mit verschiedenen Images, deshalb die Praesenzpruefung statt einer
Versionsannahme.

DELETABLE, sobald SGLang selbst nachzieht (dann traegt die args-Liste
`uniform_q_len` bereits und die Already-applied-Probe greift). Upstream-Kandidat:
`fast_prefill_plan` sollte die Tail-Argumente nicht hartkodieren.
"""

from _patchlib import Patch, target_contains

_TARGET = "sglang/srt/layers/attention/flashinfer_backend.py"

patch = Patch(
    name="fast_prefill_plan: uniform_q_len fuer flashinfer >= 0.6.16 nachreichen",
    target=_TARGET,
    when=(
        # (a) kennt das INSTALLIERTE flashinfer den Parameter ueberhaupt?
        target_contains("flashinfer/prefill.py", "uniform_q_len")
        # (b) reicht SGLang ihn nicht laengst selbst durch? Ab v0.5.18 steht
        # "0,  # uniform_q_len" im args-Literal, weil upstream seinen eigenen
        # flashinfer-Pin auf 0.6.17 gezogen hat. Dann ist hier nichts zu tun,
        # und das ist eine ENTSCHEIDUNG ("gate not matched"), kein Drift.
        and not target_contains(_TARGET, "0,  # uniform_q_len")
    ),
)

OLD = """        fixed_split_size if fixed_split_size is not None else -1,
        False,  # disable_split_kv
        0,  # num_colocated_ctas
    ]
"""

NEW = """        fixed_split_size if fixed_split_size is not None else -1,
        False,  # disable_split_kv
        0,  # num_colocated_ctas
        0,  # [patch] uniform_q_len -- neu in flashinfer 0.6.16, sonst
        # "Expected 20 but got 19 arguments" im Draft-Extend-CUDA-Graph
    ]
"""


@patch.run
def apply(p: Patch) -> None:
    p.replace(OLD, NEW, marker="# [patch] uniform_q_len", what="fast_prefill_plan uniform_q_len")
