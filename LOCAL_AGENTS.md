# LOCAL_AGENTS.md

Overview of current (as of April 2026) open-source agents that can be wired
up to a locally running LLM API — for example, SGLang or vLLM on the DGX
Sparks (`http://<host>:8000/v1` or `:8001/v1`, OpenAI-compatible) or Ollama
on spark1 (`http://spark1:11434/v1`).

Every agent listed here speaks at least the OpenAI Chat Completions dialect
and accepts `base_url` + `api_key` as configuration. "Local" here means: no
mandatory sign-up account, no telemetry requirement, runs offline against a
self-hosted endpoint.

---

## Coding agents (CLI / TUI)

### OpenCode
- Repo: <https://github.com/sst/opencode> · Web: <https://opencode.ai>
- Language: TypeScript (+ Rust/Tauri). Terminal TUI.
- License: MIT.
- Providers: 75+ via [Models.dev](https://models.dev), including Ollama,
  LM Studio, vLLM, SGLang, and any OpenAI-compatible endpoint.
- Local setup: register a custom provider in
  `~/.config/opencode/opencode.json` (`"baseURL":
  "http://spark1:8000/v1"`). Tool use goes through the standard `tools`
  field of the Chat Completions API, so the model has to support tool
  calling (Qwen3-Coder, GLM-4.7, DeepSeek-V4, Hermes-3/4).
- Strengths: polished TUI, multi-session, MCP support, very active
  development. Overtook both Cline and OpenHands on GitHub stars in early
  2026.

### Aider
- Repo: <https://github.com/Aider-AI/aider>
- Language: Python. Classic CLI, git-centric.
- License: Apache 2.0.
- Providers: anything `litellm` knows about → Ollama, LM Studio, vLLM,
  SGLang, any OpenAI-compatible API.
- Local setup:
  ```bash
  export OPENAI_API_BASE=http://spark1:8000/v1
  export OPENAI_API_KEY=dummy
  aider --model openai/Qwen/Qwen3-Coder-30B-A3B-Instruct
  ```
- Strengths: every edit becomes a git commit, mature diff/patch modes
  (`whole`, `diff`, `udiff`), works reliably even with weaker models —
  no hard tool-calling requirement.

### Cline
- Repo: <https://github.com/cline/cline> · VS Code extension.
- License: Apache 2.0.
- Providers: OpenRouter, Anthropic, OpenAI, Ollama, LM Studio, "OpenAI
  Compatible".
- Local setup: in the extension settings pick `OpenAI Compatible`, fill in
  base URL and model name.
- Strengths: native to VS Code, plan/act modes, good diff preview in the
  editor.
- Weakness: needs a strong tool-calling model, otherwise productivity
  drops noticeably.

### Continue
- Repo: <https://github.com/continuedev/continue> · VS Code and JetBrains
  extension.
- License: Apache 2.0.
- Providers: nearly all, including a generic `openai` configuration with
  `apiBase`.
- Local setup: in `~/.continue/config.yaml`:
  ```yaml
  models:
    - name: qwen3-coder-local
      provider: openai
      model: Qwen/Qwen3-Coder-30B-A3B-Instruct
      apiBase: http://spark1:8000/v1
      apiKey: dummy
  ```
- Strengths: chat, autocomplete, and edit in one; very fine-grained
  configuration.

### Goose (Block)
- Repo: <https://github.com/block/goose> · Docs: <https://goose-docs.ai>
- Language: Rust (CLI + desktop).
- License: Apache 2.0.
- Providers: 25+, including Ollama, OpenAI-compatible, Anthropic, Bedrock.
- Tool system: MCP-native (Block co-designed MCP with Anthropic) — every
  extension is an MCP server.
- Local setup: `goose configure` → `OpenAI Compatible` → enter base URL,
  or pick the Ollama provider directly.
- Strengths: strictly MCP-based, which makes the extension model very
  clean. Also runs headless (`goose run`).

### Crush (Charm)
- Repo: <https://github.com/charmbracelet/crush>
- Language: Go. Excellent TUI (Charm stack: Bubble Tea, Lip Gloss).
- License: FSL-1.1-MIT (becomes MIT after two years).
- Providers: OpenAI-compatible, Anthropic, Ollama; mid-session model
  switching while preserving context.
- Local setup: `crush.json` with `providers.<name>.base_url`.
- Strengths: fast, broad platform coverage (including BSDs), LSP
  integration, MCP.

### OpenHands (formerly OpenDevin)
- Repo: <https://github.com/All-Hands-AI/OpenHands>
- Docs: <https://docs.openhands.dev>
- Language: Python + TS frontend.
- License: MIT.
- Providers: anything via `litellm`; official guides for Ollama, vLLM,
  SGLang.
- Sandbox: execution defaults to a Docker container — good for autonomous
  agents, but the overhead is non-trivial.
- Local setup: `LLM_BASE_URL=http://spark1:8000/v1`,
  `LLM_MODEL=openai/...`.
- Strengths: multi-agent delegation, browser automation built in, its own
  coding model `OpenHands-LM` available. More of a platform than a small
  CLI.
- Weakness: relatively heavyweight; models smaller than ~30B often can't
  keep up.

---

## Personal general-purpose agent

### Hermes Agent (Nous Research)

Hermes Agent is not purely a coding agent but a persistent personal agent
with a built-in learning loop. It explicitly positions itself as "the
agent that grows with you". Current version: **v0.11.0**, released
April 23, 2026, MIT-licensed.

- Repo: <https://github.com/NousResearch/hermes-agent>
- Web: <https://hermes-agent.nousresearch.com>
- Docs: <https://hermes-agent.nousresearch.com/docs/>
- Language: Python (~88%), TypeScript (~9%).

#### Model providers and local hookup

Hermes talks to any OpenAI-compatible endpoint and additionally ships
native provider integrations:

- Nous Portal (their own hosted models)
- OpenRouter (200+ models)
- NVIDIA NIM (Nemotron)
- Xiaomi MiMo, z.ai/GLM, Kimi/Moonshot, MiniMax
- Hugging Face, OpenAI
- **Custom endpoints** — this is where SGLang, vLLM, Ollama, and LM Studio
  plug in.

Switch models with no code changes:
```bash
export OPENAI_BASE_URL=http://spark1:8000/v1
export OPENAI_API_KEY=dummy
hermes setup            # one-time wizard
hermes model            # interactive model selection
hermes                  # start CLI chat
```
Inside a running chat you can switch on the fly with
`/model openai:Qwen/Qwen3-Coder-30B-A3B-Instruct`.

#### Architecture in bullet points

- **Agent loop**: classic decision cycle (user input → tool selection →
  execution → memory update → response). Iterations are interruptible
  and redirectable (interrupt-and-redirect).
- **Tool system**: 47 built-in tools for file I/O, web browsing, terminal
  execution, code execution, system utilities. Toolsets bundle tools for
  granular permission control.
- **`execute_code` tool**: collapses multi-step pipelines into a single
  inference call — instead of one round trip per tool, the model emits a
  small Python snippet that calls multiple tools sequentially or in
  parallel via RPC. Saves significant tokens and latency.
- **Parallel tool execution**: independent tool calls run concurrently
  through a `ThreadPoolExecutor` with up to 8 workers.
- **Subagents**: isolated subagents can be spawned for parallel
  workstreams; Python scripts call tools via RPC.
- **MCP client**: in addition to the native tools, arbitrary MCP servers
  can be plugged in.

#### Memory system (the actual differentiator)

- **Persistent user profiles** via *Honcho* (dialectic user modeling) —
  Hermes builds a model across sessions of who you are, what you're
  working on, and how you work.
- **Agent-curated memory with nudges**: the agent decides itself,
  periodically, what's worth remembering and gets nudged by the system
  to do so, instead of blindly logging everything.
- **`MEMORY.md` and `USER.md`**: entries are stored in plain readable
  Markdown files — you can inspect and edit them like notes.
- **FTS5 full-text search** across all past sessions, with LLM
  summarization for cross-session recall.
- **Skill system**: after complex, successfully completed tasks Hermes
  distills them into a reusable "skill" and files it in a skill
  library. Skills self-improve through repeated use (self-improvement
  loop). Format compatible with the open
  [agentskills.io](https://agentskills.io) standard.

#### Deployment backends

Hermes runs across six terminal backends — the agent doesn't have to live
on the same machine as the LLM:

- `local` — directly on the machine.
- `docker` — sandbox container.
- `ssh` — the agent terminals into a remote host (relevant for the DGX
  cluster: agent on k3smaster, tools execute on the sparks).
- `daytona` — Daytona workspaces.
- `singularity` — HPC containers.
- `modal` — Modal serverless with hibernation.

The full stack runs anywhere from a 5 €/mo VPS to a GPU cluster.

#### Multi-platform gateway

Through the built-in gateway (`hermes gateway`) the same agent is
simultaneously reachable via Telegram, Discord, Slack, WhatsApp, Signal,
Matrix, email, and CLI. Voice memos are transcribed; sessions continue
across platforms.

#### Scheduling

Built-in cron scheduler for natural-language tasks ("every morning at
8 a.m. summarize last night's logs and DM them to me on Telegram").
Delivery happens through the gateway.

#### Slash commands in chat

Selection of the most useful ones:

| Command | Purpose |
|---------|---------|
| `/new`, `/reset` | fresh conversation |
| `/model [provider:model]` | switch model mid-session |
| `/personality [name]` | set persona |
| `/retry`, `/undo` | revise / redo last action |
| `/skills` | browse skill library |
| `/compress` | actively compress context |
| `/usage` | token / cost view |

#### CLI commands

| Command | Purpose |
|---------|---------|
| `hermes setup` | interactive setup wizard |
| `hermes model` | provider / model selection |
| `hermes tools` | enable / disable tools |
| `hermes config set …` | individual settings |
| `hermes` | start CLI chat |
| `hermes gateway` | activate messaging bridge |
| `hermes doctor` | diagnostics |
| `hermes update` | self-update |

#### Research features

Hermes also ships tooling for RL training of agents on top: batch
trajectory generation, trajectory compression for training tool-calling
models, optional *Atropos* RL environments. Not relevant for pure
daily-driver use, but worth noting.

#### Where it fits

Hermes is the right pick if you want **one** agent as a personal
companion that does coding *among other things* and also covers
research, shell automation, recurring reports, and cross-device chat —
and that builds a picture of you and your projects across weeks and
months. For pure IDE coding, Aider or OpenCode are more focused; for
"agent with memory instead of throwaway sessions", Hermes is clearly
ahead.

---

## Quick decision matrix

| Use case | Recommendation |
|---|---|
| Fast pair programmer in the terminal, git-centric | **Aider** |
| Modern terminal TUI, many providers | **OpenCode** or **Crush** |
| Stay in VS Code | **Cline** or **Continue** |
| Clean MCP ecosystem, headless capable | **Goose** |
| Autonomous agents with sandbox and browser | **OpenHands** |
| Persistent general-purpose agent that learns | **Hermes Agent** |

---

## What the models need to support

A short note on terminology first: "tool calling" in the OpenAI-API sense
only means that the *model signals* a request to invoke a tool. Actually
running the tool is always the client's job. What the inference backend
does is parse the model's tool-call tokens and serialize them into the
structured `tool_calls` schema — sometimes loosely called "server-side
tool calling", but more precisely **native tool-call parsing**.

For the *coding* agents (Aider excepted), the inference backend has to
support native tool-call parsing — i.e. it must recognize the model's
tool-call tokens (every model family uses a different convention, e.g.
Hermes-3 `<tool_call>` XML, Qwen3 / Llama-3.x each their own) and emit
them as proper `tool_calls` entries in the response. SGLang configures
this per model via `--tool-call-parser`; vLLM has the same flag. With
the existing SGLang deployment configuration this works out of the box
for the NVFP4 models `glm-4.7`, `qwen3-coder-*`, `qwen3-235b`, and
`nemotron-3` (see `roles/k8s_dgx/model_profiles/`). vLLM 0.17.0 speaks
the same tool-call dialect.

Aider and Hermes Agent also work fine with models that lack native
tool-call parsing — Aider uses structured diff patches in the prompt,
Hermes has its own parser heuristics on top of the raw text and uses
native tool calling only as a fast path when available.

---

## Sources

- <https://opencode.ai/docs/cli/>
- <https://github.com/Aider-AI/aider>
- <https://github.com/cline/cline>
- <https://github.com/continuedev/continue>
- <https://github.com/block/goose>
- <https://github.com/charmbracelet/crush>
- <https://docs.openhands.dev/openhands/usage/llms/local-llms>
- <https://github.com/NousResearch/hermes-agent>
- <https://hermes-agent.nousresearch.com/docs/>
- <https://thenewstack.io/open-source-coding-agents-like-opencode-cline-and-aider-are-solving-a-huge-headache-for-developers/>
- <https://github.com/bradAGI/awesome-cli-coding-agents>
