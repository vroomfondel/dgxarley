# Open WebUI – Architecture, Data Isolation & API Integration

## Why Open WebUI as the Central Platform?

Open WebUI is far more than a chat interface for LLMs. It has established itself as the **de facto standard** for self-hosted AI interfaces and acts as a **central AI gateway** with enterprise-grade capabilities.

### Maturity & Adoption

Open WebUI is one of the fastest-growing open-source projects in the AI space:

- **125,000+ GitHub Stars** and **17,700+ Forks** – placing it among the most popular open-source projects overall
- **341,000+ community members** and **290+ million downloads**
- **Active development** – regular releases (currently v0.8.x), with a large maintainer team and a vibrant contributor community
- **Wide adoption** – from individual developers to research institutions to companies running it in production

This widespread adoption means: extensive documentation, fast bugfixes, a large ecosystem of plugins/pipelines, and the assurance of not betting on a niche project. Open WebUI is no longer experimental software but a **mature, production-ready platform**.

### Core Advantages for Our Use Case

1. **Single Sign-On (SSO)** – Native OIDC/OAuth 2.0 support. Users authenticate via a central identity provider; no separate password management required.
2. **Strict per-user data isolation** – Every user sees exclusively their own documents and knowledge bases. Role-based access control (RBAC) allows fine-grained control down to the group level.
3. **OpenAI-compatible API** – Open WebUI exposes a complete REST API compatible with the OpenAI standard. This makes it a **universal API endpoint** through which external systems, scripts, and automations can use all features — including pipelines.
4. **Pipelines as orchestrable workflows** – Complex processes (RAG, web search, agents, external API calls) are defined as pipelines and can be used identically via both the web interface and the API.

---

## 1. Data Isolation & Permission System (RBAC)

### 1.1 Document Upload in Chat (private by default)

When a user drags a document directly into a chat, it is **accessible only to that user in that chat**. Other users can neither see nor search the file.

### 1.2 Central Knowledge Bases (Workspace)

Under **Workspace → Knowledge**, document collections can be managed systematically:

| Setting | Description |
|---|---|
| **Visibility** | Each knowledge base can be set to *Private* — only the creator has access. |
| **Group-based access** | Admins create groups (e.g., "Team A") and grant specific read/write permissions on knowledge bases to particular groups. |

### 1.3 RBAC Configuration (Admin)

Via **Admin → Users → Groups → Default Permissions**, the following restrictions can be applied:

- **Prohibit "Public Knowledge"** – Users cannot set knowledge bases to *public*.
- **Lock workspace access** – Users can only use temporary chat documents but cannot create persistent knowledge bases.

> **Conclusion:** As long as documents are not explicitly shared via Admin or group permissions, **strict multi-tenancy** exists between users.

---

## 2. Single Sign-On (SSO) via OAuth 2.0 / OIDC

Open WebUI offers native SSO support via the generic OIDC standard. This makes it compatible with any common identity provider:

- Google Workspace / Google Cloud
- Microsoft Entra ID (formerly Azure AD)
- Keycloak
- Authentik, Authelia
- Okta, GitHub

### 2.1 Configuration (Environment Variables)

Setup is done via environment variables (e.g., in `docker-compose.yml` or `.env`):

```env
OAUTH_CLIENT_ID=<Client ID from the IdP>
OAUTH_CLIENT_SECRET=<Secret>
OPENID_PROVIDER_URL=<Discovery URL, e.g. .../.well-known/openid-configuration>
OAUTH_PROVIDER_NAME=<Display name on the login button>
WEBUI_URL=<Base URL of the Open WebUI instance>
```

### 2.2 Advanced SSO Features

| Variable | Function |
|---|---|
| `ENABLE_OAUTH_SIGNUP=true` | Automatically create new users on first SSO login |
| `ENABLE_LOGIN_FORM=false` | Hide the classic login form — only SSO allowed |
| `ENABLE_OAUTH_ROLE_MANAGEMENT=true` | Adopt roles/groups directly from the IdP token |

Additionally, Open WebUI supports **Trusted Headers** for operation behind an authentication proxy (OAuth2-Proxy, Cloudflare Access, etc.).

> [!success] Implementation Status
> Open WebUI is running in production in the K3s cluster (namespace `dgx-ai`) on elite800 with:
> - **PostgreSQL backend** (`xomoxcc/postgreslocaled:latest` with pgvector extension) instead of SQLite — DBs: `openwebui` + `openwebui_vectors`
> - **Admin account + API key** are provisioned idempotently via REST API and stored in the K8s Secret `openwebui-admin-api-key`
> - **Config seeding**: `config.json` is written at every pod start via a busybox initContainer (`RESET_CONFIG_ON_START=true`)
> - **RAG configuration**: pgvector backend, `chunk_size=20000`, `chunk_overlap=750`, hybrid search enabled
> - **Accessible at**: `https://openwebui.dgx.elasticc.io` (cert-manager TLS)

---

## 3. Open WebUI as an API Endpoint

### 3.1 Core Concept: Central AI Gateway

The Open WebUI API is **100% compatible with the OpenAI API**. This makes Open WebUI the central gateway: regardless of whether Ollama (local), Anthropic, OpenAI, or other backends are connected in the background — external systems always communicate only with the Open WebUI API. Open WebUI routes internally to the correct model and logs consumption per user.

**Key advantage for data isolation:** Since every API key is bound to a user, the **exact same permissions and RAG restrictions** that apply in the web interface apply via the API as well. A script using User A's token has no access to User B's data.

### 3.2 API Key and Base URL

- **Generate API key:** Each user under *Settings → Account*
- **Base URL:** `https://<open-webui-domain>/api/v1`

### 3.3 Example: Chat via Python (OpenAI Library)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://chat.deine-domain.de/api/v1",
    api_key="sk-dein-openwebui-api-key"
)

response = client.chat.completions.create(
    model="llama3:latest",
    messages=[
        {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
        {"role": "user", "content": "Hallo! Was kannst du mir über APIs erzählen?"}
    ]
)
print(response.choices[0].message.content)
```

---

## 4. Using Pipelines via the API

### 4.1 Why This Is Crucial

Pipelines are the mechanism through which Open WebUI handles **complex, multi-step processes** — e.g., RAG with web search, agent workflows, or external API calls. The key point: **an API call traverses exactly the same processing path as an input in the web interface.** This means arbitrarily complex workflows can be controlled via the API without the calling application needing to know their logic.

### 4.2 Calling Pipelines as Models

Pipelines are registered in Open WebUI as standalone "models". The API call simply specifies the pipeline ID as the `model`:

```python
response = client.chat.completions.create(
    model="meine-rag-pipeline",  # Pipeline ID instead of model name
    messages=[{"role": "user", "content": "Analysiere den letzten Quartalsbericht."}]
)
```

### 4.3 Global Filter Pipelines

Filter pipelines (e.g., content filtering, system prompt adjustment, logging) are **automatically applied to every API call** — identical to the web interface.

### 4.4 Detailed Flow of an API Call

```
External script → Open WebUI API
   ↓
1. User identification (via API key)
2. Permission check (model/pipeline access)
3. Filter pipelines (Inlet)
4. Pipeline processing (RAG, web search, tool calls …)
5. Filter pipelines (Outlet)
   ↓
Result → returned to the script
```

---

## 5. RAG via API: Isolated Knowledge Bases

### 5.1 Using a Knowledge Base with a Request

The desired knowledge base is specified via the `extra_body` parameter:

```python
response = client.chat.completions.create(
    model="llama3:latest",
    messages=[
        {"role": "user", "content": "Was sind die wichtigsten Erkenntnisse aus dem Dokument?"}
    ],
    extra_body={
        "files": [
            {"type": "collection", "id": "<knowledge-base-id>"}
        ]
    }
)
```

### 5.2 Building a Knowledge Base Programmatically

With the `owui-client` library, knowledge bases can be created and populated automatically:

```python
import asyncio
from owui_client import OpenWebUI

async def main():
    client = OpenWebUI(
        api_url="https://chat.deine-domain.de/api",
        api_key="sk-dein-api-key-von-user-a"
    )

    # Upload file
    file = await client.files.upload("dokument.pdf")

    # Create knowledge base
    kb = await client.knowledge.create(
        name="Projekt RAG User A",
        description="Isolierte Daten für User A"
    )

    # Associate file
    await client.knowledge.add_file(knowledge_id=kb.id, file_id=file.id)

asyncio.run(main())
```

> **Note:** Since the API key is bound to User A, the knowledge base automatically belongs to User A and is protected from other users.

### 5.3 Reading Citations

Open WebUI includes the sources used as metadata in the API response:

```python
message = response.choices[0].message
message_dict = message.model_dump()

citations = message_dict.get("citations", [])
if not citations and message.model_extra:
    citations = message.model_extra.get("citations", [])

for citation in citations:
    source = citation.get("source", {}).get("name", "Unbekannt")
    snippet = citation.get("document", [""])[0][:100]
    print(f"Quelle: {source} – Auszug: '{snippet}...'")
```

---

## 6. Web Search via API

Prerequisite: A search engine (DuckDuckGo, SearXNG, Google, Tavily) must be configured in the admin panel.

```python
response = client.chat.completions.create(
    model="llama3:latest",
    messages=[
        {"role": "user", "content": "Was sind die neuesten Nachrichten zu KI-Regulierung?"}
    ],
    extra_body={
        "features": {"web_search": True}
    }
)
```

**Combining with RAG** – web search and knowledge base can be combined in a single call:

```python
extra_body={
    "features": {"web_search": True},
    "files": [{"type": "collection", "id": "<kb-id>"}]
}
```

---

## 7. Streaming & Chat History

### 7.1 Streaming (Word-by-Word Output)

```python
response = client.chat.completions.create(
    model="llama3:latest",
    messages=[...],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
```

### 7.2 Chat History (Conversation Memory)

The API is stateless — the entire conversation history must be sent with every request:

```python
chat_history = [
    {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
    {"role": "user", "content": "Mein Projekt heißt Apollo."},
    {"role": "assistant", "content": "Verstanden, ich merke mir den Projektnamen Apollo."},
    {"role": "user", "content": "Wie heißt mein Projekt?"}  # AI can answer correctly
]

response = client.chat.completions.create(
    model="llama3:latest",
    messages=chat_history
)
```

---

## 8. Function Calling (Tool Use)

Open WebUI supports function calling via the API, allowing the model to trigger **external actions** (e.g., smart home control, database queries, API calls).

```python
tools = [{
    "type": "function",
    "function": {
        "name": "schalte_licht",
        "description": "Schaltet das Licht in einem Raum an oder aus.",
        "parameters": {
            "type": "object",
            "properties": {
                "raum": {"type": "string", "description": "Name des Raums"},
                "status": {"type": "string", "enum": ["an", "aus"]}
            },
            "required": ["raum", "status"]
        }
    }
}]

response = client.chat.completions.create(
    model="llama3.1:latest",
    messages=[{"role": "user", "content": "Mach das Licht im Wohnzimmer an!"}],
    tools=tools,
    tool_choice="auto"
)
```

The model responds with a structured tool call instead of text. The calling script executes the function locally and reports the result back via a `role: tool` message, after which the model formulates a final response.

> **Note:** Function calling requires a model that supports it (e.g., Llama 3.1+).

---

## 9. Python Libraries Overview

| Library | Purpose | Installation |
|---|---|---|
| `openai` | Chat, streaming, RAG queries, function calling | `pip install openai` |
| `owui-client` | Admin tasks, file upload, knowledge base management, user management | `pip install openwebui-client` |

> **Tip:** The Swagger documentation of your own instance is available at `https://<domain>/docs` and reflects the complete live API.

---

## 10. SearXNG Web Search Integration in Open WebUI

Prerequisite: A local SearXNG instance is available and configured for JSON output (see [[DGX Spark/DGX Spark Setup#5. SearXNG Web Search Integration|main document, section 5]]).

### 10.1 Open WebUI Environment Variables

In the K8s configuration (ConfigMap or Deployment env):

```yaml
environment:
  - ENABLE_RAG_WEB_SEARCH=True
  - RAG_WEB_SEARCH_ENGINE=searxng
  - RAG_WEB_SEARCH_RESULT_COUNT=5
  - RAG_WEB_SEARCH_CONCURRENT_REQUESTS=10
  - SEARXNG_QUERY_URL=http://searxng.searxng.svc:8080/search?q=<query>
```

> [!success] Implemented SearXNG Configuration
> SearXNG runs in its own namespace `searxng` on elite800:
> - **Enabled engines**: Google, Bing, Wikipedia, Wikidata, Startpage, Mojeek, Qwant
> - **Disabled engines**: DuckDuckGo, Brave
> - **Backend**: Redis (`redis.redis.svc:6379/3`)
> - **Limiter**: disabled (private instance)
> - **Timeouts**: `request_timeout: 10.0`
> - **K8s service**: `searxng.searxng.svc:8080`

### 10.2 Enabling in Chat

In the Open WebUI chat, an integration button appears next to the "+" icon. Web search can be enabled there via toggle — per session.

### 10.3 Automatic Web Search via Native Function Calling

Open WebUI supports "Native Function Calling", where the LLM itself decides when it needs a web search. For this:

1. Admin Settings → Functions → Enable Web Search as a tool

2. Enable "Native Function Calling" in the model settings


This allows the model to autonomously perform multiple searches in sequence and also read complete web pages using the `fetch_url` tool — similar to Claude's Web Search.

**Note**: Native function calling requires the LLM to support tool use. Qwen3-235B and most current models support this.

---

## 11. Self-Reflection Pipeline (Filter)

An Open WebUI filter pipeline that automatically sends the output back for self-critique:

```python
"""
title: Self-Reflection Filter
description: Sends LLM output back for self-critique, then returns refined answer.
version: 1.0
"""

from typing import List, Optional
from pydantic import BaseModel
import aiohttp
import json

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        reflection_enabled: bool = True
        api_base_url: str = "http://<IP-SPARK-1>:8000/v1"

    def __init__(self):
        self.type = "filter"
        self.name = "Self-Reflection"
        self.valves = self.Valves()

    async def outlet(self, body: dict, user: dict) -> dict:
        """After the LLM response: self-critique loop."""
        if not self.valves.reflection_enabled:
            return body

        messages = body.get("messages", [])
        if not messages or messages[-1]["role"] != "assistant":
            return body

        original_answer = messages[-1]["content"]
        user_question = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_question = msg["content"]
                break

        # Reflection prompt
        reflection_prompt = f"""Überprüfe die folgende Antwort kritisch:

FRAGE: {user_question}

ANTWORT: {original_answer}

Prüfe auf:
1. Faktenfehler oder unbelegte Behauptungen
2. Logische Inkonsistenzen
3. Fehlende wichtige Aspekte
4. Verzerrungen oder einseitige Darstellung

Wenn die Antwort gut ist, bestätige das kurz.
Wenn Verbesserungen nötig sind, liefere eine korrigierte Version.

Beginne mit: [REFLEXION] oder [BESTÄTIGT]"""

        # Send back to the same model
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.valves.api_base_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": reflection_prompt}],
                    "max_tokens": 4096,
                    "temperature": 0.3
                },
                headers={"Content-Type": "application/json"}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    reflection = data["choices"][0]["message"]["content"]

                    if reflection.startswith("[REFLEXION]"):
                        # Answer was improved - replace it
                        messages[-1]["content"] = (
                            original_answer +
                            "\n\n---\n*Nach Selbstreflexion überarbeitet:*\n\n" +
                            reflection.replace("[REFLEXION]", "").strip()
                        )
                    else:
                        # Answer was good - add confirmation
                        messages[-1]["content"] = (
                            original_answer +
                            "\n\n---\n✅ *Selbstprüfung bestanden.*"
                        )

        body["messages"] = messages
        return body
```

**Tradeoff**: Doubles the inference time per response. At ~25 t/s on Qwen3-235B via SGLang, this means: a 500-token response takes ~20s → ~40s total. Entirely acceptable for business analysis quality. Thanks to SGLang's RadixAttention, the reflection call is even accelerated because the context of the original answer is already in the prefix cache.

### Best-of-N with Voting (for critical analyses)

For particularly important questions: generate N responses (e.g., 3) with slightly different temperatures, then have the model select or synthesize the best one. This works via a Manifold pipeline:

```python
# Concept (simplified):
responses = []
for temp in [0.3, 0.6, 0.9]:
    response = await generate(prompt, temperature=temp)
    responses.append(response)

synthesis = await generate(
    f"Hier sind 3 Analysen. Synthetisiere die beste Antwort:\n"
    f"1: {responses[0]}\n2: {responses[1]}\n3: {responses[2]}"
)
```

**Tradeoff**: Quadruples inference time. Only worthwhile for the most important queries.

---

## 12. Embedding Integration with Open WebUI

### Implemented: Ollama on DGX Spark (GPU-accelerated)

> [!success] Production Configuration
> Embedding runs via **Ollama** on spark1 (namespace `ollama`), GPU-accelerated via time-slicing:
> - K8s service: `ollama.ollama.svc:11434`
> - Model: `bge-m3` (preloaded via initContainer, warmup via postStart hook)
> - Open WebUI env: `RAG_EMBEDDING_ENGINE=ollama`, `RAG_EMBEDDING_MODEL=bge-m3`, `RAG_OLLAMA_BASE_URL=http://ollama.ollama.svc:11434`

### Alternative: Via External Embedding Service (TEI / OVMS)

In Open WebUI Admin Settings → Documents:
- Embedding Engine: **OpenAI**
- API Base URL: `http://<ELITE800-IP>:8001/v3`
- Model: `bge-m3`

### Alternative: Via Integrated Engine (no separate service)

Open WebUI has a built-in embedding engine. In Admin Settings → Documents, you can select "Default (SentenceTransformers)". This loads bge-m3 directly in the Open WebUI container on CPU. No separate service required.

> For the various embedding options and their pros/cons see [[DGX Spark/DGX Spark Setup#4. Embedding-Modell: Ollama auf DGX Spark (GPU-beschleunigt)|main document, section 4]].

---

## Summary: Why Open WebUI?

Open WebUI combines the three central requirements of our setup in a single platform:

| Requirement | Open WebUI Solution |
|---|---|
| **Authentication & SSO** | Native OIDC/OAuth 2.0, trusted headers, automatic user provisioning, roles from IdP |
| **Data isolation & permissions** | RBAC with groups, private knowledge bases, API keys bound to users → full multi-tenancy |
| **API-driven workflows** | OpenAI-compatible API as a universal endpoint; pipelines as orchestrable models; filters, RAG, web search, and function calling controllable via the same API call |

Through this architecture, Open WebUI becomes the **central orchestration point**: the web interface serves as an interactive access point for end users, while the API makes it possible to integrate the same functionality — including all pipeline logic — into external systems, automations, and custom frontends.
