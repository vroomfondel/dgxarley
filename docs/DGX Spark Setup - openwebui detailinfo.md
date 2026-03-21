# Open WebUI – Architektur, Datentrennung & API-Integration

## Warum Open WebUI als zentrale Plattform?

Open WebUI ist weit mehr als eine Chat-Oberfläche für LLMs. Es hat sich als **De-facto-Standard** für selbst gehostete AI-Interfaces etabliert und fungiert als **zentrales AI-Gateway** mit Enterprise-tauglichen Eigenschaften.

### Reife & Verbreitung

Open WebUI ist eines der am schnellsten wachsenden Open-Source-Projekte im AI-Bereich:

- **125.000+ GitHub Stars** und **17.700+ Forks** – damit gehört es zu den populärsten Open-Source-Projekten überhaupt
- **341.000+ Community-Mitglieder** und **290+ Millionen Downloads**
- **Aktive Weiterentwicklung** – regelmäßige Releases (aktuell v0.8.x), mit einem großen Maintainer-Team und einer lebendigen Contributor-Community
- **Breite Adoption** – vom Einzelentwickler über Forschungseinrichtungen bis hin zu Unternehmen im produktiven Einsatz

Diese Verbreitung bedeutet: umfangreiche Dokumentation, schnelle Bugfixes, ein großes Ökosystem an Plugins/Pipelines und die Sicherheit, nicht auf ein Nischenprojekt zu setzen. Open WebUI ist keine experimentelle Software mehr, sondern eine **ausgereifte, produktionsreife Plattform**.

### Kernvorteile für unseren Anwendungsfall

1. **Single Sign-On (SSO)** – Native OIDC/OAuth 2.0-Unterstützung. Nutzer authentifizieren sich über den zentralen Identity Provider; eine separate Passwortverwaltung entfällt.
2. **Strikte Datentrennung pro User** – Jeder Nutzer sieht ausschließlich seine eigenen Dokumente und Knowledge Bases. Rollenbasierte Zugriffssteuerung (RBAC) erlaubt feingranulare Kontrolle bis auf Gruppenebene.
3. **OpenAI-kompatible API** – Open WebUI stellt eine vollständige REST-API bereit, die zum OpenAI-Standard kompatibel ist. Damit wird es zum **universellen API-Endpoint**, über den externe Systeme, Skripte und Automatisierungen sämtliche Features nutzen können – inklusive Pipelines.
4. **Pipelines als orchestrierbare Workflows** – Komplexe Abläufe (RAG, Web-Suche, Agenten, externe API-Aufrufe) werden als Pipelines definiert und sind sowohl über die Weboberfläche als auch über die API identisch nutzbar.

---

## 1. Datentrennung & Berechtigungssystem (RBAC)

### 1.1 Dokumenten-Upload im Chat (standardmäßig privat)

Wenn ein Nutzer ein Dokument direkt in einen Chat zieht, ist es **ausschließlich für diesen Nutzer in diesem Chat** zugänglich. Andere Nutzer können die Datei weder sehen noch durchsuchen.

### 1.2 Zentrale Knowledge Bases (Workspace)

Unter **Workspace → Knowledge** können Dokumentensammlungen systematisch verwaltet werden:

| Einstellung | Beschreibung |
|---|---|
| **Visibility (Sichtbarkeit)** | Jede Knowledge Base kann auf *Privat* gesetzt werden – nur der Ersteller hat Zugriff. |
| **Gruppenbasierter Zugriff** | Admins erstellen Gruppen (z. B. „Team A") und erteilen Knowledge Bases gezielt Lese-/Schreibrechte für bestimmte Gruppen. |

### 1.3 RBAC-Konfiguration (Admin)

Über **Admin → Users → Groups → Default Permissions** lassen sich folgende Einschränkungen vornehmen:

- **„Public Knowledge" verbieten** – Nutzer können Knowledge Bases nicht auf *öffentlich* stellen.
- **Workspace-Zugriff sperren** – Nutzer können nur temporäre Chat-Dokumente nutzen, aber keine persistenten Wissensdatenbanken anlegen.

> **Fazit:** Solange Dokumente nicht ausdrücklich über Admin oder Gruppenfreigabe geteilt werden, herrscht **strikte Mandantenfähigkeit** zwischen den Nutzern.

---

## 2. Single Sign-On (SSO) via OAuth 2.0 / OIDC

Open WebUI bietet native SSO-Unterstützung über den generischen OIDC-Standard. Dadurch funktioniert es mit jedem gängigen Identity Provider:

- Google Workspace / Google Cloud
- Microsoft Entra ID (ehemals Azure AD)
- Keycloak
- Authentik, Authelia
- Okta, GitHub

### 2.1 Konfiguration (Umgebungsvariablen)

Die Einrichtung erfolgt über Environment Variables (z. B. in `docker-compose.yml` oder `.env`):

```env
OAUTH_CLIENT_ID=<Client-ID aus dem IdP>
OAUTH_CLIENT_SECRET=<Secret>
OPENID_PROVIDER_URL=<Discovery-URL, z.B. .../.well-known/openid-configuration>
OAUTH_PROVIDER_NAME=<Anzeigename auf dem Login-Button>
WEBUI_URL=<Basis-URL der Open-WebUI-Instanz>
```

### 2.2 Erweiterte SSO-Features

| Variable | Funktion |
|---|---|
| `ENABLE_OAUTH_SIGNUP=true` | Automatisches Anlegen neuer Nutzer bei erster SSO-Anmeldung |
| `ENABLE_LOGIN_FORM=false` | Klassisches Login-Formular ausblenden – nur SSO erlaubt |
| `ENABLE_OAUTH_ROLE_MANAGEMENT=true` | Rollen/Gruppen direkt aus dem IdP-Token übernehmen |

Zusätzlich unterstützt Open WebUI **Trusted Headers** für den Betrieb hinter einem Authentifizierungs-Proxy (OAuth2-Proxy, Cloudflare Access etc.).

> [!success] Implementierungsstatus
> Open WebUI läuft produktiv im K3s-Cluster (Namespace `dgx-ai`) auf k3smaster mit:
> - **PostgreSQL-Backend** (`xomoxcc/postgreslocaled:latest` mit pgvector-Extension) statt SQLite — DBs: `openwebui` + `openwebui_vectors`
> - **Admin-Account + API-Key** werden idempotent via REST-API provisioniert und im K8s-Secret `openwebui-admin-api-key` gespeichert
> - **Config-Seeding**: `config.json` wird bei jedem Pod-Start via busybox-initContainer geschrieben (`RESET_CONFIG_ON_START=true`)
> - **RAG-Konfiguration**: pgvector-Backend, `chunk_size=20000`, `chunk_overlap=750`, Hybrid Search aktiviert
> - **Erreichbar unter**: `https://openwebui.dgx.example.com` (cert-manager TLS)

---

## 3. Open WebUI als API-Endpoint

### 3.1 Kernkonzept: Zentrales AI-Gateway

Die API von Open WebUI ist **zu 100 % kompatibel zur OpenAI-API**. Damit wird Open WebUI zum zentralen Gateway: Egal ob im Hintergrund Ollama (lokal), Anthropic, OpenAI oder andere Backends angebunden sind – externe Systeme kommunizieren immer nur mit der Open-WebUI-API. Open WebUI routet intern an das richtige Modell und protokolliert den Verbrauch pro User.

**Entscheidender Vorteil für die Datentrennung:** Da jeder API-Key an einen User gebunden ist, greifen über die API **exakt dieselben Berechtigungen und RAG-Beschränkungen** wie über die Weboberfläche. Ein Skript mit dem Token von User A hat keinen Zugriff auf die Daten von User B.

### 3.2 API-Key und Base-URL

- **API-Key generieren:** Jeder Nutzer unter *Settings → Account*
- **Base-URL:** `https://<open-webui-domain>/api/v1`

### 3.3 Beispiel: Chat via Python (OpenAI-Bibliothek)

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

## 4. Pipelines über die API nutzen

### 4.1 Warum das entscheidend ist

Pipelines sind der Mechanismus, mit dem Open WebUI **komplexe, mehrstufige Abläufe** abbildet – z. B. RAG mit Web-Suche, Agenten-Workflows oder externe API-Aufrufe. Das Besondere: **Ein API-Call durchläuft exakt denselben Verarbeitungspfad wie eine Eingabe in der Weboberfläche.** Damit lassen sich über die API beliebig komplexe Workflows steuern, ohne dass die aufrufende Applikation deren Logik kennen muss.

### 4.2 Pipelines als Modelle aufrufen

Pipelines werden in Open WebUI als eigenständige „Modelle" registriert. Beim API-Call wird einfach die Pipeline-ID als `model` angegeben:

```python
response = client.chat.completions.create(
    model="meine-rag-pipeline",  # Pipeline-ID statt Modellname
    messages=[{"role": "user", "content": "Analysiere den letzten Quartalsbericht."}]
)
```

### 4.3 Globale Filter-Pipelines

Filter-Pipelines (z. B. Content-Filterung, System-Prompt-Anpassung, Logging) werden **automatisch auf jeden API-Call angewendet** – identisch zur Weboberfläche.

### 4.4 Ablauf eines API-Calls im Detail

```
Externes Skript → Open WebUI API
   ↓
1. User-Identifikation (via API-Key)
2. Berechtigungsprüfung (Modell-/Pipeline-Zugriff)
3. Filter-Pipelines (Inlet)
4. Pipeline-Verarbeitung (RAG, Web-Suche, Tool-Calls …)
5. Filter-Pipelines (Outlet)
   ↓
Ergebnis → zurück an das Skript
```

---

## 5. RAG per API: Isolierte Knowledge Bases

### 5.1 Knowledge Base mit einer Anfrage nutzen

Über den `extra_body`-Parameter wird die gewünschte Knowledge Base angegeben:

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

### 5.2 Knowledge Base programmatisch aufbauen

Mit der `owui-client`-Bibliothek lassen sich Knowledge Bases automatisiert erstellen und befüllen:

```python
import asyncio
from owui_client import OpenWebUI

async def main():
    client = OpenWebUI(
        api_url="https://chat.deine-domain.de/api",
        api_key="sk-dein-api-key-von-user-a"
    )

    # Datei hochladen
    file = await client.files.upload("dokument.pdf")

    # Knowledge Base erstellen
    kb = await client.knowledge.create(
        name="Projekt RAG User A",
        description="Isolierte Daten für User A"
    )

    # Datei zuordnen
    await client.knowledge.add_file(knowledge_id=kb.id, file_id=file.id)

asyncio.run(main())
```

> **Hinweis:** Da der API-Key an User A gebunden ist, gehört die Knowledge Base automatisch User A und ist vor anderen Nutzern geschützt.

### 5.3 Quellen (Citations) auslesen

Open WebUI liefert in der API-Antwort die verwendeten Quellen als Metadaten mit:

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

## 6. Web-Suche per API

Voraussetzung: Im Admin-Panel muss eine Suchmaschine (DuckDuckGo, SearXNG, Google, Tavily) konfiguriert sein.

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

**Kombination mit RAG** – Web-Suche und Knowledge Base lassen sich in einem Call verbinden:

```python
extra_body={
    "features": {"web_search": True},
    "files": [{"type": "collection", "id": "<kb-id>"}]
}
```

---

## 7. Streaming & Chat-History

### 7.1 Streaming (Wort-für-Wort-Ausgabe)

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

### 7.2 Chat-History (Konversationsgedächtnis)

Die API ist zustandslos – der gesamte bisherige Verlauf muss bei jeder Anfrage mitgeschickt werden:

```python
chat_history = [
    {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
    {"role": "user", "content": "Mein Projekt heißt Apollo."},
    {"role": "assistant", "content": "Verstanden, ich merke mir den Projektnamen Apollo."},
    {"role": "user", "content": "Wie heißt mein Projekt?"}  # KI kann korrekt antworten
]

response = client.chat.completions.create(
    model="llama3:latest",
    messages=chat_history
)
```

---

## 8. Function Calling (Tool Use)

Open WebUI unterstützt Function Calling über die API, wodurch das Modell **externe Aktionen** auslösen kann (z. B. Smart-Home-Steuerung, Datenbank-Abfragen, API-Aufrufe).

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

Das Modell antwortet mit einem strukturierten Tool-Call statt mit Text. Das aufrufende Skript führt die Funktion lokal aus und meldet das Ergebnis über eine `role: tool`-Nachricht zurück, woraufhin das Modell eine finale Antwort formuliert.

> **Hinweis:** Function Calling erfordert ein Modell, das dies unterstützt (z. B. Llama 3.1+).

---

## 9. Python-Bibliotheken im Überblick

| Bibliothek | Zweck | Installation |
|---|---|---|
| `openai` | Chat, Streaming, RAG-Queries, Function Calling | `pip install openai` |
| `owui-client` | Admin-Aufgaben, Datei-Upload, Knowledge-Base-Management, User-Verwaltung | `pip install openwebui-client` |

> **Tipp:** Die Swagger-Dokumentation der eigenen Instanz ist unter `https://<domain>/docs` erreichbar und bildet die komplette API live ab.

---

## 10. SearXNG Web-Search-Integration in Open WebUI

Voraussetzung: Eine lokale SearXNG-Instanz ist vorhanden und für JSON-Ausgabe konfiguriert (siehe [Hauptdokument, Abschnitt 5](DGX%20Spark%20Setup.md#5-searxng-web-search-integration)).

### 10.1 Open WebUI Environment Variables

In der K8s-Konfiguration (ConfigMap oder Deployment-Env):

```yaml
environment:
  - ENABLE_RAG_WEB_SEARCH=True
  - RAG_WEB_SEARCH_ENGINE=searxng
  - RAG_WEB_SEARCH_RESULT_COUNT=5
  - RAG_WEB_SEARCH_CONCURRENT_REQUESTS=10
  - SEARXNG_QUERY_URL=http://searxng.searxng.svc:8080/search?q=<query>
```

> [!success] Implementierte SearXNG-Konfiguration
> SearXNG läuft im eigenen Namespace `searxng` auf k3smaster:
> - **Aktivierte Engines**: Google, Bing, Wikipedia, Wikidata, Startpage, Mojeek, Qwant
> - **Deaktivierte Engines**: DuckDuckGo, Brave
> - **Backend**: Redis (`redis.redis.svc:6379/3`)
> - **Limiter**: deaktiviert (private Instanz)
> - **Timeouts**: `request_timeout: 10.0`
> - **K8s-Service**: `searxng.searxng.svc:8080`

### 10.2 Im Chat aktivieren

Im Open-WebUI-Chat erscheint neben dem "+"-Icon ein Integrations-Button. Dort kann Web Search per Toggle aktiviert werden — pro Session.

### 10.3 Automatische Web-Suche via Native Function Calling

Open WebUI unterstützt "Native Function Calling", wobei das LLM selbst entscheidet, wann es eine Web-Suche braucht. Dafür:

1. Admin Settings → Functions → Web Search als Tool aktivieren

2. In den Model-Einstellungen "Native Function Calling" aktivieren


Damit kann das Modell autonom mehrere Suchen hintereinander durchführen und mit dem `fetch_url`-Tool auch vollständige Webseiten lesen — ähnlich wie Claude's Web Search.

**Hinweis**: Native Function Calling erfordert, dass das LLM Tool-Use beherrscht. Qwen3-235B und die meisten aktuellen Modelle können das.

---

## 11. Self-Reflection Pipeline (Filter)

Eine Open WebUI Filter-Pipeline, die den Output automatisch zur Selbstkritik zurückschickt:

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
        """Nach der LLM-Antwort: Selbstkritik-Loop."""
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

        # Reflexions-Prompt
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

        # Sende an das gleiche Modell zurück
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
                        # Antwort wurde verbessert - ersetze sie
                        messages[-1]["content"] = (
                            original_answer +
                            "\n\n---\n*Nach Selbstreflexion überarbeitet:*\n\n" +
                            reflection.replace("[REFLEXION]", "").strip()
                        )
                    else:
                        # Antwort war gut - füge Bestätigung hinzu
                        messages[-1]["content"] = (
                            original_answer +
                            "\n\n---\n✅ *Selbstprüfung bestanden.*"
                        )

        body["messages"] = messages
        return body
```

**Tradeoff**: Verdoppelt die Inferenz-Zeit pro Antwort. Bei ~25 t/s auf Qwen3-235B via SGLang heißt das: eine 500-Token-Antwort braucht ~20s → ~40s gesamt. Für Business-Analyse-Qualität absolut akzeptabel. Durch SGLangs RadixAttention wird der Reflexions-Call sogar beschleunigt, weil der Kontext der Originalantwort bereits im Prefix-Cache liegt.

### Best-of-N mit Voting (für kritische Analysen)

Für besonders wichtige Fragen: Generiere N Antworten (z.B. 3) mit leicht unterschiedlicher Temperature, lasse das Modell dann die beste auswählen oder synthetisieren. Das geht über eine Manifold-Pipeline:

```python
# Konzept (vereinfacht):
responses = []
for temp in [0.3, 0.6, 0.9]:
    response = await generate(prompt, temperature=temp)
    responses.append(response)

synthesis = await generate(
    f"Hier sind 3 Analysen. Synthetisiere die beste Antwort:\n"
    f"1: {responses[0]}\n2: {responses[1]}\n3: {responses[2]}"
)
```

**Tradeoff**: Vervierfacht die Inferenz-Zeit. Nur für die wichtigsten Anfragen sinnvoll.

---

## 12. Embedding-Anbindung an Open WebUI

### Implementiert: Ollama auf DGX Spark (GPU-beschleunigt)

> [!success] Produktive Konfiguration
> Embedding läuft über **Ollama** auf spark1 (Namespace `ollama`), GPU-beschleunigt via Time-Slicing:
> - K8s-Service: `ollama.ollama.svc:11434`
> - Modell: `bge-m3` (vorgeladen via initContainer, warmup via postStart-Hook)
> - Open WebUI Env: `RAG_EMBEDDING_ENGINE=ollama`, `RAG_EMBEDDING_MODEL=bge-m3`, `RAG_OLLAMA_BASE_URL=http://ollama.ollama.svc:11434`

### Alternative: Via externen Embedding-Service (TEI / OVMS)

In Open WebUI Admin Settings → Documents:
- Embedding Engine: **OpenAI**
- API Base URL: `http://<k3smaster-IP>:8001/v3`
- Model: `bge-m3`

### Alternative: Via integrierte Engine (kein separater Service)

Open WebUI hat eine eingebaute Embedding-Engine. In den Admin Settings → Documents kann man "Default (SentenceTransformers)" wählen. Das lädt bge-m3 direkt im Open-WebUI-Container auf CPU. Kein separater Service nötig.

> Für die verschiedenen Embedding-Optionen und ihre Vor-/Nachteile siehe [Hauptdokument, Abschnitt 4](DGX%20Spark%20Setup.md#4-embedding-modell-ollama-auf-dgx-spark-gpu-beschleunigt).

---

## Zusammenfassung: Warum Open WebUI?

Open WebUI vereint die drei zentralen Anforderungen unseres Setups in einer Plattform:

| Anforderung | Open-WebUI-Lösung |
|---|---|
| **Authentifizierung & SSO** | Native OIDC/OAuth 2.0, Trusted Headers, automatische User-Provisionierung, Rollen aus IdP |
| **Datentrennung & Berechtigungen** | RBAC mit Gruppen, private Knowledge Bases, API-Keys an User gebunden → volle Mandantenfähigkeit |
| **API-gesteuerte Workflows** | OpenAI-kompatible API als universeller Endpoint; Pipelines als orchestrierbare Modelle; Filter, RAG, Web-Suche und Function Calling über denselben API-Call steuerbar |

Durch diese Architektur wird Open WebUI zum **zentralen Orchestrierungspunkt**: Die Weboberfläche dient als interaktiver Zugang für Endnutzer, während die API es ermöglicht, dieselben Funktionen – einschließlich aller Pipeline-Logik – in externe Systeme, Automatisierungen und eigene Frontends einzubinden.
