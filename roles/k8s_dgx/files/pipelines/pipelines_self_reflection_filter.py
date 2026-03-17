"""
title: Self-Reflection Filter
author: dgxarley
version: 1.1
description: Sends LLM output back for self-critique, appends refined answer.
"""

import os
from pprint import pprint
from typing import List, Optional
from pydantic import BaseModel, Field
import aiohttp

# Read SGLang endpoint from environment (set via pipelines-config ConfigMap).
# Fallback to a sensible cluster-internal default.
_DEFAULT_SGLANG_URL: str = os.environ.get(
    "SGLANG_API_BASE_URL", "http://localhost:8000/v1"
)

_DEFAULT_SELF_REFLECTION_ENABLED: bool = os.environ.get(
    "SELF_REFLECTION_ENABLED", "false"
).lower() in ("true", "1", "yes")


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = ["*"]
        priority: int = 0
        reflection_enabled: bool = Field(
            default=_DEFAULT_SELF_REFLECTION_ENABLED,
            description="Enable self-reflection quality check on LLM responses (from SELF_REFLECTION_ENABLED env)",
        )
        api_base_url: str = Field(
            default=_DEFAULT_SGLANG_URL,
            description="SGLang OpenAI-compatible API endpoint (from SGLANG_API_BASE_URL env)",
        )
        max_tokens: int = Field(default=4096)
        temperature: float = Field(default=0.3)
        include_snippets: bool = Field(
            default=True,
            description="Include document text snippets from web search results in the reflection prompt",
        )
        snippet_max_chars: int = Field(
            default=500,
            description="Maximum characters per document snippet",
        )
        min_relevance_score: float = Field(
            default=0.0,
            description="Minimum relevance score for web search sources (0.0 = include all)",
        )

    # --- Optional: UserValves ---
    # Wenn diese Klasse einkommentiert wird, kann jeder User den Filter
    # individuell in seinen OpenWebUI-Settings (Pipelines-Tab) deaktivieren.
    # Valves (oben) bleiben admin-only; UserValves erscheinen pro User.
    # In outlet() muss dann die User-Valve ausgelesen werden, z.B.:
    #     user_valves = user.get("valves", {}) if user else {}
    #     if not user_valves.get("reflection_enabled", self.valves.reflection_enabled):
    #         return body
    #
    # class UserValves(BaseModel):
    #     reflection_enabled: bool = Field(
    #         default=True,
    #         description="Self-Reflection für deine Antworten aktivieren",
    #     )

    def __init__(self):
        self.type = "filter"
        self.name = "Self-Reflection"
        self.valves = self.Valves()
        # Per-chat flags set in inlet(), consumed in outlet().
        # Keyed by chat_id, cleaned up on read via pop().
        self._inlet_flags: dict[str, dict] = {}

    async def on_startup(self):
        print(f"Self-Reflection Filter started, target: {self.valves.api_base_url}")

    async def on_shutdown(self):
        print("Self-Reflection Filter stopped")

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if not self.valves.reflection_enabled:
            return body

        # Remember per-chat flags for outlet (web_search, etc.)
        chat_id = body.get("chat_id", "")
        features = body.get("features", {})
        if chat_id:
            self._inlet_flags[chat_id] = {
                "web_search": features.get("web_search", False),
            }
            print(f"[inlet] chat_id={chat_id[:12]}... web_search={features.get('web_search', False)}")

        # Check last user message for /no_think or /think prefix.
        # If found, strip it and inject chat_template_kwargs into the request body
        # so SGLang toggles thinking mode accordingly.
        messages = body.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                if content.startswith("/no_think"):
                    msg["content"] = content[len("/no_think"):].lstrip("\n ")
                    body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
                    print(f"[inlet] /no_think detected — enable_thinking=False, stripped msg: {msg['content'][:80]}")
                elif content.startswith("/think"):
                    msg["content"] = content[len("/think"):].lstrip("\n ")
                    body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True
                    print(f"[inlet] /think detected — enable_thinking=True, stripped msg: {msg['content'][:80]}")
                else:
                    print(f"[inlet] no thinking prefix, passthrough msg: {content[:80]}")
            break
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        if not self.valves.reflection_enabled:
            return body

        # Retrieve inlet flags for this chat (pop to clean up)
        chat_id = body.get("chat_id", "")
        inlet_flags = self._inlet_flags.pop(chat_id, {})
        had_web_search = inlet_flags.get("web_search", False)
        print(f"[outlet] chat_id={chat_id[:12]}... had_web_search={had_web_search}")

        messages = body.get("messages", [])
        if not messages or messages[-1].get("role") != "assistant":
            return body

        original_answer = messages[-1]["content"]

        user_question = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_question = msg["content"]
                break

        if not user_question:
            return body

        pprint(body, indent=4, width=120)

        try:
            # Extract web search citations/sources.
            # OpenWebUI stores them in multiple possible locations:
            # - body["sources"] or body["citations"] (top-level)
            # - messages[-1]["sources"] (on the assistant message)
            # - messages[-1]["sources"][i]["document"] (nested documents)
            sources_text = ""
            last_msg = messages[-1] if messages else {}
            if isinstance(last_msg, str):
                last_msg = {}
            sources = (
                body.get("sources")
                or body.get("citations")
                or last_msg.get("sources")
                or []
            )
            # Parse OpenWebUI web_search source structure into XML:
            # sources[i]["metadata"] = list of {title, description, source (url), score}
            # sources[i]["document"] = list of full extracted page texts (parallel to metadata)
            # sources[i]["source"]["queries"] = search queries used
            min_score = self.valves.min_relevance_score
            include_snippets = self.valves.include_snippets
            snippet_max = self.valves.snippet_max_chars
            source_count = 0
            skipped_count = 0
            xml_parts = ['<web_search_context>']
            for src in sources:
                if not isinstance(src, dict):
                    continue
                src_info = src.get("source", {})
                if isinstance(src_info, dict):
                    queries = src_info.get("queries", [])
                    if queries:
                        xml_parts.append('  <search_queries>')
                        for q in queries:
                            if isinstance(q, str):
                                xml_parts.append(f'    <query>{q}</query>')
                        xml_parts.append('  </search_queries>')
                metadata_list = src.get("metadata", [])
                if not isinstance(metadata_list, list):
                    metadata_list = []
                # Document texts parallel to metadata entries
                doc_list = src.get("document", [])
                if not isinstance(doc_list, list):
                    doc_list = []
                if metadata_list:
                    xml_parts.append('  <sources>')
                    for idx, meta in enumerate(metadata_list):
                        if not isinstance(meta, dict):
                            continue
                        title = meta.get("title", "")
                        desc = meta.get("description", "")
                        url = meta.get("source", "")
                        score = meta.get("score", 0)
                        lang = meta.get("language", "")
                        # Relevance filter
                        if min_score > 0 and score < min_score:
                            skipped_count += 1
                            continue
                        if title or url:
                            xml_parts.append(f'    <source relevance="{score:.2f}" language="{lang}">')
                            if title:
                                xml_parts.append(f'      <title>{title}</title>')
                            if desc:
                                xml_parts.append(f'      <description>{desc}</description>')
                            if url:
                                xml_parts.append(f'      <url>{url}</url>')
                            # Document snippet (parallel index into doc_list)
                            if include_snippets and idx < len(doc_list):
                                raw_doc = doc_list[idx]
                                if isinstance(raw_doc, str) and raw_doc.strip():
                                    snippet = raw_doc.strip()[:snippet_max]
                                    if len(raw_doc.strip()) > snippet_max:
                                        snippet += "..."
                                    xml_parts.append(f'      <snippet>{snippet}</snippet>')
                            xml_parts.append('    </source>')
                            source_count += 1
                    xml_parts.append('  </sources>')
            xml_parts.append('</web_search_context>')

            if source_count > 0:
                sources_text = "\n" + "\n".join(xml_parts) + "\n"
                skipped_msg = f", {skipped_count} below min_score={min_score}" if skipped_count else ""
                snippets_msg = f", snippets={'on' if include_snippets else 'off'} (max {snippet_max} chars)"
                print(f"[outlet] Found {source_count} web search sources{skipped_msg}{snippets_msg}")
            else:
                sources_text = ""
                print(f"[outlet] No web search sources found (checked body keys: {[k for k in body.keys() if k != 'messages']}, "
                      f"assistant msg keys: {list(last_msg.keys()) if last_msg else 'n/a'})")

            from datetime import datetime
            datum_zeit = datetime.now().strftime("%d.%m.%Y, %H:%M Uhr")
            has_web_sources = source_count > 0

            if had_web_search:
                # Web-search mode: be very careful not to reject real facts
                system_prompt = (
                    f"Heute ist der {datum_zeit}. "
                    "Du bist ein Qualitätsprüfer für KI-generierte Antworten.\n\n"
                    "WICHTIGER KONTEXT ZU DEINEN EIGENEN GRENZEN:\n"
                    "Dein Trainingsdaten-Cutoff liegt in der Vergangenheit. Die Antwort, die du prüfen sollst, "
                    "wurde von einem System erstellt, das ZUSÄTZLICH eine Live-Websuche durchgeführt hat "
                    "und dabei aktuelle Nachrichtenquellen (z.B. tagesschau.de, welt.de, sueddeutsche.de) "
                    "ausgewertet hat.\n\n"
                    "Das bedeutet KONKRET:\n"
                    "- Die Antwort enthält Fakten aus dem Jahr 2026 — Wahlergebnisse, aktuelle Amtsinhaber, "
                    "Gerichtsprozesse, Wirtschaftsereignisse, Kulturpreise usw.\n"
                    "- Diese Fakten sind REAL und stammen aus seriösen Nachrichtenquellen.\n"
                    "- Du wirst viele dieser Fakten NICHT kennen, weil sie nach deinem Trainings-Cutoff liegen.\n"
                    "- Beispiel: Wenn die Antwort sagt 'Bundeskanzler Friedrich Merz', dann IST er Kanzler — "
                    "auch wenn du das aus deinem Training nicht bestätigen kannst. "
                    "Die Websuche hat es aus aktuellen Quellen.\n\n"
                    "REGELN:\n"
                    "1. Dein fehlendes Wissen ist NIEMALS ein Argument gegen die Antwort.\n"
                    "2. Markiere NICHTS als 'Halluzination', 'fiktiv', 'erfunden', 'spekulativ' oder "
                    "'nicht bestätigt', nur weil du es nicht kennst.\n"
                    "3. Hinterfrage KEINE Personen in politischen Ämtern, Wahlergebnisse, Filmpreise "
                    "oder andere Fakten, die aus den Web-Quellen stammen könnten.\n"
                    "4. Prüfe AUSSCHLIESSLICH:\n"
                    "   a) Widerspricht sich die Antwort INTERN (sagt sie an einer Stelle X und an anderer Stelle nicht-X)?\n"
                    "   b) Sind Wochentage/Datumsangaben INNERHALB der Antwort konsistent?\n"
                    "   c) Ist die Darstellung auffällig einseitig oder unausgewogen?\n"
                    "   d) Fehlen offensichtlich wichtige Aspekte des Themas?\n"
                    "5. Im Zweifel: [BESTÄTIGT]. Lieber eine korrekte Antwort durchlassen "
                    "als eine korrekte Antwort fälschlich als fehlerhaft markieren.\n"
                )
            else:
                # No web search: standard reflection, can check facts against training knowledge
                system_prompt = (
                    f"Heute ist der {datum_zeit}. "
                    "Du bist ein Qualitätsprüfer für KI-generierte Antworten.\n\n"
                    "Die Antwort wurde OHNE Websuche generiert — das Modell hat ausschließlich "
                    "sein Trainingswissen verwendet. Du kannst daher auch Fakten prüfen.\n\n"
                    "Prüfe auf:\n"
                    "1. Faktenfehler oder unbelegte Behauptungen\n"
                    "2. Logische Inkonsistenzen\n"
                    "3. Fehlende wichtige Aspekte\n"
                    "4. Verzerrungen oder einseitige Darstellung\n"
                    "5. Veraltete Informationen (dein Wissensstand könnte aktueller sein als der des Modells)\n\n"
                    "Wenn die Antwort korrekt und vollständig ist, bestätige das kurz.\n"
                    "Wenn Verbesserungen nötig sind, liefere eine korrigierte Version.\n"
                )

            reflection_prompt = "Überprüfe die folgende Antwort:\n\n"
            if has_web_sources:
                reflection_prompt += (
                    "Die Antwort basiert auf einer Live-Websuche. Folgende Quellen wurden abgerufen:\n"
                    f"{sources_text}\n"
                )
            reflection_prompt += (
                f"FRAGE: {user_question}\n\n"
                f"ANTWORT: {original_answer}\n\n"
                "Beginne mit [BESTÄTIGT] wenn die Antwort in sich schlüssig ist, "
                "oder [REFLEXION] wenn du interne Widersprüche oder Darstellungsprobleme findest."
            )
        except Exception as e:
            import traceback
            print(f"Self-Reflection Filter error:\n{traceback.format_exc()}")
            return body

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.valves.api_base_url}/chat/completions",
                    json={
                        "model": body.get("model", ""),
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": reflection_prompt},
                        ],
                        "max_tokens": self.valves.max_tokens,
                        "temperature": self.valves.temperature,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        msg = data["choices"][0]["message"]
                        # SGLang may return thinking in reasoning_content and answer in content,
                        # or content may be None if the model only produced reasoning.
                        reflection = msg.get("content") or msg.get("reasoning_content") or ""
                        print(f"Self-Reflection response (status={resp.status}, len={len(reflection)}): {reflection[:30]}")

                        print("Full Self-Reflection response:")
                        print(reflection)

                        if not reflection.strip():
                            print("not reflection.strip()")
                            pass  # Empty response — keep original
                        elif reflection.startswith("[REFLEXION]"):
                            print("ELIF")
                            messages[-1]["content"] = (
                                original_answer
                                + "\n\n---\n*Nach Selbstreflexion überarbeitet:*\n\n"
                                + reflection.replace("[REFLEXION]", "").strip()
                            )
                        else:
                            print("ELSE")
                            messages[-1]["content"] = (
                                original_answer
                                + "\n\n---\n✅ *Selbstprüfung bestanden.*"
                            )
                    else:
                        body_text = await resp.text()
                        print(f"Self-Reflection API error: status={resp.status}, body={body_text[:500]}")
        except Exception:
            import traceback
            print(f"Self-Reflection Filter error:\n{traceback.format_exc()}")

        body["messages"] = messages
        return body
