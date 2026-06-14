"""
Custom Hindsight TenantExtension — static bearer-key -> Postgres-schema map.

Deployed by roles/k8s_dgx/tasks/hindsight.yml: this file is shipped into the
hindsight-api / hindsight-worker containers via the `hindsight-tenant-ext`
ConfigMap (subPath-mounted at /opt/hindsight-ext/hindsight_tenant_keymap.py)
and loaded through:

    HINDSIGHT_API_TENANT_EXTENSION=hindsight_tenant_keymap:KeyMapTenantExtension
    PYTHONPATH=/opt/hindsight-ext

It gives real per-user memory isolation (each tenant gets its OWN Postgres
schema, exactly like the built-in SupabaseTenantExtension) but WITHOUT any
external auth server (no Supabase / GoTrue): authentication is a constant-time
lookup of the request's bearer token in a static key -> tenant map supplied via
env (HINDSIGHT_API_TENANT_KEYMAP, a JSON object {"<api_key>": "<tenant>"}).

Config (all from HINDSIGHT_API_TENANT_* env vars, see extensions/loader.py):
    HINDSIGHT_API_TENANT_KEYMAP        JSON {"<key>": "<tenant>"}   (required)
    HINDSIGHT_API_TENANT_SCHEMA_PREFIX schema = <prefix>_<tenant>   (default "tenant")

Each tenant's data lives in schema "<prefix>_<tenant>"; all schemas are created
+ migrated at startup so the (optional) dedicated worker can discover them via
list_tenants() for background maintenance.
"""

import hmac
import json
import re

from hindsight_api.extensions.tenant import (
    AuthenticationError,
    Tenant,
    TenantContext,
    TenantExtension,
)
from hindsight_api.models import RequestContext

# Postgres identifier rules; full schema name (<prefix>_<tenant>) must also fit
# the 63-byte identifier limit.
_IDENT = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")
_MAX_IDENT_LEN = 63


class KeyMapTenantExtension(TenantExtension):
    """Authenticate a bearer key against a static map and isolate per tenant."""

    def __init__(self, config: dict[str, str]) -> None:
        super().__init__(config)

        raw = config.get("keymap") or ""
        try:
            self._map: dict[str, str] = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"HINDSIGHT_API_TENANT_KEYMAP is not valid JSON: {exc}") from exc

        if not self._map:
            raise ValueError(
                "HINDSIGHT_API_TENANT_KEYMAP is required (a JSON object mapping "
                'api keys to tenant names, e.g. {"sk-alice": "alice"}).'
            )

        self._prefix = config.get("schema_prefix", "tenant")
        if not _IDENT.match(self._prefix):
            raise ValueError(f"Invalid schema_prefix {self._prefix!r}: must be a valid Postgres identifier.")

        # key -> schema, validated up front so a bad config fails fast at boot.
        self._key_to_schema: dict[str, str] = {}
        for key, tenant in self._map.items():
            if not isinstance(tenant, str) or not _IDENT.match(tenant):
                raise ValueError(f"Invalid tenant name {tenant!r}: must be a valid Postgres identifier.")
            schema = f"{self._prefix}_{tenant}"
            if len(schema) > _MAX_IDENT_LEN:
                raise ValueError(f"Schema name {schema!r} exceeds {_MAX_IDENT_LEN} characters.")
            self._key_to_schema[key] = schema

        self._schemas = set(self._key_to_schema.values())

    async def on_startup(self) -> None:
        """Create + migrate every tenant schema so they're ready (and discoverable)."""
        for schema in sorted(self._schemas):
            await self.context.run_migration(schema)

    async def authenticate(self, context: RequestContext) -> TenantContext:
        """Constant-time match the bearer key to its tenant schema."""
        token = context.api_key or ""

        # Constant-time over all configured keys to avoid leaking which (if any)
        # key prefix matched via timing.
        matched: str | None = None
        for key, schema in self._key_to_schema.items():
            if hmac.compare_digest(token, key):
                matched = schema
        if matched is None:
            raise AuthenticationError("Invalid API key")
        return TenantContext(schema_name=matched)

    async def list_tenants(self) -> list[Tenant]:
        """All configured tenant schemas (used by the worker for task polling)."""
        return [Tenant(schema=schema, tenant_id=schema) for schema in sorted(self._schemas)]
