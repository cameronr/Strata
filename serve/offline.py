"""The server's --offline: what its children are told, and which configs cannot be strictly offline.

Every child the server starts gets variables that the common runtimes honour (Hugging Face, Transformers, pip, and
the proxy variables that make an accidental HTTP call fail fast instead of hanging). A child that ignores them and
opens a socket directly is not stopped: that would need a sandbox.
"""

from __future__ import annotations

from typing import Any, Mapping

OFFLINE_ENV: dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "PIP_NO_INDEX": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "NO_PROXY": "*",
    "no_proxy": "*",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "http_proxy": "",
    "https_proxy": "",
    "STRATA_OFFLINE": "1",
}


def ensure_strictly_offline(cfg: Mapping[str, Any]) -> list[str]:
    """Why this config cannot be strictly offline (empty: it can): the parts of it that need the network."""
    needs: list[str] = []
    servers = cfg.get("mcp_servers") or cfg.get("mcpServers")
    if servers:
        count = len(servers) if hasattr(servers, "__len__") else 1
        needs.append(f"the config declares {count} MCP server(s), which are started as subprocesses "
                     "and may fetch tools")
    for key in ("before_load", "beforeLoad"):
        if cfg.get(key):
            needs.append(f"the config declares a {key} command, which runs before the model loads")
    for key in ("images", "vision"):
        section = cfg.get(key)
        if isinstance(section, dict):
            for field in ("remote_sources", "allowed_hosts", "url"):
                if section.get(field):
                    needs.append(f"the config's {key}.{field} permits network sources")
    return needs
