"""
gene_info -- Fetch gene annotations from mygene.info.

Queries the public mygene.info REST API for a mouse gene symbol and returns
the gene name, summary, GO Biological Process terms, and KEGG pathways.

API docs: https://docs.mygene.info/
"""

from __future__ import annotations

import json
import urllib.request

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "gene_info",
        "description": (
            "Look up annotations for a mouse gene from mygene.info, "
            "including gene name, summary, Gene Ontology biological "
            "process terms, and KEGG pathways."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gene_symbol": {
                    "type": "string",
                    "description": "Mouse gene symbol (e.g. 'Stat1').",
                },
            },
            "required": ["gene_symbol"],
        },
    },
}

_API_URL = (
    "https://mygene.info/v3/query?"
    "q=symbol:{symbol}&species=mouse"
    "&fields=symbol,name,summary,go.BP,pathway.kegg&size=1"
)


def gene_info(gene_symbol: str) -> str:
    """Fetch gene annotations and return a human-readable summary."""
    url = _API_URL.format(symbol=gene_symbol)
    try:
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return f"Error querying mygene.info for {gene_symbol}: {e}"

    hits = data.get("hits", [])
    if not hits:
        return f"No results found for gene symbol '{gene_symbol}' in mouse."

    hit = hits[0]
    lines = [f"Gene: {hit.get('symbol', gene_symbol)}"]

    name = hit.get("name")
    if name:
        lines.append(f"Full name: {name}")

    summary = hit.get("summary")
    if summary:
        lines.append(f"Summary: {summary}")

    go_bp = hit.get("go", {}).get("BP", [])
    if isinstance(go_bp, dict):
        go_bp = [go_bp]
    if go_bp:
        terms = list({t["term"] for t in go_bp if "term" in t})[:8]
        if terms:
            lines.append(
                f"GO Biological Process ({len(terms)} shown): " + "; ".join(terms)
            )

    pathways = hit.get("pathway", {}).get("kegg", [])
    if isinstance(pathways, dict):
        pathways = [pathways]
    if pathways:
        pnames = [p.get("name", p.get("id", "?")) for p in pathways][:5]
        lines.append(f"KEGG Pathways: " + "; ".join(pnames))

    return "\n".join(lines)
