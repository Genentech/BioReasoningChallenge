#!/usr/bin/env python
"""Start vLLM serve with a patch that fixes -inf logprobs in JSON responses.

Starlette's JSONResponse uses ``json.dumps(allow_nan=False)``, which
crashes when vLLM produces -inf log-probabilities (common with reasoning
models).  This wrapper patches the response renderer to clamp -inf/nan
to -9999.0 (matching OpenAI's convention) before serialization.

Usage (drop-in replacement for ``vllm serve``):

    python serve_with_logprobs_fix.py openai/gpt-oss-120b --port 8000

    # or with uv:
    uv run --extra serve python serve_with_logprobs_fix.py \\
        openai/gpt-oss-120b --port 8000

All arguments are forwarded to ``vllm serve``.
"""

import math
import sys


def _patch_json_response() -> None:
    import starlette.responses

    _orig_render = starlette.responses.JSONResponse.render

    def _sanitize(obj):
        if isinstance(obj, float):
            if math.isinf(obj) or math.isnan(obj):
                return -9999.0
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_sanitize(v) for v in obj)
        return obj

    def _safe_render(self, content):
        return _orig_render(self, _sanitize(content))

    starlette.responses.JSONResponse.render = _safe_render


_patch_json_response()

if __name__ == "__main__":
    # Inject "serve" as the subcommand if the user didn't provide it,
    # so `python serve_with_logprobs_fix.py model --port 8000` works.
    if len(sys.argv) < 2 or sys.argv[1] != "serve":
        sys.argv.insert(1, "serve")

    from vllm.entrypoints.cli.main import main  # noqa: E402

    sys.exit(main())
