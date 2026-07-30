"""Product API layer.

This package turns the research runtime in `aether` into the public Text
API described in plan.md (§5-6): a constrained, streaming event contract
over HTTP. `aether` never imports from here — this package depends on
`aether`, not the other way round, so the research core stays usable
without FastAPI or any product-only dependency installed.
"""
