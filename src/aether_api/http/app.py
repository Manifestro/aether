"""FastAPI wiring for the AETHER Text API (plan.md §5 B1-B2).

Thin transport layer only: HTTP concerns (auth, request validation, SSE
framing) live here. All turn orchestration and event semantics live in
`aether_api.turn_service` / `aether_api.event_mapper`, neither of which
import FastAPI and both of which are tested without an HTTP server.
"""

from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from aether_api.auth import ApiKeyStore
from aether_api.turn_service import TurnRequest, TurnService

# First public preview: no user-supplied MCP endpoints, only this
# sandboxed, server-controlled allowlist (plan.md §5 B2). A request may ask
# for any subset of it; TurnService/AllowlistToolExecutor still enforce the
# per-request subset end to end.
SANDBOXED_TOOLS = frozenset({"weather"})


class TurnRequestBody(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    tools: List[str] = Field(default_factory=list)


def create_app(turn_service: TurnService, api_keys: ApiKeyStore) -> FastAPI:
    app = FastAPI(title="AETHER Text API")

    def _authenticate(request: Request):
        header = request.headers.get("authorization", "")
        if not header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        api_key = api_keys.resolve(header[len("Bearer "):])
        if api_key is None:
            raise HTTPException(status_code=401, detail="invalid api key")
        return api_key

    @app.post("/v1/turns")
    async def create_turn(body: TurnRequestBody, request: Request) -> StreamingResponse:
        api_key = _authenticate(request)

        disallowed = [tool for tool in body.tools if tool not in SANDBOXED_TOOLS]
        if disallowed:
            raise HTTPException(status_code=400, detail=f"tool not allowed: {disallowed}")

        if api_key.active_turns >= api_key.max_concurrent_turns:
            raise HTTPException(status_code=429, detail="max_concurrent_turns exceeded")

        turn_request = TurnRequest(message=body.message, tools=body.tools)

        async def event_stream():
            with api_keys.claim_turn_slot(api_key):
                async for event in turn_service.stream_turn(turn_request):
                    yield event.to_sse()

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
