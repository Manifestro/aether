# AETHER — Handoff 1 (API Track)

**Organization:** Manifestro
**Current route:** semantic core → Text API → Web Chat
**Research route:** native Voice Head remains parallel, not blocking the API

Этот handoff обновлён после успешного Stage 2. Следующий агент должен начинать с продуктового API-слоя, а не с повторного подключения Qwen или Voice Head.

---

## 1. Что доказано

На Qwen3-1.7B и NVIDIA A100 был запущен interleaved dual-session runtime:

```text
MCP started:          2220 ms
Speaker first token:  4003 ms
MCP completed:        5221 ms
Factual chunk start:  5980 ms
```

Speaker начал безопасный ответ примерно за **1218 мс до MCP result**. Factual chunk появился только после получения результата.

Финальный ответ:

```text
Проверка погоды началась, результат ещё ожидается.
В городе Алматы сейчас дождь, температура 24 градуса Цельсия.
```

Отчёт: [`technical_report_01.md`](technical_report_01.md).

Это доказывает semantic/action lookahead. Ещё не доказаны native audio, streaming input, barge-in и production API.

---

## 2. Главный следующий результат

Сделать публично пригодный Text API поверх уже работающего `DualSessionRuntime`.

```text
Web Chat ─┐
          ├──► Text API ─► Planner ─► MCP ─► Text Speaker
Developers┘
```

API должен показывать не chain-of-thought, а безопасный поток событий:

```text
tool_started
response.safe_delta
tool_completed
response.delta
turn_completed
```

Позже `response.safe_delta`/`response.delta` заменяются или дополняются audio chunks. Planner, MCP, dependencies, revisions, scheduler и commit horizon сохраняются.

---

## 3. Текущая архитектура

### Domain

- `SemanticEvent`, `EventKind`;
- `ToolCall`, `ToolResult`;
- `SpeechChunk` и строгие состояния;
- dependencies и factual commitment;
- монотонный `Timeline`.

### Runtime

- `SequentialBaseline` — контрольный последовательный pipeline;
- `DualSessionRuntime` — Planner/MCP/Speaker overlap;
- `InterleavedDecodeScheduler` — token-step scheduling;
- нормальный `turn_complete`, cancellation и revision boundaries ещё нужно унифицировать в API telemetry.

### Model adapters

- `QwenPlannerAdapter`;
- `QwenSpeakerAdapter`;
- `SharedQwenBackbone` с lazy load;
- `QwenTokenStepEngine` с отдельным decode state/KV-cache;
- `SemanticEventStreamParser`;
- `ScriptedSharedBackend` и `FakeTokenStepEngine` для тестов без весов.

### MCP

Пока используются deterministic fake tools. Официальный MCP client подключается следующим отдельным adapter-слоем; domain/runtime не должны зависеть от конкретного транспорта.

---

## 4. Структура проекта

```text
aether/
├── spec.md
├── plan.md
├── invest_pitch.md
├── technical_report_01.md
├── handoff-1.md
├── COLAB.md
├── pyproject.toml
├── configs/
│   ├── model/
│   └── experiments/
├── notebooks/
│   ├── aether_stage1_colab.ipynb
│   └── aether_stage2_colab.ipynb
├── src/aether/
│   ├── domain/
│   ├── model/
│   ├── runtime/
│   ├── testing/
│   └── experiments/
└── tests/
```

---

## 5. Проверка

Dependency-free локальная проверка:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Актуальный результат после scheduler и sequence-repair тестов:

```text
Ran 21 tests
OK
```

Также проверяются:

```bash
PYTHONPYCACHEPREFIX=/tmp/aether-pycache \
PYTHONPATH=src \
python3 -m compileall -q src tests

python3 -m json.tool notebooks/aether_stage2_colab.ipynb >/dev/null

git diff --check
```

На development machine не скачивать модели, weights или ML dependencies. Для настоящего Qwen smoke test использовать отдельную ML-среду и `notebooks/aether_stage2_colab.ipynb`.

---

## 6. Product Track — следующий milestone

### 6.1. API domain contract

Создать API-independent service layer, который преобразует внутренние runtime events в публичные события.

Предлагаемые события:

```text
turn.started
plan.tool_started
response.safe_delta
tool.completed
response.delta
turn.completed
turn.failed
```

Каждое событие должно содержать:

- `turn_id`;
- `sequence`;
- `timestamp_ms`;
- `type`;
- `payload`;
- `revision_id` при изменении плана.

Не публиковать hidden states, chain-of-thought или внутренние prompts.

### 6.2. Text API

Основной endpoint:

```http
POST /v1/turns
```

Request:

```json
{
  "message": "Какая погода в Алматы?",
  "tools": ["weather"],
  "stream": true
}
```

Response transport:

- SSE для первого публичного preview;
- WebSocket позже для bidirectional audio и barge-in.

Пример SSE:

```text
event: plan.tool_started
data: {"tool":"weather","arguments":{"location":"Almaty"}}

event: response.safe_delta
data: {"text":"Проверка погоды началась..."}

event: tool.completed
data: {"temperature_c":24,"condition":"rain"}

event: response.delta
data: {"text":"Ожидается дождь, зонт лучше взять."}

event: turn.completed
data: {}
```

### 6.3. API security

Первый public preview должен иметь:

- API keys;
- per-key rate limit;
- daily quota;
- max input/context length;
- tool timeout;
- max concurrent turns;
- structured error responses;
- usage counters;
- key revoke/rotate.

Не разрешать произвольные пользовательские MCP URLs. На старте использовать только allowlisted sandboxed tools.

### 6.4. Web Chat

Сделать Web Chat тонким клиентом собственного API.

Показывать:

- streaming safe response;
- понятный статус запуска tool;
- результат tool;
- продолжение ответа;
- latency debug panel для internal mode.

Не показывать:

- chain-of-thought;
- raw hidden states;
- системные prompts;
- секреты и внутренние tool credentials.

---

## 7. Product Definition of Done

API preview готов, когда:

- один turn стабильно стримит события через SSE;
- weather и ещё 2–4 allowlisted tools работают;
- tool result никогда не заменяется догадкой;
- timeout/error превращается в корректный `turn.failed` или safe continuation;
- API keys и quotas работают;
- каждый turn сохраняет trace;
- Web Chat использует ровно тот же API;
- fake tests остаются независимыми от сети и модели;
- есть базовые p50/p95 latency measurements.

---

## 8. Research Track после API

API не отменяет Voice Head, но не должен блокировать его разработку.

Следующая research последовательность:

1. plan versions и revision queue;
2. cancel/replan только buffered output;
3. latency sweep MCP `3000/1500/750/300 ms`;
4. constrained Planner grammar;
5. text Speaker revision test;
6. audio backbone candidate evaluation;
7. Mimi codec prototype;
8. semantic plan → audio codebook generation;
9. audio commit horizon;
10. VAD и barge-in;
11. hidden-state bridge ablation;
12. unified semantic/audio heads только при доказанном выигрыше.

Qwen3-1.7B остаётся Planner-кандидатом. Не обучать LLM с нуля. Финальный speech backbone выбирать экспериментально.

---

## 9. Инварианты для следующего агента

1. Не добавлять закрытый LLM API в критический research path.
2. Не загружать веса на development machine.
3. Не ломать `SequentialBaseline` — он нужен для измерений.
4. Tool-dependent chunk не становится committed до успешного факта.
5. `COMMITTED` и `PLAYED` не переписываются.
6. Не раскрывать chain-of-thought в API или UI.
7. Любое ускорение подтверждать timeline, а не впечатлением.
8. Product API должен использовать существующий runtime, а не отдельную имитацию.
9. Web Chat — клиент API, не отдельная логика.
10. Fake tests не должны зависеть от GPU, сети или Hugging Face.

---

## 10. Ближайшая задача для следующего агента

Реализовать первый API vertical slice:

```text
DualSessionRuntime
        ↓
EventMapper
        ↓
FastAPI POST /v1/turns
        ↓
SSE stream
        ↓
curl integration test
```

Минимальный запуск должен работать с deterministic fake Planner/Speaker/MCP без ML-зависимостей. После этого добавить optional Qwen provider через существующие adapters.

Ожидаемый первый curl-сценарий:

```bash
curl -N http://localhost:8000/v1/turns \
  -H 'content-type: application/json' \
  -H 'authorization: Bearer dev-key' \
  -d '{"message":"Какая погода в Алматы?","tools":["weather"],"stream":true}'
```

Definition of done этой задачи:

- SSE отдаёт `tool_started`, `safe_delta`, `tool_completed`, `delta`, `completed`;
- sequence и timestamps монотонны;
- ошибки сериализуются в API event;
- 3–5 API integration tests проходят без модели;
- README содержит запуск API;
- новый результат фиксируется в `technical_report_02.md` или handoff-2.
