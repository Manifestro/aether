# AETHER — план развития

**Организация:** Manifestro  
**Статус:** active research and productization plan  
**Версия:** 0.1  
**Дата:** 30 июля 2026 года

---

## 1. Стратегическое направление

AETHER развивается одновременно в двух связанных направлениях:

```text
                         AETHER Core
                         /               \
                        /                 \
             Product Track             Research Track
          Text API + Web Chat        Native Voice Head
```

### Product Track

Цель — быстро превратить уже доказанный semantic dual-stream runtime в доступный продукт:

- Text API;
- Web Chat Playground;
- потоковые semantic/tool events;
- API keys и лимиты;
- ранние пользователи и реальные сценарии MCP.

### Research Track

Цель — заменить текстовый Speaker на нативный streaming audio Speaker, который видит изменяющийся semantic state и умеет перестраивать ещё не зафиксированную часть речи.

Оба направления используют одно ядро:

- Planner;
- MCP Engine;
- semantic event protocol;
- dependency graph;
- commit horizon;
- scheduler;
- timeline и telemetry.

---

## 2. Что уже доказано

На Qwen3-1.7B и NVIDIA A100 был проведён реальный Stage 2 эксперимент:

- один shared backbone;
- независимые Planner/Speaker decode states;
- interleaved token scheduler;
- synthetic MCP с задержкой 3000 мс;
- safe Speaker chunk до завершения MCP;
- factual Speaker chunk после результата инструмента.

Зафиксированный результат:

```text
MCP completed:        5221 ms
Speaker first token:  4003 ms
Lead:                 approximately 1218 ms
```

Это подтверждает semantic/action lookahead. Пока не доказаны:

- native audio generation;
- streaming audio input;
- barge-in;
- physical concurrent CUDA execution;
- hidden-state conditioning;
- production API reliability.

---

## 3. Архитектурные принципы

1. **Не обучать LLM с нуля.** Использовать pretrained open-weight backbone и обучать собственные heads, projectors и adapters.
2. **Qwen оставить Planner-кандидатом.** Он уже показал пригодность для semantic planning и tool use, но не считается автоматически финальным speech backbone.
3. **Не передавать chain-of-thought.** Между потоками передаются observable semantic updates, tool actions, facts и dependencies.
4. **Изменять только незафиксированную речь.** Committed/playback audio необратим; buffered/speculative output можно отменить.
5. **Каждое усложнение сравнивать с baseline.** Если hidden bridge или native Voice Head не дают измеримого выигрыша, сохраняется более простая архитектура.
6. **Никаких весов в development workspace.** Model loading выполняется только в отдельной ML-среде.
7. **Все этапы должны оставлять trace.** Любое утверждение о скорости подтверждается временной шкалой.

---

## 4. Фаза A — укрепление semantic core

**Цель:** сделать Planner/Scheduler/MCP runtime надёжным и пригодным для API.

### A1. Semantic event protocol

Перейти от свободного JSONL к ограниченной event grammar:

```text
tool_call
speech_plan(safe)
speech_plan(dependent)
fact / tool_error
replan
turn_complete
```

Задачи:

- убрать необязательный `intent` для actionable запросов;
- добавить constrained decoding или grammar validation;
- устранить повторяющиеся sequence numbers без silent repair в production;
- добавить `plan_version` и `revision_id`;
- добавить `safe_to_say` и `depends_on` поля;
- различать `normal_complete`, `cancelled` и `failed` в telemetry.

**Критерий выхода:** 1000 синтетических turns без dependency violation и с корректным event schema.

### A2. Revision protocol

Ввести очередь обновлений для Speaker:

```json
{
  "revision_id": 4,
  "status": "tool_completed",
  "safe_chunks": [],
  "facts": {"temperature_c": 24},
  "invalidate_after": "lead-in"
}
```

Speaker обязан:

- принимать новую revision во время генерации;
- отменять только speculative/buffered output;
- сохранять committed prefix;
- запускать продолжение с актуального semantic state.

**Критерий выхода:** 100% тестов на смену плана до commit horizon проходят без противоречащей речи.

### A3. Scheduler benchmark

Сравнить четыре режима:

1. sequential Planner → MCP → Speaker;
2. async runtime с заблокированным `generate`;
3. interleaved token scheduler;
4. interleaved scheduler с разными весами Planner/Speaker.

Провести sweep MCP latency:

```text
3000 ms → 1500 ms → 750 ms → 300 ms
```

Собирать:

- time-to-tool-call;
- first Speaker token;
- safe chunk commit;
- tool completion;
- factual chunk start;
- total turn latency;
- token steps per stream;
- VRAM.

**Критерий выхода:** scheduler сохраняет положительный overlap на всех tool latency, где safe chunk физически успевает быть сгенерирован.

---

## 5. Фаза B — Text API и Web Chat

**Цель:** выпустить первый публичный интерфейс поверх semantic core, не ожидая Voice Head.

### B1. API contract

Основной endpoint:

```http
POST /v1/turns
```

Пример запроса:

```json
{
  "message": "Какая погода в Алматы?",
  "tools": ["weather"],
  "stream": true
}
```

Поток ответа через SSE или WebSocket:

```text
tool_started
response.safe_delta
tool_completed
response.delta
turn_completed
```

API не раскрывает chain-of-thought. Публичными являются только безопасные статусы, действия, результаты и текстовые chunks.

### B2. Auth, quotas и безопасность

- API keys;
- per-key rate limit;
- daily/monthly quota;
- max context length;
- max tool timeout;
- concurrency cap;
- usage logging;
- revoke/rotate keys;
- sandboxed встроенные tools;
- запрет произвольного MCP endpoint на первом этапе.

Бесплатный план должен иметь чёткий лимит. «Бесплатный» означает ограниченный early-access tier, а не безлимитную инфраструктуру.

### B3. Web Chat Playground

Web Chat является клиентом того же API, а не отдельной системой.

UI показывает:

- пользовательское сообщение;
- безопасный streaming response;
- статус запущенного инструмента;
- результат инструмента;
- продолжение ответа;
- latency timeline в debug mode.

Внутренние рассуждения не показываются.

### B4. Product acceptance

Публичный preview можно выпускать, когда:

- streaming API стабилен;
- 3–5 встроенных tools работают в sandbox;
- есть rate limits и API key management;
- ошибки и timeout не приводят к выдуманным фактам;
- каждый turn имеет trace;
- есть хотя бы один удобный chat playground.

---

## 6. Фаза C — Native Voice Head

**Цель:** получить настоящий streaming audio output, не меняя semantic core.

### C1. Выбор аудиобазиса

Исследовать два направления:

1. speech-text foundation model с готовыми audio streams;
2. собственный Voice Head поверх открытого semantic backbone и Mimi.

Qwen3-1.7B не считать готовым audio backbone. Он остаётся Planner-кандидатом, пока отдельный эксперимент не подтвердит обратное.

### C2. Audio representation

Основной кандидат — Mimi:

```text
semantic plan + speaker state
            │
            ▼
Voice Head
            │
            ▼
multi-codebook audio tokens
            │
            ▼
Mimi decoder → PCM
```

Voice Head должен учитывать:

- temporal dependencies;
- несколько codebooks;
- semantic/acoustic separation;
- delayed/interleaved decoding;
- voice style and prosody;
- safe transition boundaries.

### C3. Минимальный audio prototype

Сначала не пытаться решить всю речь end-to-end. Проверить:

- text/semantic plan → audio codebook tokens;
- streaming decode;
- chunk cancellation;
- continuation после MCP result;
- отсутствие audible splice на границе.

**Критерий выхода:** первый audio chunk генерируется до tool completion, а factual audio chunk — после результата.

---

## 7. Фаза D — Real-time input и barge-in

**Цель:** перейти от text turns к полноценному голосовому циклу.

Компоненты:

- streaming audio encoder;
- VAD;
- input turn detector;
- barge-in cancellation;
- playback buffer;
- commit horizon;
- аудиоперепланирование;
- interruption policy для MCP.

Сценарии:

1. перебивание во время safe lead-in;
2. перебивание во время MCP ожидания;
3. перебивание после factual chunk;
4. отмена старого плана и начало нового turn;
5. сохранение или отмена MCP согласно idempotency policy.

**Критерии выхода:**

- stop-on-barge-in `<150 мс`;
- ни одного слышимого factual contradiction в тестовом наборe;
- buffered audio корректно отменяется;
- committed audio не переписывается.

---

## 8. Фаза E — Hidden-state bridge и unified architecture

**Цель:** проверить, даёт ли прямой latent bridge преимущество над structured events.

### E1. Hidden-state bridge

```text
Planner hidden states
        │
        ▼
Projector + gate
        │
        ▼
Speaker cross-attention
```

Параметры первого эксперимента:

- frozen backbone;
- trainable projector;
- gated residual cross-attention;
- optional LoRA;
- structured events сохраняются как safety/control channel.

### E2. Обязательная ablation

Сравнить:

- A — только structured events;
- B — events + hidden bridge;
- C — только hidden bridge.

Hidden bridge принимается только при измеримом выигрыше в:

- latency;
- semantic consistency;
- качество continuation;
- naturalness;
- reduction of revision artifacts.

### E3. Unified dual-head backbone

Только после подтверждения E1/E2 рассматривать:

```text
Shared temporal backbone
        ├── Semantic/Action Head
        └── Voice/Audio Head
```

Это не означает обучение LLM с нуля. Используется pretrained backbone с собственными heads, adapters и objective.

---

## 9. Метрики проекта

### Semantic/action

- tool-call accuracy;
- invalid event rate;
- duplicate sequence rate;
- time-to-tool-call;
- dependency violation rate;
- unsupported factual commitment rate;
- successful replan rate.

### Dual-stream

- Speaker first token before MCP completion;
- safe chunk commit before MCP completion;
- factual chunk start after MCP completion;
- planner/speaker token-step ratio;
- scheduler idle time;
- KV-cache memory per stream.

### Voice

- TTFA;
- first audio token latency;
- first PCM chunk latency;
- audible splice rate;
- audio continuation success;
- prosody consistency;
- stop-on-barge-in;
- committed/buffered cancellation correctness.

### Product

- API p50/p95 latency;
- tool success rate;
- streamed completion rate;
- quota usage;
- cost per turn;
- retention and repeated usage;
- user-rated responsiveness.

---

## 10. Decision gates

После каждой фазы принимается одно из решений:

### Continue

Метрики подтверждают гипотезу — переходим дальше.

### Refine

Эффект есть, но требуется изменить protocol, scheduler или training objective.

### Fallback

Сложный механизм не выигрывает у более простого. Оставляем modular architecture.

### Pivot

Пользовательский эффект или технический выигрыш не подтверждены. Меняем направление, сохраняя данные и измерения.

Ни один компонент не становится обязательным только потому, что был в исходной vision-схеме.

---

## 11. Ближайшие задачи

Порядок работы после текущего Stage 2:

1. Синхронизировать normal completion telemetry в runtime и Colab runner.
2. Убрать `intent` из actionable Planner output.
3. Добавить constrained event grammar.
4. Провести latency sweep `3000/1500/750/300 ms`.
5. Реализовать revision queue и plan versions.
6. Подготовить Text API с SSE streaming.
7. Сделать Web Chat Playground поверх API.
8. Выбрать audio backbone для первого Voice prototype.
9. Подключить Mimi и проверить text/semantic-to-audio continuation.
10. Добавить barge-in и real-time input.
11. Исследовать hidden-state bridge.
12. Только после ablation решать вопрос unified dual-head model.

---

## 12. Definition of success

### Ближайший product result

Manifestro предоставляет Text API и Web Chat, где агент:

- начинает потоковый ответ;
- запускает tool параллельно;
- показывает безопасный статус ожидания;
- продолжает ответ с подтверждёнными данными;
- не раскрывает внутренние рассуждения;
- имеет измеримый trace каждого turn.

### Исследовательский result

AETHER получает native streaming Voice Head, который:

- принимает semantic revisions в реальном времени;
- генерирует безопасную речь до MCP result;
- генерирует factual speech после MCP result;
- отменяет только buffered/speculative output;
- сохраняет committed audio;
- поддерживает перебивание;
- демонстрирует latency и naturalness лучше sequential baseline.

---

## 13. Финальная позиция

Мы не обязаны выбирать между продуктом и исследованием.

Text API и Web Chat могут стать первым публичным проявлением AETHER уже сейчас. Их ядро будет тем же, что потребуется Voice Head: Planner, MCP, revisions, dependencies, scheduler и commit horizon.

Voice Head следует строить как отдельный серьёзный research track. Qwen остаётся полезным Planner, но финальный speech backbone должен быть выбран экспериментально. Собственную LLM с нуля не обучаем; создаём собственную архитектуру поверх открытых pretrained компонентов.

Главный критерий — не завершить заранее выбранную архитектуру, а доказать, что AETHER делает голосового агента быстрее, естественнее и способнее действовать в реальном времени.
