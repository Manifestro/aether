# VOX-SYNAPSE — Handoff 1

> **Актуализация:** после первоначального handoff был завершён model-ready adapter layer. Текущий статус и новый следующий шаг перечислены ниже. Эта секция имеет приоритет над историческим разделом «Следующий конкретный milestone».

### Выполнено после первоначального handoff

- добавлен provider-neutral `TextGenerationBackend`;
- добавлены `GenerationRequest` и `GenerationSettings`;
- реализован incremental JSONL parser для semantic events;
- parser валидирует типы событий, возрастающий `sequence`, `tool_call` и `speech_plan` payload;
- реализованы `QwenPlannerAdapter` и `QwenSpeakerAdapter`;
- реализован ленивый `SharedQwenBackbone`;
- Transformers/PyTorch импортируются только внутри явного `load()`;
- `allow_download=False` и `local_files_only=True` используются по умолчанию;
- один backend обслуживает независимые `planner:<turn_id>` и `speaker:<turn_id>` сессии;
- добавлен `ScriptedSharedBackend`, проверяющий весь model contract без весов и ML-зависимостей;
- `SpeechChunk` теперь хранит `turn_id` для изоляции Speaker-сессий;
- полный набор содержит **17 dependency-free tests**, все проходят.

Новые ключевые файлы:

```text
src/vox/model/generation.py
src/vox/model/event_parser.py
src/vox/model/qwen_adapters.py
src/vox/model/qwen_backbone.py
tests/test_event_parser.py
tests/test_qwen_adapters.py
tests/test_qwen_backbone_safety.py
```

### Актуальный следующий milestone

В среде, где разрешены веса, выполнить offline smoke test `SharedQwenBackbone` с Qwen3-1.7B и записать фактический model output. Затем, не меняя adapter contract:

1. исправить реальные отклонения JSONL через prompt/parser/constrained decoding;
2. добавить JSONL trace writer;
3. отделить model loading notebook/script от библиотечного кода;
4. измерить VRAM и latency одного Planner и одного Speaker запроса;
5. заменить сериализованный `generate` lock на экспериментальный decode scheduler с двумя KV-cache;
6. сравнить sequential model baseline и dual-session model runtime;
7. сохранить fake tests полностью независимыми от модели и сети.

Текущий `SharedQwenBackbone` намеренно сериализует Hugging Face `generate` вызовы. Он проверяет загрузку одного набора весов и adapter contract, но ещё не обеспечивает истинное конкурентное декодирование на GPU. Это следующая исследовательская граница.

## Назначение документа

Этот файл предназначен для следующего агента, который продолжит проект с текущего состояния. Сначала прочитай:

1. [`spec.md`](spec.md) — полная исследовательская спецификация v0.3;
2. [`invest_pitch.md`](invest_pitch.md) — краткое позиционирование идеи;
3. этот handoff;
4. код и тесты, перечисленные ниже.

Не заменяй исследовательское ядро связкой закрытых LLM/ASR/TTS API. Цель проекта — проверить управляемую нами dual-stream архитектуру на открытых весах.

---

## Идея проекта

VOX-SYNAPSE проверяет гипотезу, что один открытый LLM backbone может обслуживать два параллельных логических потока:

- **Planner** опережает голос, определяет намерение, строит semantic events и рано запускает MCP;
- **Speaker** начинает с безопасных фрагментов и использует результат MCP только в ещё не зафиксированной части ответа.

Ключевой эффект:

> Агент начинает действовать раньше, чем заканчивает говорить.

Долгосрочно Speaker должен генерировать нативные токены аудиокодека Mimi. Текущий этап изолирует и доказывает orchestration, event ordering и factual commitment до подключения модели и аудио.

---

## Зафиксированные архитектурные решения

1. Основной backbone-кандидат: **Qwen3-1.7B**.
2. Контрольный апгрейд при недостаточном качестве: **Qwen3-4B**.
3. Один набор весов должен быть загружен один раз.
4. Planner и Speaker имеют независимые prompts, token sequences и KV-cache.
5. Первичный канал между потоками — типизированные structured semantic events.
6. После event baseline исследуется hidden-state projector + gated cross-attention.
7. MCP и factual commitment не должны зависеть только от непрозрачных embeddings.
8. Основной аудиокодек позднего этапа — Kyutai Mimi.
9. Язык проекта — Python; целевой runtime Python 3.12.
10. Закрытые API допустимы только как внешний baseline/разметчик, не как доказательство концепции.

---

## Что уже реализовано

### Доменное ядро

- `SemanticEvent` и `EventKind`;
- типы `ToolCall` и `ToolResult`;
- `SpeechChunk` и строгая машина состояний;
- dependency resolution;
- запрет отмены после `COMMITTED`;
- append-only монотонный `Timeline`;
- измерение интервалов между trace events.

### Adapter protocols

В `src/vox/model/protocols.py` определены:

- `Planner`;
- `Speaker`;
- `ToolExecutor`;
- `EventValidator`.

Реальный Qwen adapter должен реализовать эти контракты либо минимально расширить их без протекания Transformers-деталей в domain/runtime.

### SequentialBaseline — Этап 0

Файл: `src/vox/runtime/sequential.py`.

Порядок выполнения:

```text
Planner completes/tool executes → Speaker starts
```

Это контрольный вариант для сравнения порядка событий и latency.

### DualSessionRuntime — первая часть Этапа 1

Файл: `src/vox/runtime/dual_session.py`.

Реализовано:

- Planner потоково выдаёт semantic events;
- `tool_call` немедленно создаёт отдельную `asyncio.Task`;
- ready chunks поступают в bounded `asyncio.Queue`;
- Speaker работает конкурентно с MCP;
- зависимые chunks остаются `BLOCKED`;
- успешный tool result разблокирует зависимости;
- tool exception конвертируется в наблюдаемый failed `ToolResult`;
- после ошибки зависимый фрагмент не генерируется и не коммитится;
- сохраняется полная временная трасса;
- при исключении Planner дочерние tool/Speaker задачи отменяются.

Доказанный порядок для медленного weather tool:

```text
tool_started
safe lead-in generated/played
tool_completed
tool-dependent answer generated/played
```

В sequential baseline:

```text
tool_started
tool_completed
speaker_started
```

Это первое локальное доказательство overlap, пока на deterministic fake-компонентах.

### Детерминированные компоненты

Файл: `src/vox/testing/fakes.py`.

- `WeatherPlanner`;
- `FakeWeatherTool` с latency/error режимами;
- `DeterministicSpeaker`.

Они нужны как стабильные fixtures. Не удаляй их после подключения Qwen: реальные adapters должны сравниваться с детерминированным runtime baseline.

---

## Текущая структура

```text
vox/
├── README.md
├── spec.md
├── invest_pitch.md
├── handoff-1.md
├── pyproject.toml
├── configs/
│   └── experiments/
│       └── sequential_weather.yaml
├── src/vox/
│   ├── domain/
│   │   ├── chunks.py
│   │   ├── events.py
│   │   └── timeline.py
│   ├── model/
│   │   └── protocols.py
│   ├── runtime/
│   │   ├── sequential.py
│   │   └── dual_session.py
│   └── testing/
│       └── fakes.py
└── tests/
    ├── test_chunks.py
    ├── test_dual_session.py
    ├── test_sequential_weather.py
    └── test_timeline.py
```

---

## Проверка текущего состояния

Локальный системный Python в исходной среде — 3.9.6, хотя целевая версия проекта в `pyproject.toml` — Python 3.12. Доменное ядро намеренно пока совместимо с 3.9, чтобы его можно было проверить без установки зависимостей.

Запуск dependency-free тестов:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Ожидаемый актуальный результат:

```text
Ran 17 tests
OK
```

Проверка компиляции в ограниченной macOS-среде:

```bash
PYTHONPYCACHEPREFIX=/tmp/vox-pycache \
PYTHONPATH=src \
python3 -m compileall -q src tests
```

Обычная `compileall` без `PYTHONPYCACHEPREFIX` может пытаться писать в системный каталог `~/Library/Caches`, недоступный sandbox. Это не ошибка кода.

Для целевой среды:

```bash
uv sync --extra dev
uv run pytest
```

ML-зависимости устанавливаются отдельно:

```bash
uv sync --extra dev --extra ml
```

Не устанавливай `audio` extra до этапа, где действительно подключается Mimi.

---

## Следующий конкретный milestone

Реализовать **shared Qwen backbone adapter** для Уровня A.

Минимальный результат следующего этапа:

1. `QwenBackbone` загружает `Qwen3-1.7B` один раз.
2. Создаются две независимые logical sessions: Planner и Speaker.
3. В первой версии допустим обычный `generate`/streaming loop без оптимального batching.
4. Planner выдаёт валидные `SemanticEvent`, а не свободный текст.
5. Speaker выполняет `speech_plan` и возвращает текстовый chunk.
6. Оба adapter используются существующим `DualSessionRuntime`.
7. Weather scenario проходит с реальной моделью.
8. Trace показывает, что MCP запущен до tool-dependent Speaker chunk.
9. Fake tests остаются зелёными.

### Рекомендуемая последовательность

1. Добавить `src/vox/model/qwen_backbone.py`:
   - tokenizer;
   - один `AutoModelForCausalLM`;
   - lifecycle/load/unload;
   - device/dtype configuration;
   - доступ к raw forward и hidden states для следующего этапа.
2. Добавить `src/vox/model/qwen_planner.py`:
   - строгий prompt;
   - incremental parsing JSONL events;
   - validation;
   - bounded output;
   - non-thinking/короткий action режим.
3. Добавить `src/vox/model/qwen_speaker.py`:
   - отдельный conversation context;
   - генерация по `SpeechChunk.goal` и доступным facts;
   - запрет самостоятельного добавления неподтверждённых фактов на уровне prompt и validator.
4. Добавить offline integration script/notebook для одного weather turn.
5. Сохранить trace в JSONL.

### Важное замечание о «двух экземплярах»

Не загружать модель дважды. На первом неоптимизированном варианте два последовательных вызова одного model object допустимы, но логические контексты должны быть раздельными. Следующий шаг после функционального adapter — собственный decode scheduler с двумя KV-cache и dynamic batching.

---

## Недостающие части и известные ограничения

- Сейчас Planner и Speaker — deterministic fakes, реальных весов ещё нет.
- `DualSessionRuntime` доказывает task overlap, но ещё не управляет decode steps или GPU batching.
- Dependency key сейчас равен имени инструмента; для нескольких одновременных вызовов одного tool потребуется dependency по `call_id` или отдельному fact key.
- `facts` и `chunks` изменяются в одном event loop без lock; это корректно для текущего asyncio runtime, но правила владения состоянием нужно сохранить при добавлении threads/processes.
- Нет constrained decoder для JSON events.
- Нет persistence JSONL trace writer.
- Нет timeout policy на уровне runtime; fake tool моделирует latency/error, но не зависание.
- Нет barge-in и общего cancellation token turn-а.
- Нет hidden-state bridge.
- Нет Voice Head и Mimi.
- Нет аудиовхода или realtime transport.
- В проекте на момент handoff нет Git-репозитория; не предполагай наличие истории Git.

---

## Инварианты, которые нельзя случайно сломать

1. Tool-dependent chunk не переходит в `READY/COMMITTED` без успешной зависимости.
2. `COMMITTED` и `PLAYED` не отменяются.
3. Planner event sequence строго возрастает.
4. Tool failure является наблюдаемым результатом, а не причиной выдумать факт.
5. Timeline использует монотонные часы.
6. Sequential baseline сохраняется как контроль.
7. Fake tests не зависят от GPU, сети или Hugging Face.
8. Модельные детали не должны проникать в domain types.
9. Один backbone загружается один раз; Planner/Speaker разделяют веса, но не KV-cache.
10. Любое усложнение сравнивается с более простым baseline.

---

## Definition of Done следующего handoff

Следующий агент может завершить свой этап, когда:

- все текущие 10 тестов проходят;
- добавлены unit tests для parser/validator model output;
- один Qwen backbone обслуживает оба adapter;
- weather integration scenario выполняется на открытых весах;
- trace сохраняется и содержит `tool_started`, `speaker_started`, `tool_completed`, `chunk_committed`;
- задокументированы память, latency и обнаруженные ограничения;
- создан следующий `handoff-2.md`.
