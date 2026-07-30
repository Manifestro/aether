# VOX-SYNAPSE — исследовательская спецификация PoC (v0.3)

**Класс системы:** predictive dual-stream speech engine with asynchronous MCP tool binding  
**Статус:** исследовательский proof of concept  
**Основной принцип:** vision сохраняется; конкретная архитектура изменяется только на основании измеримых результатов.

---

## 1. Vision

**VOX-SYNAPSE** — голосовой агент, в котором семантическое планирование и голосовая генерация являются двумя параллельными потоками. Семантический поток опережает голосовой, заранее принимает решение о вызове инструмента и передаёт голосовому потоку смысл будущего ответа.

Целевой эффект:

- агент действует раньше, чем заканчивает говорить;
- MCP-инструменты выполняются параллельно с генерацией ответа;
- голосовой поток начинает с безопасной части ответа и не произносит неподтверждённые данные;
- результат инструмента влияет на ещё не произнесённую часть ответа;
- ожидание, продолжение и перебивание звучат естественно;
- в конечной архитектуре речь генерируется нативно через аудиотокены, без внешнего TTS в критическом контуре.

Проект не является сборкой закрытых ASR/LLM/TTS API. Критические компоненты PoC должны работать на открытых весах и быть доступны для модификации, обучения и измерения внутренних состояний.

---

## 2. Что именно доказывает PoC

Главная гипотеза:

> Один открытый языковой backbone может обслуживать два специализированных потока — Planner и Speaker — так, чтобы Planner стабильно опережал Speaker, заранее запускал MCP и направлял ещё не зафиксированную часть ответа.

PoC должен последовательно доказать три уровня.

### Уровень A — системный dual-stream

- один набор весов обслуживает две независимые decoding-сессии;
- Planner потоково формирует типизированные semantic events;
- Speaker начинает генерацию до завершения Planner и MCP;
- scheduler управляет приоритетами обоих потоков;
- результат MCP изменяет только незафиксированную часть ответа.

### Уровень B — нейронный semantic bridge

- Planner передаёт не только структурированные события, но и hidden states;
- обучаемый projector преобразует их в контекст Speaker;
- Speaker использует этот контекст через cross-attention или prefix embeddings;
- измеряется преимущество hidden-state bridge относительно текстового event baseline.

### Уровень C — нативный voice stream

- Speaker предсказывает дискретные токены аудиокодека;
- semantic lookahead направляет содержание и акустическое продолжение;
- аудио декодируется потоково;
- MCP, commit horizon и barge-in работают без внешнего TTS.

Успех одного уровня не считается автоматическим доказательством следующего.

---

## 3. Инварианты и изменяемые решения

### 3.1. Инварианты идеи

- два параллельных логических потока: Planner и Speaker;
- Planner опережает слышимый голосовой вывод;
- ранний асинхронный MCP-вызов;
- семантическая передача между потоками;
- разделение безопасных и зависящих от инструмента речевых фрагментов;
- управляемый commit horizon;
- отмена и перепланирование при перебивании;
- открытые веса в критическом исследовательском контуре.

### 3.2. Решения, которые разрешено менять

- конкретный открытый backbone;
- размер модели;
- формат semantic events;
- слой, из которого извлекаются hidden states;
- projector, cross-attention или prefix conditioning;
- аудиокодек и схема нескольких codebooks;
- LoRA, partial fine-tuning или обучение только новых модулей;
- размер lookahead и политика scheduler;
- единый dual-head forward pass или две decoding-сессии;
- способ генерации filler и безопасных lead-in фраз.

Мы не сохраняем архитектурное решение только потому, что оно присутствовало в первоначальной схеме. Оно должно выигрывать у baseline по измеряемым критериям.

---

## 4. Выбор backbone

### 4.1. Основная модель: Qwen3-1.7B

Для первого PoC выбирается **Qwen3-1.7B**:

- открытые веса и лицензия Apache 2.0;
- 1.7B параметров позволяют проводить локальные эксперименты с двумя KV-cache;
- стандартная causal Transformer архитектура доступна через Hugging Face Transformers;
- поддерживает многоязычные инструкции и tool-oriented сценарии;
- допускает извлечение hidden states и модификацию forward pass;
- подходит для LoRA и обучения небольших дополнительных модулей.

Planner должен работать в режиме коротких структурированных действий, без длинной генерации рассуждений. Speaker использует тот же backbone, но отдельный system role, контекст, KV-cache и sampling policy.

### 4.2. Контрольная модель: Qwen3-4B

**Qwen3-4B** используется как следующий эксперимент, если 1.7B недостаточно хорошо:

- выбирает инструменты;
- соблюдает схемы semantic events;
- поддерживает связность ответа после MCP;
- генерирует качественный смысловой план.

Переход на 4B выполняется после получения полного baseline на 1.7B. Архитектура и протокол не должны зависеть от размера backbone.

### 4.3. Что не делаем

- не обучаем языковую модель с нуля;
- не загружаем две полные копии одинаковых весов;
- не используем закрытый LLM API как основной Planner или Speaker;
- не привязываем semantic protocol к внутреннему chat template одной модели.

Собственная модель VOX-SYNAPSE означает модифицированный открытый backbone с нашим scheduler, semantic bridge, heads и voice decoder, а не повторное обучение общих языковых знаний с нуля.

---

## 5. Архитектура Уровня A: один backbone, две сессии

```text
                               ┌─────────────────────────┐
                               │ Shared Qwen3 weights    │
                               │ loaded once             │
                               └────────────┬────────────┘
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      │                                           │
                      ▼                                           ▼
          ┌──────────────────────┐                    ┌──────────────────────┐
          │ Planner session      │                    │ Speaker session      │
          │ independent KV-cache │                    │ independent KV-cache │
          │ deterministic policy │                    │ speech policy        │
          └──────────┬───────────┘                    └──────────▲───────────┘
                     │ semantic events                           │
                     ├───────────────────────────────────────────┤
                     │                                           │
                     ▼                                           │
          ┌──────────────────────┐       MCP result               │
          │ Async MCP Engine     ├────────────────────────────────┤
          └──────────────────────┘                                │
                                                                  │
                                                        speakable chunks
                                                                  │
                                                                  ▼
                                                       Commit/Output Buffer
```

Обе сессии используют одни матрицы весов, но имеют:

- разные prompts и роли;
- отдельные последовательности токенов;
- отдельные KV-cache;
- разные sampling parameters;
- независимые критерии остановки;
- общий scheduler вычислений.

При возможности два sequence обрабатываются одним batched forward pass. Planner получает повышенный приоритет до генерации `tool_call`; после запуска инструмента вычислительный приоритет переносится на Speaker.

---

## 6. Semantic Event Protocol

Planner не передаёт Speaker свободный внутренний монолог. Он формирует короткие наблюдаемые события, которые можно валидировать, логировать и воспроизводить.

Минимальные типы событий:

```text
intent
tool_call
tool_pending
speech_plan
fact
tool_error
replan
turn_complete
```

Пример начала запроса о погоде:

```json
{
  "type": "tool_call",
  "sequence": 1,
  "tool": "weather",
  "arguments": {"location": "Almaty"}
}
```

```json
{
  "type": "speech_plan",
  "sequence": 2,
  "goal": "Подтвердить, что погода в Алматы проверяется",
  "dependencies": [],
  "commit_policy": "safe"
}
```

После MCP:

```json
{
  "type": "fact",
  "sequence": 3,
  "source": "weather",
  "content": {
    "temperature_c": 24,
    "condition": "rain"
  }
}
```

```json
{
  "type": "speech_plan",
  "sequence": 4,
  "goal": "Сообщить температуру и дать рекомендацию о зонте",
  "dependencies": ["weather"],
  "commit_policy": "after_dependencies"
}
```

Полный chain-of-thought не сохраняется и не требуется. Исследуемым объектом являются semantic representation, действия и временные зависимости, а не длинные текстовые рассуждения.

---

## 7. Dependency Graph и factual commitment

Ответ моделируется как граф фрагментов:

```text
                       ┌─► [Safe lead-in] ────────────────► READY
[User request] ────────┤
                       └─► [MCP weather] ─► [Weather fact] ─► READY
```

У каждого speech chunk есть:

- `chunk_id`;
- смысловая цель;
- список зависимостей;
- состояние `blocked | ready | generating | buffered | committed | played | cancelled`;
- версия плана;
- временные метки.

Фрагмент, зависящий от MCP, не может перейти в `committed`, пока зависимость не разрешена. Speaker может заранее подготовить только независимые фрагменты либо несколько отменяемых кандидатов.

---

## 8. Scheduler двух потоков

Scheduler — один из центральных компонентов VOX-SYNAPSE.

Его задачи:

- запускать Planner и Speaker конкурентно;
- назначать приоритет следующего decode step;
- батчировать сессии, когда это не увеличивает задержку;
- запускать MCP сразу после валидного `tool_call`;
- не позволять Speaker пересечь неразрешённую dependency boundary;
- останавливать speculative generation после изменения плана;
- собирать точную временную трассу.

Начальная политика:

```text
1. Planner получает приоритет до intent/tool_call.
2. MCP запускается немедленно.
3. Speaker генерирует ready chunks.
4. Planner продолжает формировать dependency graph.
5. MCP result разблокирует factual chunks.
6. Speaker продолжает с новой версией плана.
```

Lookahead измеряется не числом текстовых токенов, а временем между semantic decision и моментом, когда соответствующая информация достигает commit horizon голосового потока.

---

## 9. Commit Horizon и перепланирование

```text
PLANNED ─► GENERATING ─► BUFFERED ─► COMMITTED ─► PLAYED
    ▲            │           │
    └────────────┴───────────┘
        можно отменить/заменить
```

Правила:

1. Уже воспроизведённая речь считается необратимой частью контекста.
2. `BUFFERED` аудио или текст можно отменить при новом MCP result.
3. Tool-dependent факт запрещено переводить в `COMMITTED` до результата.
4. Ветвление выполняется на безопасной границе фразы или акустического сегмента.
5. При barge-in Speaker останавливается, буфер очищается, а MCP отменяется или сохраняется согласно политике инструмента.
6. Новый план всегда учитывает то, что пользователь уже услышал.

---

## 10. Уровень B: Hidden-State Semantic Bridge

После стабильного event baseline Planner дополнительно передаёт состояния выбранного слоя:

\[
H_p \in \mathbb{R}^{T_p \times d}
\]

Обучаемый projector преобразует их в ограниченный semantic memory:

\[
Z_p = P(H_p)
\]

Speaker использует память через gated cross-attention:

\[
H_s' = H_s + g \cdot \operatorname{CrossAttention}(H_s, Z_p)
\]

где `g` — обучаемый gate, позволяющий постепенно включать новый канал без разрушения поведения pretrained backbone.

На первом эксперименте:

- backbone заморожен;
- обучаются projector, cross-attention и gate;
- при необходимости добавляется LoRA в ограниченное число слоёв;
- structured events продолжают управлять MCP и factual commitment;
- hidden bridge отвечает за более богатую семантическую передачу, но не является единственным каналом безопасности.

Обязательная ablation-проверка:

| Вариант | Канал между потоками |
| --- | --- |
| A | Только structured events |
| B | Structured events + projected hidden states |
| C | Только projected hidden states, исследовательский контроль |

Hidden bridge сохраняется только при измеримом выигрыше по задержке, связности или качеству продолжения.

---

## 11. Уровень C: Voice Head и аудиокодек

Основной кандидат аудиокодека — **Kyutai Mimi**:

- потоковая работа с 24 kHz mono audio;
- частота представления 12.5 Hz;
- несколько residual codebooks;
- первый codebook содержит усиленную семантическую информацию;
- доступна открытая PyTorch-реализация.

Voice Head не является одним linear classifier над объединённым словарём. Он должен учитывать межкодбуковые зависимости:

```text
Speaker hidden states
        +
semantic memory
        │
        ▼
Temporal Voice Transformer
        │
        ▼
Depth / Codebook Transformer
        │
        ▼
Mimi codebook tokens
        │
        ▼
Streaming Mimi decoder → PCM
```

Исследуемые варианты:

- сначала предсказывать semantic/first codebook, затем acoustic codebooks;
- delayed/interleaved codebook pattern;
- отдельный небольшой Depth Transformer;
- teacher forcing по токенам референсного аудио;
- conditioning на structured plan и projected hidden states.

На переходном эксперименте Speaker может генерировать текст. Это допустимо только как baseline Уровня A/B и не считается доказательством нативного speech-to-speech.

---

## 12. Async MCP Engine

MCP Engine работает как отдельный асинхронный контур:

1. получает валидированный `tool_call` от Planner;
2. фиксирует время решения и фактического старта;
3. исполняет инструмент параллельно с Speaker;
4. поддерживает timeout и cancellation;
5. возвращает типизированный `fact` или `tool_error`;
6. разблокирует зависимые chunks;
7. инициирует `replan`, если результат расходится со speculative plan.

Стратегии ожидания:

- silent fast path;
- безопасный semantic lead-in;
- короткий filler;
- явное сообщение о долгом ожидании;
- обработка ошибки без выдумывания результата.

---

## 13. Технологический стек

### 13.1. Ядро

| Компонент | Выбор |
| --- | --- |
| Язык | Python 3.12 |
| Deep learning | PyTorch |
| Backbone API | Hugging Face Transformers |
| Модель | Qwen3-1.7B; контрольный вариант Qwen3-4B |
| Fine-tuning | PEFT / LoRA |
| Веса и checkpoints | Safetensors |
| Аудио | torchaudio + Mimi |
| Асинхронность | asyncio, TaskGroup, bounded Queue |
| MCP | официальный MCP Python SDK |
| Схемы событий | Pydantic |
| Конфигурация | dataclasses + YAML; Hydra только при росте матрицы экспериментов |
| Метрики обучения | TensorBoard; W&B как опциональный адаптер |
| Тестирование | pytest, pytest-asyncio |
| Формат трасс | JSONL для отладки, Parquet для анализа |

### 13.2. Осознанно не используем в ядре PoC

- закрытые LLM/ASR/TTS API;
- LangChain/LangGraph как scheduler;
- vLLM до стабилизации кастомного forward pass;
- микросервисную декомпозицию;
- Kafka, Celery и Kubernetes;
- обязательную сетевую realtime-инфраструктуру на первых модельных экспериментах.

Первый runtime работает на подготовленных текстовых и WAV-сценариях с симулируемыми задержками. Realtime transport подключается после корректной работы scheduler и voice stream.

---

## 14. Структура проекта

```text
vox/
├── pyproject.toml
├── configs/
│   ├── model/
│   ├── experiments/
│   └── tools/
├── src/vox/
│   ├── domain/
│   │   ├── events.py
│   │   ├── chunks.py
│   │   └── timeline.py
│   ├── model/
│   │   ├── backbone.py
│   │   ├── sessions.py
│   │   ├── semantic_bridge.py
│   │   └── voice_head.py
│   ├── runtime/
│   │   ├── scheduler.py
│   │   ├── planner.py
│   │   ├── speaker.py
│   │   └── commit_buffer.py
│   ├── audio/
│   │   ├── codec.py
│   │   ├── streams.py
│   │   └── playback.py
│   ├── tools/
│   │   ├── mcp_client.py
│   │   └── registry.py
│   ├── training/
│   │   ├── datasets.py
│   │   ├── losses.py
│   │   └── trainer.py
│   └── telemetry/
│       ├── trace.py
│       └── metrics.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── scenarios/
└── experiments/
    ├── notebooks/
    └── reports/
```

---

## 15. Этапы реализации

### Этап 0 — Sequential Baseline

```text
User text → Planner completes → MCP completes → Speaker completes
```

Цель: получить контрольные latency, tool accuracy и качество ответа на одном backbone.

### Этап 1 — Dual Session Runtime

```text
                    ┌─► Planner session ─► MCP
User text ──────────┤                       │
                    └─► Speaker session ◄───┘
```

Результат этапа:

- один загруженный backbone;
- две независимые KV-сессии;
- semantic event protocol;
- dependency graph;
- конкурентный scheduler;
- synthetic fast/slow/error MCP;
- полный timeline каждого turn.

### Этап 2 — Hidden-State Bridge

- извлечение Planner hidden states;
- projector и gated cross-attention;
- обучение новых модулей на синтетических и размеченных парах;
- ablation A/B/C;
- решение о полезности нейронного моста.

### Этап 3 — Native Voice Head

- токенизация референсного аудио через Mimi;
- обучение temporal/depth audio decoder;
- генерация первого и остальных codebooks;
- streaming decode в PCM;
- conditioning от Planner.

### Этап 4 — Realtime Interaction

- аудиовход;
- VAD и barge-in;
- playback buffer;
- измерение TTFA и stop latency;
- полный speech-to-speech сценарий.

### Этап 5 — Consolidation

Если две сессии создают лишнюю вычислительную стоимость, проверить объединение в один shared forward с двумя heads. Переход выполняется только после сохранения воспроизводимого baseline.

---

## 16. Данные и обучение

### 16.1. Данные Уровня A

Синтетические и вручную проверенные сценарии:

- запрос пользователя;
- ранний intent;
- tool call и arguments;
- safe speech plans;
- dependency graph;
- tool result/error;
- корректное продолжение;
- временной профиль инструмента.

### 16.2. Данные Уровня B

Пары:

```text
Planner context/hidden states
        → target Speaker continuation
```

Обучение сначала проводится с frozen backbone. Добавление LoRA разрешено после baseline новых модулей.

### 16.3. Данные Уровня C

- речь с текстовой/семантической разметкой;
- Mimi codebook tokens;
- границы слов и фраз;
- паузы и filler;
- переходы до и после tool result;
- перебивания;
- разные задержки MCP.

Оценка объёма данных определяется learning curves. Число часов заранее не считается гарантией достаточности.

### 16.4. Функция потерь

Общая исследовательская форма:

\[
\mathcal{L}_{total} =
\lambda_s \mathcal{L}_{semantic} +
\lambda_t \mathcal{L}_{tool} +
\lambda_v \mathcal{L}_{voice} +
\lambda_b \mathcal{L}_{bridge} +
\lambda_c \mathcal{L}_{commit}
\]

Конкретные компоненты и веса добавляются поэтапно. На каждом этапе оптимизируются только необходимые для текущей гипотезы параметры.

---

## 17. Метрики

### 17.1. Архитектурные метрики

| Метрика | Начальная цель |
| --- | ---: |
| Tool-call accuracy | `> 95%` на ограниченном наборе инструментов |
| Invalid semantic events | `< 1%` после constrained validation |
| Tool-start advantage | Planner запускает MCP раньше sequential baseline |
| Unsupported factual commitment | `0` в тестовом наборе |
| Dependency violations | `0` |
| MCP continuation success | `> 90%` |
| Semantic lookahead | положительный и стабильный на tool-сценариях |
| Dual-stream latency gain | измеримый выигрыш относительно sequential baseline |

### 17.2. Нейронные метрики

- improvement hidden bridge относительно event-only baseline;
- consistency между semantic plan и Speaker output;
- качество continuation после смены плана;
- число токенов/миллисекунд, отменённых после MCP result;
- дополнительная VRAM и latency projector/cross-attention;
- качество и стабильность Mimi codebook generation.

### 17.3. Realtime-метрики позднего этапа

| Метрика | Цель PoC |
| --- | ---: |
| Time-to-First-Audio | `< 500–700 мс`, затем оптимизация |
| Stop-on-barge-in | `< 150 мс` |
| Audible splice rate | оценивается слепым тестом |
| Commit horizon | измеряется в миллисекундах |
| Subjectively hidden MCP latency | сравнение с baseline |

Долгосрочная цель `TTFA < 200–280 мс` сохраняется, но не используется как критерий первых модельных этапов.

---

## 18. Обязательные сценарии

1. Ответ без инструмента.
2. Быстрый MCP без filler.
3. Медленный MCP с безопасным lead-in.
4. Медленный MCP с коротким filler.
5. Ошибка или timeout MCP без выдуманного результата.
6. Результат, противоречащий speculative continuation.
7. Перебивание во время safe chunk.
8. Перебивание во время ожидания инструмента.

Каждый сценарий должен иметь deterministic fixture, event trace и ожидаемые dependency transitions.

---

## 19. Основные риски

| Риск | Проверка / снижение риска |
| --- | --- |
| 1.7B не соблюдает event schema | constrained parsing, fine-tuning; сравнение с 4B |
| Две сессии слишком медленны | dynamic batching, priority scheduler, shared weights |
| Planner не опережает Speaker | ранний action objective и короткий event protocol |
| Hidden states не помогают Speaker | обязательная ablation; удаление bridge при отсутствии выигрыша |
| Cross-attention разрушает pretrained поведение | frozen backbone и gated residual connection |
| Speaker произносит неподтверждённые факты | dependency graph и commit validator вне модели |
| Генерация всех codebooks слишком тяжёлая | hierarchical/depth decoder и поэтапный codec objective |
| Акустические переходы слышны | ветвление на безопасных границах и обучение transition examples |
| MCP возвращается после commit horizon | safe lead-in, blocked chunks и replanning |

---

## 20. Критерии решений

После каждого уровня принимается явное решение:

1. **Continue:** гипотеза подтверждена — перейти к следующему уровню.
2. **Refine:** механизм работает, но не достигает метрик — изменить scheduler, protocol или training objective.
3. **Fallback:** сложный механизм не выигрывает у простого baseline — сохранить более простую архитектуру.
4. **Pivot:** dual-stream не даёт измеримого преимущества — пересмотреть концепцию на основании трасс и тестов.

Финальная архитектура определяется результатами A/B и ablation-тестов, а не заранее выбранной красотой схемы.

---

## 21. Определение результата PoC

Минимальный доказательный результат:

> Один открытый backbone, загруженный один раз, обслуживает Planner и Speaker как две конкурентные decoding-сессии. Planner заранее формирует валидный MCP-вызов и semantic plan. Speaker начинает безопасный ответ до завершения инструмента, затем использует результат только в незафиксированной части. Полная временная трасса показывает положительный semantic lookahead и выигрыш относительно последовательного baseline.

Полный доказательный результат:

> Structured events и обучаемый hidden-state bridge направляют нативный Voice Head, который потоково генерирует токены аудиокодека, поддерживает MCP-aware commitment и корректно реагирует на перебивание.

---

## 22. Исходные технические ориентиры

- [Qwen3-1.7B model card](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)
- [Kyutai Moshi and Mimi](https://github.com/kyutai-labs/moshi)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
