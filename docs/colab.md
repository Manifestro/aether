# AETHER — запуск real-model экспериментов в Google Colab

Три notebook'а, по нарастающей сложности:

1. [`notebooks/aether_stage1_colab.ipynb`](../notebooks/aether_stage1_colab.ipynb) — smoke test:
   один backend, сериализованные вызовы `generate`, проверка совместимости модели/окружения и
   базового Planner→MCP→Speaker сценария.
2. [`notebooks/aether_stage2_colab.ipynb`](../notebooks/aether_stage2_colab.ipynb) — interleaved
   token decoding: один набор весов, два независимых decode state (Planner/Speaker), scheduler с
   политикой Speaker 3 / Planner 2. Здесь впервые измеряется semantic lookahead.
3. [`notebooks/aether_stage3_colab.ipynb`](../notebooks/aether_stage3_colab.ipynb) — latency sweep
   (3000/1500/750/300 мс) × 3 сценария (успешный tool call, падение tool call, реплика без tool
   call), с явным разделением hard safety checks и soft/latency-dependent observations.

Продуктовый API (Stage 4, `aether_api`) переехал в отдельный репозиторий
[`Manifestro/aether-api`](https://github.com/Manifestro/aether-api) вместе со своим notebook.

## Перед запуском

1. Запуш текущую ветку в GitHub.
2. Открой нужный notebook в Google Colab.
3. Выбери GPU runtime: **Runtime → Change runtime type → GPU**.
4. В первой исполняемой ячейке укажи `REPO_URL`/`BRANCH`.

Каждый notebook выполняет: клонирование репозитория → проверку GPU → установку проекта с
`[dev,ml]` extras → dependency-free тесты → явную загрузку `Qwen/Qwen3-1.7B` → эксперимент →
упаковку диагностических файлов в архив.

## Что прислать для анализа

После выполнения последней ячейки Colab скачает архив `aether-colab-stage{N}-logs.zip`, внутри:

- `report.json` — структурированный результат (у stage2/3 — с `checks`/`observations` или `proof`);
- `raw_generations.jsonl` (stage1/2) — сырой текст, сгенерированный каждой логической сессией;
- `tests.log`, `model_run.log`, `traceback.txt` (если была ошибка);
- сведения о Python, PyTorch, Transformers, GPU и CUDA.

Архив не должен содержать токены GitHub или Hugging Face. Не добавляй секреты в `REPO_URL` или
параметры запуска.

## Ограничения

- Stage 1 не доказывает параллельность — GPU-вызовы там сериализованы одним локом. Он проверяет
  только совместимость и корректность JSONL/dependency-логики.
- Stage 2/3 доказывают семантический lookahead на interleaved scheduler, но не физически
  одновременное исполнение CUDA-ядер (см. `docs/reports/technical_report_01.md` §7.3).
- Lookahead (`Speaker раньше tool.completed`) — измеримое, но не гарантированное свойство: оно
  зависит от tool latency и от скорости GPU (latency crossover, см.
  `docs/reports/technical_report_02.md` §6). Единственный жёсткий инвариант — факт никогда не
  озвучивается до подтверждения tool result; это проверяется в каждом прогоне отдельно от lookahead.
