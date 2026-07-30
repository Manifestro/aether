# VOX-SYNAPSE — запуск Stage 1 в Google Colab

Основной способ запуска — notebook [`notebooks/vox_stage1_colab.ipynb`](notebooks/vox_stage1_colab.ipynb).

## Перед запуском

1. Загрузи текущий проект в GitHub-репозиторий.
2. Открой notebook в Google Colab.
3. Выбери GPU runtime: **Runtime → Change runtime type → GPU**.
4. В первой исполняемой ячейке укажи URL репозитория.

Notebook выполнит:

1. клонирование репозитория;
2. проверку GPU;
3. установку проекта и ML extra;
4. запуск dependency-free тестов;
5. явную загрузку `Qwen/Qwen3-1.7B`;
6. weather-сценарий Planner → MCP → Speaker;
7. сохранение и упаковку диагностических файлов.

## Что прислать для анализа

После выполнения последней ячейки Colab скачает архив:

```text
vox-colab-stage1-logs.zip
```

Пришли этот архив целиком. В нём будут:

- `report.json`;
- `raw_generations.jsonl`;
- `traceback.txt`, если произошла ошибка;
- результат unit tests;
- сведения о Python, PyTorch, Transformers, GPU и CUDA;
- timeline Planner/MCP/Speaker;
- сырой текст, сгенерированный каждой логической сессией.

Архив не должен содержать токены GitHub или Hugging Face. Не добавляй секреты в URL репозитория или параметры запуска.

## Важное ограничение текущего эксперимента

Текущий Hugging Face backend загружает веса один раз, но сериализует вызовы `generate`. Поэтому этот smoke test проверяет:

- совместимость модели и окружения;
- соблюдение Planner JSONL schema;
- качество Speaker;
- корректность MCP dependencies;
- память и базовую latency.

Он ещё не доказывает одновременное GPU-декодирование двух KV-cache. Это следующий этап после анализа логов.
