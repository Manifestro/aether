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
4. [`notebooks/aether_stage4_voice_colab.ipynb`](../notebooks/aether_stage4_voice_colab.ipynb) —
   Phase C (`docs/plan.md` §7), минимальный **структурный** probe Voice Head: один Mimi codebook,
   from-scratch и намеренно необученный Voice Head поверх Speaker-текста. Проверяет только одно —
   переносится ли `ChunkState`'s commit horizon на аудио без изменений (аудио-чанк не коммитится
   раньше подтверждённого факта, буферизованный синтез корректно отменяется при replan). Качество
   звучания, просодия и эмоции здесь не оцениваются и не являются критерием выхода — см.
   `aether/model/voice_head.py` и `aether/experiments/colab_stage4.py`.

Продуктовый API (`aether_api`, HTTP/FastAPI) переехал в отдельный репозиторий
[`Manifestro/aether-api`](https://github.com/Manifestro/aether-api) вместе со своим notebook — это
не то же самое, что Stage 4 выше (тот остаётся research-core).

Отдельно, вне нумерации этапов: [`notebooks/aether_spike_moshi_teacher_colab.ipynb`](../notebooks/aether_spike_moshi_teacher_colab.ipynb)
— разведка перед Stage 5 (обучение Voice Head на hidden state, учитель — Kyutai Moshi). Публичный
пакет `moshi` — full-duplex conversational loop, а не TTS-вызов "текст → токены"; какой именно вызов
даёт нам аудио-токены учителя для заданного текста, никем не проверено. Notebook не гадает конкретный
API, а печатает его реальную поверхность (методы, сигнатуры) и пробует несколько кандидатов —
результат нужен, чтобы спроектировать реальную интеграцию по фактам, а не по предположению.

## Перед запуском

1. Запуш текущую ветку в GitHub.
2. Открой нужный notebook в Google Colab.
3. Выбери GPU runtime: **Runtime → Change runtime type → GPU**.
4. В первой исполняемой ячейке укажи `REPO_URL`/`BRANCH`.

Каждый notebook выполняет: клонирование репозитория → проверку GPU → установку проекта
(`[dev,ml]` extras для stage1-3, `[dev,ml,audio]` для stage4) → dependency-free тесты → явную
загрузку `Qwen/Qwen3-1.7B` (и для stage4 — Mimi) → эксперимент → упаковку диагностических файлов
в архив.

**GPU для stage4:** тот же T4, что уже проверен на stage1-3, — Mimi компактен, а сам Voice Head
работает на CPU по умолчанию. A100 не обязателен для этого эксперимента.

## Что прислать для анализа

После выполнения последней ячейки Colab скачает архив `aether-colab-stage{N}-logs.zip`, внутри:

- `report.json` — структурированный результат (у stage2/3/4 — с `checks`/`observations` или `proof`);
- `raw_generations.jsonl` (stage1/2) — сырой текст, сгенерированный каждой логической сессией;
- `wav/*.wav` (stage4) — сырой аудио-выход через Mimi decoder; ожидаемо звучит как шум/артефакты
  (Voice Head необучен намеренно), это не свидетельствует о поломке эксперимента;
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
- Stage 4 не доказывает нативный speech-to-speech и не измеряет качество голоса — Mimi decoder
  реален (frozen, pretrained), но Voice Head, предсказывающий его токены, специально не обучен.
  Проверяется только структурный перенос commit horizon на аудио (§20 `spec.md`: Continue/Refine/
  Fallback/Pivot принимается по трассе этого прогона, не по звучанию). Обучение Voice Head,
  дополнительные codebooks (Depth Transformer) и просодия — следующие, отдельные шаги.
- **Известный сбой окружения (найден и исправлен в первом прогоне).** Extra `audio` (`moshi`)
  подтягивает новый `torch`, не трогая предустановленный в Colab `torchvision` — тот ломается по
  ABI (`operator torchvision::nms does not exist`), и это рушит совершенно не связанный с нами
  импорт `transformers.AutoModelForCausalLM` (через vision-утилиты), маскируясь под
  `ModuleNotFoundError: Could not import module 'Qwen3ForCausalLM'`. Ни разу не доходит до
  собственно Voice Head кода. Notebook уже переустанавливает `torch`/`torchvision`/`torchaudio`
  одной командой сразу после установки extras — если ошибка повторится, проверь именно
  совместимость этой тройки первой.
