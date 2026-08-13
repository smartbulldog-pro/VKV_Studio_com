# Synapse v2 — Fine-Tuning Guide

> Свежие правила дообучения под **датасет v2** (`.system/datasets/synapse_v2.jsonl`, 2592 строки, EN/RU 50/50).
> Написано Opus 4.8, 2026-07-05, взамен старого брифинга. Всё сверено с реальными скриптами в `training/scripts/` и с тем, как модель развёрнута в проде (см. §6 + memory `synapse-model-swap-contract`).
> Цель прогона: превратить Synapse из «помощника веб-дизайнера» в **сеньора-AI-архитектора с характером**, сохранив базовые знания Gemma 4.

---

## 0. TL;DR (что реально делать сегодня)

```bash
cd C:/projects/VKVstudio/training
# venv уже готов — в нём unsloth 2026.6.1 + torch 2.10 + CUDA. НЕ переставляем.
source .venv-cuda/Scripts/activate      # или .\.venv-cuda\Scripts\Activate.ps1

# 1. Логин в HF (нужен твой токен + принятая лицензия Gemma) — см. §2
# 2. Скачать базу (E4B основная, E2B — лёгкая для роутинга) — см. §2
# 3. Обучить:  python scripts/04_train.py --model E4B
# 4. Экспорт в GGUF (Q8_0):  python scripts/05_export_gguf_win.py --model E4B
# 5. Подменить прод-файл и перезапустить сервер — см. §6
```

Обе модели: повторить шаги 3–4 с `--model E2B`.

---

## 1. Что уже готово (не трогаем)

- **`training/.venv-cuda/`** — рабочее окружение: `unsloth 2026.6.1`, `torch 2.10.0+cu126`, CUDA доступна. Переустановка не нужна и вредна.
- **`training/chat_template.jinja`** — подтверждённый Gemma-4 `<|turn|>` шаблон (пригодится, если понадобится ручная сборка — см. gotcha §5.1).
- **`training/scripts/`** — пайплайн `00_check_env` → `06_deploy_ollama`. `04_train.py` и `05_export` обновлены под v2.
- **Датасет** — `.system/datasets/synapse_v2.jsonl`. Схема: одна строка = `{"messages":[{system},{user},{assistant},...]}`, system-промпт одинаковый на каждой строке (канон — `SYSTEM_PROMPT` в `.system/datasets/_build_dedup.py`).

---

## 2. Базовая модель (HF, gated — нужен токен)

QLoRA обучает **HF safetensors**, а НЕ Ollama-GGUF. Обе базы gated:

1. Принять лицензию под своим HF-аккаунтом: `google/gemma-4-E4B-it` и `google/gemma-4-E2B-it`.
2. Токен: https://huggingface.co/settings/tokens (read).
3. Логин: `python -c "from huggingface_hub import login; login('ТОКЕН')"`.
4. Скачать в `training/models/`:
   ```bash
   bash scripts/03_download_model.sh E4B     # → training/models/gemma-4-E4B-it
   bash scripts/03_download_model.sh E2B     # → training/models/gemma-4-E2B-it
   ```
   (или `huggingface-cli download google/gemma-4-E4B-it --local-dir training/models/gemma-4-E4B-it`)

**VRAM (RTX 4080 Laptop, 12 ГБ):** E2B QLoRA ~8 ГБ, E4B QLoRA ~10 ГБ. E4B влезает, но впритык — если OOM, снизь `MAX_SEQ_LENGTH` до 1536 или `BATCH_SIZE` до 1.

---

## 3. Гиперпараметры и **почему именно так** (это и есть «правила»)

Датасет — 2592 качественных примера, ~50% факты/знания, ~50% персона/навыки. Это меняет расчёт против v1 (1024 строки, вышел недоучен по персоне).

| Параметр | E4B | E2B | Обоснование |
|---|---|---|---|
| LoRA `r` | **16** | **16** | 2592 примера несут заметный факт-объём → ранг 8 недоберёт знания. 16 — мейнстрим-дефолт Unsloth, ёмкости хватает, при этом не заучивает дословно. 32 — только если после прогона персона всё ещё слабая. |
| `lora_alpha` | 32 | 32 | `alpha = 2·r` — привычный масштаб; при r=16 даёт стабильный градиент. |
| `lora_dropout` | 0 | 0 | Unsloth оптимизирует dropout=0; регуляризацию берём из weight_decay + ранней остановки. |
| target_modules | все 7 | все 7 | `q,k,v,o,gate,up,down` — полный attention+MLP, иначе персона «не прилипает». |
| эпохи | **2** | **2** | На 2592 строках 3 эпохи для 2–4B модели уже на грани заучивания. Стартуем с 2, смотрим loss (§4). |
| LR | 2e-4 | **1e-4** | E2B чувствительнее (меньше параметров) — половинный LR бережёт базовые знания. cosine + warmup 5–8%. |
| effective batch | 8 (2×4) | 8 (2×4) | `per_device=2`, `grad_accum=4`. Если E4B в OOM → `per_device=1, grad_accum=8` (тот же eff. batch). |
| `max_seq_length` | 2048 | 2048 | Наши строки < ~1024 токенов; 2048 с запасом. `packing=True` для эффективности. |
| optim / precision | adamw_8bit / bf16 | то же | bf16 нативен для Ada (4080); 8-bit Adam экономит VRAM. |

**Против переобучения** (ты просил взвесить это): датасет уже сбалансирован по темам и не имеет частой «фирменной фразы» (>1.1%), так что главный рычаг — **эпохи и loss**, а не веса категорий. Правило: 2 эпохи по умолчанию, третью — только если на held-out смоук-тесте (§4) персона слабая, а loss ещё высок.

---

## 4. Прогон и контроль качества

```bash
python scripts/04_train.py --model E4B     # ~15–30 мин на 4080
```

**Что смотреть в логах:**
- `train_loss` должен плавно падать и **осесть в районе 0.5–0.9**. Если ушёл < 0.3 — модель зубрит датасет (переобучение): снизь эпохи/LR.
- Если loss скачет/растёт — LR великоват или OOM-своп.

**Смоук-тест встроен в конец `04_train.py`** — промпты уже v2 (архитектор + границы + отказы), НЕ старые «про Astro». Оцени руками: тон (тёплый, самоирония), корректность фактов, отказ на инъекцию/оффтоп, RU и EN. Если ответы «веб-дизайнерские» или сухие — персона недоучена (↑ эпоху или r=32).

---

## 5. Две засады Gemma-4 (проверить обязательно)

1. **chat_template отдельным файлом.** У Gemma-4 `tokenizer_config.json.chat_template` пустой, шаблон лежит в `chat_template.jinja` (HF #45205). Unsloth через `FastLanguageModel`/`AutoProcessor` подхватывает сам. Если грузишь токенайзер вручную — `hf_hub_download(..., "chat_template.jinja")` и присвой `tokenizer.chat_template`, иначе `apply_chat_template` вернёт пусто. (Готовый эталон шаблона лежит в `training/chat_template.jinja`.)
2. **Порча EOS при merge.** Unsloth #5386: после слияния адаптера `eos_token` может стать `<eos>` вместо `<turn|>` → генерация не останавливается. **После экспорта проверь `generation_config.json`/EOS в GGUF.** В проде это уже подстраховано: `inference/config.py` → `LLM_STOP_TOKENS = ["<turn|>", "<eos>"]` — оставь оба.

Обе issue были открыты на 2026-07-04 — переепроверь актуальность перед прогоном.

---

## 6. Экспорт и подмена прода (swap-контракт)

```bash
python scripts/05_export_gguf_win.py --model E4B --quant Q8_0
```

Прод-сервер (`inference/`, llama-cpp-python) грузит **один файл по жёсткому пути**:
`inference/config.py:19` → `LLM_MODEL_PATH = MODELS_DIR / "synapse-q8.gguf"` (`MODELS_DIR = inference/models`).

**Подмена без правки кода:** положи новый GGUF (Q8_0) в **`inference/models/synapse-q8.gguf`** (то же имя) → перезапусти сервер (или `POST /api/session/close`, следующий запрос перегрузит). Если хочешь держать обе модели рядом — дай разные имена и правь `LLM_MODEL_PATH` (или заведи env-переключатель).

⚠️ **Важно после подмены:** `inference/config.py` содержит СВОЙ `SYSTEM_PROMPT` (ещё v1). Модель v2 обучена на каноничном промпте из `_build_dedup.py`. Приведи `config.py:SYSTEM_PROMPT` в соответствие тому, что вшито в датасет, иначе на инференсе модель получит незнакомый системный промпт. Комментарий «Gemma 4 E2B» в конфиге тоже обнови, если кладёшь E4B.

**Роутинг E2B/E4B** (твоя идея — лёгкие темы на маленькую): держи оба GGUF в `inference/models/`, а выбор модели по сложности запроса делаем на стороне сервера отдельным шагом (не входит в этот прогон — сначала обучаем обе).

Старый прод-файл `synapse-q8.gguf` (v1) НЕ удалён намеренно — это fallback, пока v2 не проверен. Снести после успешного свапа.

---

## 7. Чек-лист «готово»

- [ ] HF-логин + лицензии приняты
- [ ] `training/models/gemma-4-E4B-it` (и `-E2B-it`) скачаны
- [ ] `04_train.py --model E4B` прошёл, `train_loss` в 0.5–0.9
- [ ] смоук-тест: тон/факты/RU+EN/отказы — ок
- [ ] EOS в GGUF = `<turn|>` (gotcha §5.2)
- [ ] `config.py:SYSTEM_PROMPT` синхронизирован с датасетом
- [ ] новый GGUF в `inference/models/synapse-q8.gguf`, сервер перезапущен
- [ ] то же для E2B
- [ ] fallback-модель снесена после проверки
