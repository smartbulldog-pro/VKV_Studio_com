#!/usr/bin/env bash
# ============================================================
# Шаг 5: Экспорт в GGUF для Ollama
# ============================================================
set -euo pipefail

TRAINING_DIR="c:/projects/VKVstudio/training"
VENV_DIR="$TRAINING_DIR/.venv"
LORA_DIR="$TRAINING_DIR/output/synapse-lora"
GGUF_DIR="$TRAINING_DIR/output/synapse-gguf"

echo "╔══════════════════════════════════════════════╗"
echo "║   Шаг 5: Экспорт LoRA → GGUF                ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

source "$VENV_DIR/bin/activate"

# Проверяем что LoRA существует
if [ ! -d "$LORA_DIR" ]; then
    echo "❌ LoRA адаптер не найден: $LORA_DIR"
    echo "   Сначала запусти 04_train.py"
    exit 1
fi

mkdir -p "$GGUF_DIR"

echo "📦 Экспортируем в GGUF (Q4_K_M)..."
echo "   Это объединит LoRA с базовой моделью и квантует."
echo "   Может занять 5-10 минут и ~16GB RAM."
echo ""

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -c "
from unsloth import FastLanguageModel
import os

LORA_DIR = '$LORA_DIR'
GGUF_DIR = '$GGUF_DIR'
BASE_MODEL_DIR = 'c:/projects/VKVstudio/training/models/gemma-4-E2B-it' # 👈 Твой путь к базе

print('📦 Загружаю LoRA модель и базу локально...')
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_DIR,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=False, # 👈 Важно: для экспорта GGUF лучше грузить в FP16, а не в 4-bit
    local_files_only=True, # 👈 Отрезаем сеть
)

print()
print('📦 Экспортирую GGUF Q4_K_M...')
model.save_pretrained_gguf(
    GGUF_DIR,
    tokenizer,
    quantization_method='q4_k_m',
)

print()
print('✅ GGUF экспортирован!')
print(f'   Файлы:')
for f in sorted(os.listdir(GGUF_DIR)):
    size = os.path.getsize(os.path.join(GGUF_DIR, f)) / 1024 / 1024
    print(f'   {f}: {size:.1f} MB')
"

echo ""
echo "═══════════════════════════════════════════════"
echo "  Шаг 5 ГОТОВ. Следующий: bash scripts/06_deploy_ollama.sh"
echo "═══════════════════════════════════════════════"
