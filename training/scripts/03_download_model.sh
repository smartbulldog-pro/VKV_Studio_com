#!/usr/bin/env bash
# ============================================================
# Шаг 3: Скачивание базовой модели Gemma 4 (E2B | E4B)
# ============================================================
# Использование:  bash scripts/03_download_model.sh E4B
#                 bash scripts/03_download_model.sh E2B
# Требует: принятую лицензию Gemma + HF-логин (см. README §2).
# Качает ПОЛНЫЕ safetensors в training/models/ (04_train.py грузит локально).
# ============================================================
set -euo pipefail

MODEL="${1:-E4B}"
case "$MODEL" in E2B|E4B) ;; *) echo "arg must be E2B or E4B"; exit 1;; esac

TRAINING_DIR="c:/projects/VKVstudio/training"
VENV_DIR="$TRAINING_DIR/.venv-cuda"
REPO="google/gemma-4-${MODEL}-it"
DEST="$TRAINING_DIR/models/gemma-4-${MODEL}-it"

echo "╔══════════════════════════════════════════════╗"
echo "║   Шаг 3: Скачивание $REPO"
echo "╚══════════════════════════════════════════════╝"

source "$VENV_DIR/Scripts/activate"
mkdir -p "$DEST"

echo "📦 snapshot_download → $DEST"
echo "   (gated: нужен HF-логин; ~5-6GB E2B / ~9-10GB E4B)"
python -c "
from huggingface_hub import snapshot_download
p = snapshot_download(
    repo_id='$REPO',
    local_dir=r'$DEST',
    allow_patterns=['*.safetensors','*.json','*.jinja','*.model','tokenizer*'],
)
print('✅ Скачано в', p)
"

echo ""
echo "═══════════════════════════════════════════════"
echo "  Шаг 3 ГОТОВ. Дальше: python scripts/04_train.py --model $MODEL"
echo "═══════════════════════════════════════════════"
