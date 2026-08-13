#!/usr/bin/env bash
# ============================================================
# Шаг 2: Установка Unsloth + все ML зависимости
# ============================================================
set -euo pipefail

TRAINING_DIR="c:/projects/VKVstudio/training"
VENV_DIR="$TRAINING_DIR/.venv"

echo "╔══════════════════════════════════════════════╗"
echo "║   Шаг 2: Unsloth + ML пакеты                ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Активируем venv ──────────────────────────────────────────
source "$VENV_DIR/bin/activate"
echo "✅ Venv активирован: $(python --version)"

# ── Проверяем PyTorch CUDA ───────────────────────────────────
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'" 2>/dev/null || {
    echo "❌ PyTorch CUDA не доступна! Сначала запусти 01_setup_venv.sh"
    exit 1
}
echo "✅ PyTorch CUDA OK"
echo ""

# ── Устанавливаем Unsloth ────────────────────────────────────
echo "📦 Устанавливаем Unsloth (с зависимостями)..."
echo "   Это установит: transformers, peft, trl, bitsandbytes, accelerate, etc."
echo ""

pip install unsloth

# ── Дополнительные пакеты ────────────────────────────────────
echo ""
echo "📦 Доустанавливаем пакеты..."
pip install datasets sentencepiece protobuf

# ── Проверяем всё ────────────────────────────────────────────
echo ""
echo "🔍 Проверяем установку..."
python -c "
packages = {
    'torch': None,
    'unsloth': None,
    'transformers': None,
    'peft': None,
    'trl': None,
    'bitsandbytes': None,
    'accelerate': None,
    'datasets': None,
    'triton': None,
}
all_ok = True
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', '?')
        print(f'  ✅ {pkg}: {ver}')
    except ImportError:
        print(f'  ❌ {pkg}: NOT installed')
        all_ok = False

import torch
print(f'\n  GPU: {torch.cuda.get_device_name(0)}')
print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'  BF16: {torch.cuda.is_bf16_supported()}')

if all_ok:
    print('\n✅ Все пакеты установлены!')
else:
    print('\n⚠️  Некоторые пакеты отсутствуют.')
"

echo ""
echo "═══════════════════════════════════════════════"
echo "  Шаг 2 ГОТОВ. Следующий: bash scripts/03_download_model.sh"
echo "═══════════════════════════════════════════════"
