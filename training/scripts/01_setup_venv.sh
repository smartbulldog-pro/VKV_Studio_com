#!/usr/bin/env bash
# ============================================================
# Шаг 1: Создание виртуального окружения с PyTorch CUDA
# ============================================================
# CachyOS, Python 3.14, RTX 4080 Laptop 12GB, CUDA 13.2
# ============================================================
set -euo pipefail

TRAINING_DIR="c:/projects/VKVstudio/training"
VENV_DIR="$TRAINING_DIR/.venv"

echo "╔══════════════════════════════════════════════╗"
echo "║   Шаг 1: Виртуальное окружение + PyTorch     ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Создаём venv ─────────────────────────────────────────────
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Venv уже существует: $VENV_DIR"
    echo "   Удалить и создать заново? (y/n)"
    read -r REPLY
    if [ "$REPLY" = "y" ]; then
        rm -rf "$VENV_DIR"
    else
        echo "Используем существующий."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Создаём venv..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Venv создан: $VENV_DIR"
fi

# ── Активируем ───────────────────────────────────────────────
echo ""
echo "📦 Активируем venv..."
source "$VENV_DIR/bin/activate"
echo "✅ Python: $(python --version) @ $(which python)"

# ── Обновляем pip ────────────────────────────────────────────
echo ""
echo "📦 Обновляем pip..."
pip install --upgrade pip setuptools wheel

# ── Устанавливаем PyTorch с CUDA ─────────────────────────────
echo ""
echo "📦 Устанавливаем PyTorch с CUDA..."
echo "   (Это ~2.5GB, может занять несколько минут)"
echo ""
# PyTorch nightly для CUDA 13.2 + Python 3.14
# Сначала пробуем stable, если нет — nightly
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# ── Проверяем ────────────────────────────────────────────────
echo ""
echo "🔍 Проверяем PyTorch + CUDA..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    print(f'BF16: {torch.cuda.is_bf16_supported()}')
    print('✅ PyTorch + CUDA работает!')
else:
    print('❌ CUDA не доступна! Проверь установку.')
    exit(1)
"

echo ""
echo "═══════════════════════════════════════════════"
echo "  Шаг 1 ГОТОВ. Следующий: bash scripts/02_install_unsloth.sh"
echo "═══════════════════════════════════════════════"
