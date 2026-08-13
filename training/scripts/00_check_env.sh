#!/usr/bin/env bash
# ============================================================
# Synapse Fine-Tune — Environment Diagnostic
# Проверяет всё необходимое для QLoRA fine-tune Gemma 4 E4B
# ============================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Synapse Fine-Tune — Environment Check      ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── 1. OS ────────────────────────────────────────────────────
echo -e "${CYAN}[1/10] OS${NC}"
if [ -f /etc/os-release ]; then
    source /etc/os-release
    echo -e "  ${GREEN}✓${NC} $PRETTY_NAME"
else
    echo -e "  ${YELLOW}? Unknown OS${NC}"
fi
echo ""

# ── 2. Kernel ────────────────────────────────────────────────
echo -e "${CYAN}[2/10] Kernel${NC}"
echo -e "  $(uname -r)"
echo ""

# ── 3. NVIDIA Driver ────────────────────────────────────────
echo -e "${CYAN}[3/10] NVIDIA Driver${NC}"
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
    echo -e "  ${GREEN}✓${NC} Driver: $DRIVER_VER"
    echo -e "  ${GREEN}✓${NC} GPU: $GPU_NAME"
    echo -e "  ${GREEN}✓${NC} VRAM Total: $GPU_MEM"
    echo -e "  ${GREEN}✓${NC} VRAM Used: $GPU_USED"
else
    echo -e "  ${RED}✗ nvidia-smi NOT FOUND${NC}"
    echo -e "  ${RED}  Install: sudo pacman -S nvidia nvidia-utils${NC}"
fi
echo ""

# ── 4. CUDA Toolkit ──────────────────────────────────────────
echo -e "${CYAN}[4/10] CUDA Toolkit${NC}"
if command -v nvcc &>/dev/null; then
    CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo -e "  ${GREEN}✓${NC} nvcc: CUDA $CUDA_VER"
else
    echo -e "  ${YELLOW}⚠ nvcc NOT FOUND (may be OK — PyTorch bundles its own CUDA)${NC}"
fi

# Check CUDA runtime from nvidia-smi
if command -v nvidia-smi &>/dev/null; then
    CUDA_DRIVER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | sed 's/.*CUDA Version: //' | sed 's/ .*//')
    echo -e "  ${GREEN}✓${NC} CUDA (driver-supported): $CUDA_DRIVER"
fi
echo ""

# ── 5. Python ────────────────────────────────────────────────
echo -e "${CYAN}[5/10] Python${NC}"
for PY in python3 python; do
    if command -v $PY &>/dev/null; then
        PY_VER=$($PY --version 2>&1)
        PY_PATH=$(which $PY)
        echo -e "  ${GREEN}✓${NC} $PY_VER ($PY_PATH)"
    fi
done
echo ""

# ── 6. pip / venv ────────────────────────────────────────────
echo -e "${CYAN}[6/10] pip / venv${NC}"
if command -v pip3 &>/dev/null || command -v pip &>/dev/null; then
    PIP_VER=$(pip3 --version 2>/dev/null || pip --version 2>/dev/null)
    echo -e "  ${GREEN}✓${NC} $PIP_VER"
else
    echo -e "  ${RED}✗ pip NOT FOUND${NC}"
fi
echo ""

# ── 7. PyTorch + CUDA ────────────────────────────────────────
echo -e "${CYAN}[7/10] PyTorch${NC}"
python3 -c "
import sys
try:
    import torch
    print(f'  \033[0;32m✓\033[0m PyTorch {torch.__version__}')
    print(f'  \033[0;32m✓\033[0m CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  \033[0;32m✓\033[0m CUDA version (PyTorch): {torch.version.cuda}')
        print(f'  \033[0;32m✓\033[0m GPU from PyTorch: {torch.cuda.get_device_name(0)}')
        print(f'  \033[0;32m✓\033[0m VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        # BF16 support
        print(f'  \033[0;32m✓\033[0m BF16 support: {torch.cuda.is_bf16_supported()}')
    else:
        print(f'  \033[0;31m✗ CUDA NOT available in PyTorch!\033[0m')
        print(f'  \033[0;31m  This means PyTorch was installed without GPU support.\033[0m')
except ImportError:
    print(f'  \033[0;31m✗ PyTorch NOT installed\033[0m')
" 2>/dev/null || echo -e "  ${RED}✗ Python error running PyTorch check${NC}"
echo ""

# ── 8. Key packages ──────────────────────────────────────────
echo -e "${CYAN}[8/10] Key ML Packages${NC}"
python3 -c "
packages = [
    ('transformers', 'transformers'),
    ('bitsandbytes', 'bitsandbytes'),
    ('peft', 'peft'),
    ('trl', 'trl'),
    ('datasets', 'datasets'),
    ('accelerate', 'accelerate'),
    ('unsloth', 'unsloth'),
    ('triton', 'triton'),
]
for name, pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', '?')
        print(f'  \033[0;32m✓\033[0m {name}: {ver}')
    except ImportError:
        print(f'  \033[1;33m⚠ {name}: NOT installed\033[0m')
" 2>/dev/null || echo -e "  ${RED}✗ Python error${NC}"
echo ""

# ── 9. Ollama ────────────────────────────────────────────────
echo -e "${CYAN}[9/10] Ollama${NC}"
if command -v ollama &>/dev/null; then
    OLLAMA_VER=$(ollama --version 2>/dev/null || echo "unknown")
    echo -e "  ${GREEN}✓${NC} ollama: $OLLAMA_VER"
    echo -e "  ${CYAN}  Installed models:${NC}"
    ollama list 2>/dev/null | head -10 || echo "  (could not list)"
else
    echo -e "  ${YELLOW}⚠ ollama NOT installed${NC}"
fi
echo ""

# ── 10. Disk Space ───────────────────────────────────────────
echo -e "${CYAN}[10/10] Disk Space${NC}"
TRAINING_DIR="c:/projects/VKVstudio/training"
DISK_FREE=$(df -h "$TRAINING_DIR" 2>/dev/null | tail -1 | awk '{print $4}')
echo -e "  Free space (training dir): $DISK_FREE"
echo -e "  ${YELLOW}Need: ~20GB free (model weights + training checkpoints)${NC}"
echo ""

# ── Summary ──────────────────────────────────────────────────
echo -e "${CYAN}══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  SUMMARY${NC}"
echo -e "${CYAN}══════════════════════════════════════════════${NC}"
echo ""
echo -e "  Save this output and send it to the orchestrator."
echo -e "  We'll determine next steps based on what's installed."
echo ""
