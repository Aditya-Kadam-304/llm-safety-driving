#! /bin/bash
#SBATCH --time=23:59:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=128000
#SBATCH --partition=gpu

# ── activate conda environment ──
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate adi

# ── redirect temp files to local scratch (avoid filling /home NFS) ──
export TMPDIR="/tmp/$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/hub"

# ═══════════════════════════════════════════════════════════════
# sensor_code_14.py — Multi-Model LLM Comparison (thesis-ready)
# 3 models: Qwen2.5-7B, Qwen2.5-1.5B, Mistral-7B
# Each model: Full + 4 ablations × 5 iters = 1250 rows per model
# v14 adds: iteration column, RNG re-seeding, LLM pre-cage cols,
#           metadata.json, statistical analysis (CI + McNemar)
# Run with --seed 44
# ═══════════════════════════════════════════════════════════════

# ---- MODEL 1: Qwen2.5-7B-Instruct (7B, same-family baseline) ----
echo "===== MODEL 1: Qwen2.5-7B ====="
python sensor_code_14.py --model Qwen/Qwen2.5-7B-Instruct --mode final --iters 5 --seed 44
ABLATE_DISABLE_LLM=1 python sensor_code_14.py --model Qwen/Qwen2.5-7B-Instruct --mode ablate --iters 5 --seed 44
ABLATE_DISABLE_CAGE=1 python sensor_code_14.py --model Qwen/Qwen2.5-7B-Instruct --mode ablate --iters 5 --seed 44
ABLATE_DISABLE_BIST=1 python sensor_code_14.py --model Qwen/Qwen2.5-7B-Instruct --mode ablate --iters 5 --seed 44
ABLATE_DISABLE_LLM=1 ABLATE_DISABLE_CAGE=1 ABLATE_DISABLE_BIST=1 python sensor_code_14.py --model Qwen/Qwen2.5-7B-Instruct --mode ablate --iters 5 --seed 44

# ---- MODEL 2: Qwen2.5-1.5B-Instruct (1.5B, same-family small) ----
echo "===== MODEL 2: Qwen2.5-1.5B ====="
python sensor_code_14.py --model Qwen/Qwen2.5-1.5B-Instruct --mode final --iters 5 --seed 44
ABLATE_DISABLE_LLM=1 python sensor_code_14.py --model Qwen/Qwen2.5-1.5B-Instruct --mode ablate --iters 5 --seed 44
ABLATE_DISABLE_CAGE=1 python sensor_code_14.py --model Qwen/Qwen2.5-1.5B-Instruct --mode ablate --iters 5 --seed 44
ABLATE_DISABLE_BIST=1 python sensor_code_14.py --model Qwen/Qwen2.5-1.5B-Instruct --mode ablate --iters 5 --seed 44
ABLATE_DISABLE_LLM=1 ABLATE_DISABLE_CAGE=1 ABLATE_DISABLE_BIST=1 python sensor_code_14.py --model Qwen/Qwen2.5-1.5B-Instruct --mode ablate --iters 5 --seed 44

# ---- MODEL 3: Mistral-7B-Instruct-v0.3 (7B, different family) ----
echo "===== MODEL 3: Mistral-7B ====="
python sensor_code_14.py --model mistralai/Mistral-7B-Instruct-v0.3 --mode final --iters 5 --seed 44
ABLATE_DISABLE_LLM=1 python sensor_code_14.py --model mistralai/Mistral-7B-Instruct-v0.3 --mode ablate --iters 5 --seed 44
ABLATE_DISABLE_CAGE=1 python sensor_code_14.py --model mistralai/Mistral-7B-Instruct-v0.3 --mode ablate --iters 5 --seed 44
ABLATE_DISABLE_BIST=1 python sensor_code_14.py --model mistralai/Mistral-7B-Instruct-v0.3 --mode ablate --iters 5 --seed 44
ABLATE_DISABLE_LLM=1 ABLATE_DISABLE_CAGE=1 ABLATE_DISABLE_BIST=1 python sensor_code_14.py --model mistralai/Mistral-7B-Instruct-v0.3 --mode ablate --iters 5 --seed 44

# ---- STEP 4: Generate all plots + statistical analysis ----
echo "===== GENERATING PLOTS + STATS ====="
python sensor_code_14.py --summarize



