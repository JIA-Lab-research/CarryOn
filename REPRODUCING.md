# Reproducing CarryOn

Run these Bash commands from the repository root on Linux. Each run uses a new output directory. The runners start and stop the model services automatically.

## Install

Install Python 3.10, 3.11, 3.12 and a CUDA toolkit, then install CarryOn:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[models]"

export CARRYON_WORK="$HOME/carryon-work"
export CUDA_HOME=/usr/local/cuda
mkdir -p "$CARRYON_WORK"

PRESET=smoke
```

`PRESET` selects the experiment size: `smoke` runs a small subset and `paper` runs the full task inventory. Both Recovery experiments start with empty memory and generate M1/M2 from failures in the current run.

Install each required environment once. Reuse it across experiments.

| Experiment | Required environments |
| --- | --- |
| RoboLab | `planner` |
| LIBERO Perturbation / LIBERO-Recovery | `simulation`, `planner`, `starvla`, `sam3`, `molmo`, `curobo` |
| LIBERO-Safety | `simulation`, `safety-client`, `openpi`, `oft`, `starvla`, `sam3`, `molmo` |

Run only the installation lines needed for your experiment:

```bash
python scripts/setup_environments.py --profile planner --python python3.10 --output "$CARRYON_WORK/envs/planner"
python scripts/setup_environments.py --profile simulation --python python3.10 --output "$CARRYON_WORK/envs/simulation"
python scripts/setup_environments.py --profile starvla --python python3.10 --output "$CARRYON_WORK/envs/starvla"
python scripts/setup_environments.py --profile oft --python python3.10 --output "$CARRYON_WORK/envs/oft"
python scripts/setup_environments.py --profile curobo --python python3.10 --output "$CARRYON_WORK/envs/curobo"
python scripts/setup_environments.py --profile safety-client --python python3.11 --output "$CARRYON_WORK/envs/safety-client"
python scripts/setup_environments.py --profile openpi --python python3.11 --output "$CARRYON_WORK/envs/openpi"
python scripts/setup_environments.py --profile molmo --python python3.11 --output "$CARRYON_WORK/envs/molmo"
python scripts/setup_environments.py --profile sam3 --python python3.12 --output "$CARRYON_WORK/envs/sam3"
```

Dependency versions and model downloads are supplied with the installation scripts under `requirements/`. Change the device IDs in the commands below to match your machine.

## RoboLab

Planner and endpoint checking use Qwen3-VL-32B-Instruct at temperature 0.2. These commands use local inference and require no hosted inference key.

```bash
ROBO_RUN="$CARRYON_WORK/robolab-${PRESET}-$(date +%Y%m%d-%H%M%S)"

python scripts/configure_experiment.py --suite robolab \
  --environments "$CARRYON_WORK/envs" \
  --cache "$ROBO_RUN/cache" --output "$ROBO_RUN/config" --planner-device 0
python scripts/prepare_resources.py \
  --config "$ROBO_RUN/config/experiment.json" --output "$ROBO_RUN/cache"
python scripts/run_experiment.py --suite robolab \
  --config "$ROBO_RUN/config/experiment.json" --preset "$PRESET" \
  --output "$ROBO_RUN/results"
python -m json.tool "$ROBO_RUN/results/RESULT.json"
```

The small run has 32 endpoint-selection requests and 32 memory-revision requests. Generate the endpoint-selection table and both memory-revision summaries with:

```bash
ANALYSIS=experiments/robolab/benchmark/evaluation
ANALYSIS_PYTHON="$CARRYON_WORK/envs/planner/venv/bin/python"
MAIN="$ROBO_RUN/results/main/workspace/logs/robolab"

"$ANALYSIS_PYTHON" "$ANALYSIS/analyse_main.py" \
  --episodes-t "$MAIN/text/episodes.jsonl" \
  --episodes-m "$MAIN/text_memory/episodes.jsonl" \
  --episodes-img "$MAIN/image/episodes.jsonl" \
  --preset "$PRESET" --out "$ROBO_RUN/results/endpoint-selection.json"

for modality in text image; do
  LOGS="$ROBO_RUN/results/${modality}-stale/workspace/logs/robolab"
  if [ "$modality" = text ]; then
    PREFIX=text_revision_stage1
  else
    PREFIX=image_revision_stage1
  fi
  STAGE2=()
  if [ "$PRESET" = paper ]; then
    STAGE2=(--stage2 "$LOGS/${modality}_revision_stage2/episodes.jsonl")
  fi
  "$ANALYSIS_PYTHON" "$ANALYSIS/analyse_revision.py" \
    --stage1 "$LOGS/$PREFIX/episodes.jsonl" "${STAGE2[@]}" --preset "$PRESET" \
    --out "$ROBO_RUN/results/${modality}-memory-revision.json"
done
```

`memory_corrected` requires the target to be recorded at its new location and its stale association to be removed. T/P/M/PM denote text, image, text with memory, and image with memory. Dispatch times include requests that dispatched an endpoint.

## API access for the LIBERO experiments

Use an OpenRouter key for hosted model calls and a Hugging Face token with access to SAM 3 for model downloads. If these variables are not already set:

```bash
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  read -rsp "OpenRouter API key: " OPENROUTER_API_KEY
  printf '\n'
  export OPENROUTER_API_KEY
fi
if [ -z "${HF_TOKEN:-}" ]; then
  read -rsp "Hugging Face token: " HF_TOKEN
  printf '\n'
  export HF_TOKEN
fi

read -rp "Execution API request limit: " EXECUTION_MAX_ATTEMPTS
read -rp "Execution API spending limit (USD): " EXECUTION_BUDGET_USD
REFLECTION_MAX_ATTEMPTS=0
REFLECTION_BUDGET_USD=0
```

For either Recovery experiment, also set the reflection limits before generating its configuration:

```bash
read -rp "Reflection API request limit: " REFLECTION_MAX_ATTEMPTS
read -rp "Reflection API spending limit (USD): " REFLECTION_BUDGET_USD
```

Use positive execution limits and, for Recovery runs, positive reflection limits. Reaching a limit leaves the run incomplete. Select limits for the chosen preset before running an experiment.

## LIBERO-Safety

Run CarryOn with all four policy backbones:

```bash
SAFETY_RUN="$CARRYON_WORK/libero-safety-${PRESET}-$(date +%Y%m%d-%H%M%S)"

python scripts/configure_experiment.py --suite libero_safety \
  --environments "$CARRYON_WORK/envs" \
  --cache "$SAFETY_RUN/cache" --output "$SAFETY_RUN/config" \
  --policy-device 0 --perception-device 1 \
  --policies pi05 pi05_sft openvla_oft vlact \
  --execution-max-attempts "$EXECUTION_MAX_ATTEMPTS" \
  --execution-budget-usd "$EXECUTION_BUDGET_USD"
python scripts/prepare_resources.py \
  --config "$SAFETY_RUN/config/experiment.json" --output "$SAFETY_RUN/cache"
python scripts/run_experiment.py --suite libero_safety \
  --config "$SAFETY_RUN/config/experiment.json" --preset "$PRESET" \
  --output "$SAFETY_RUN/results"
python experiments/libero_safety/aggregate.py "$SAFETY_RUN/results/RESULT.json" \
  --output "$SAFETY_RUN/results/summary.json"
```

The small run uses six cases per backbone, for 24 trajectories. To select fewer backbones, change the list after `--policies`. The analysis command writes `summary.json` and `summary.csv`, with six experiment groups and an overall summary for each backbone.

## LIBERO Perturbation

```bash
PERTURBATION_RUN="$CARRYON_WORK/libero-perturbation-${PRESET}-$(date +%Y%m%d-%H%M%S)"

python scripts/configure_experiment.py --suite libero_perturbation \
  --environments "$CARRYON_WORK/envs" \
  --cache "$PERTURBATION_RUN/cache" --output "$PERTURBATION_RUN/config" \
  --policy-device 0 --perception-device 1 --planner-device 2 --cuda-home "$CUDA_HOME" \
  --execution-max-attempts "$EXECUTION_MAX_ATTEMPTS" \
  --execution-budget-usd "$EXECUTION_BUDGET_USD" \
  --reflection-max-attempts "$REFLECTION_MAX_ATTEMPTS" \
  --reflection-budget-usd "$REFLECTION_BUDGET_USD"
python scripts/prepare_resources.py \
  --config "$PERTURBATION_RUN/config/experiment.json" --output "$PERTURBATION_RUN/cache"
python scripts/run_experiment.py --suite libero_perturbation \
  --config "$PERTURBATION_RUN/config/experiment.json" --preset "$PRESET" \
  --output "$PERTURBATION_RUN/results"
python experiments/libero_perturbation/aggregate.py --run "$PERTURBATION_RUN/results" \
  --output "$PERTURBATION_RUN/results/summary.json"
```

## LIBERO-Recovery

```bash
RECOVERY_RUN="$CARRYON_WORK/libero-recovery-${PRESET}-$(date +%Y%m%d-%H%M%S)"

python scripts/configure_experiment.py --suite libero_recovery \
  --environments "$CARRYON_WORK/envs" \
  --cache "$RECOVERY_RUN/cache" --output "$RECOVERY_RUN/config" \
  --policy-device 0 --perception-device 1 --planner-device 2 --cuda-home "$CUDA_HOME" \
  --execution-max-attempts "$EXECUTION_MAX_ATTEMPTS" \
  --execution-budget-usd "$EXECUTION_BUDGET_USD" \
  --reflection-max-attempts "$REFLECTION_MAX_ATTEMPTS" \
  --reflection-budget-usd "$REFLECTION_BUDGET_USD"
python scripts/prepare_resources.py \
  --config "$RECOVERY_RUN/config/experiment.json" --output "$RECOVERY_RUN/cache"
python scripts/run_experiment.py --suite libero_recovery \
  --config "$RECOVERY_RUN/config/experiment.json" --preset "$PRESET" \
  --output "$RECOVERY_RUN/results"
python experiments/libero_recovery/aggregate.py --run "$RECOVERY_RUN/results" \
  --output "$RECOVERY_RUN/results/summary.json"
```
