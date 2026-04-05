#!/usr/bin/env bash
# ─── Stage 7: Batch Fine-Tuned Evaluation ────────────────────────────────────
# Evaluates every fine-tuned checkpoint against the base model on each
# instrument's held-out test window (last 15% of data).
#
# Produces:
#   logs/evaluation/finetune_summary.csv  — one row per instrument with
#     accuracy, Sharpe, delta vs base, and recommended checkpoint
#
# Usage:
#   bash run_eval_finetuned.sh
#   bash run_eval_finetuned.sh 2>&1 | tee logs/evaluation/batch_eval.log
# ─────────────────────────────────────────────────────────────────────────────

EVAL_LOG_DIR="logs/evaluation"
FT_LOG_DIR="logs/finetune"
mkdir -p "$EVAL_LOG_DIR"
mkdir -p "$FT_LOG_DIR"

SUMMARY_CSV="$EVAL_LOG_DIR/finetune_summary.csv"

# Write CSV header
echo "instrument,resolution,phase,ft_h1,ft_h5,ft_h20,ft_mean,base_h1,base_h5,base_h20,base_mean,delta_h1,delta_h5,delta_h20,delta_mean,recommendation,checkpoint_path" \
    > "$SUMMARY_CSV"

PASS=0
FAIL=0
SKIP=0
START=$(date +%s)

echo "=========================================="
echo " Stage 7 Batch Fine-Tuned Evaluation"
echo " Started: $(date)"
echo "=========================================="

# ── Loop over fine-tuned checkpoints ─────────────────────────────────────────
# Process phase3 checkpoints first, then phase1.
# If a phase3 exists for an instrument, the phase1 entry is skipped.

for ckpt_path in models/fine_tuned/checkpoint_*_phase3_*.pt \
                 models/fine_tuned/checkpoint_*_phase1_*.pt; do

    [ -f "$ckpt_path" ] || continue

    ckpt_name=$(basename "$ckpt_path" .pt)

    # Strip leading "checkpoint_"
    inner="${ckpt_name#checkpoint_}"

    # Split on underscore: EXCHANGE_TICKER_RESOLUTION_phase?_DATE
    IFS='_' read -ra parts <<< "$inner"

    if [ ${#parts[@]} -lt 4 ]; then
        echo "[SKIP] Cannot parse: $ckpt_name"
        SKIP=$((SKIP + 1))
        continue
    fi

    instrument="${parts[0]}_${parts[1]}"
    resolution="${parts[2]}"
    phase_str="${parts[3]}"
    phase_num="${phase_str#phase}"

    # Skip if this instrument+resolution already has a row (phase3 processed first).
    # Use grep -q (quiet) rather than grep -c to avoid the "0\n0" issue where
    # grep -c prints "0" then exits with code 1, triggering || echo 0 and
    # producing a two-line string that fails the [ -gt ] integer comparison.
    if grep -q "^${instrument},${resolution}," "$SUMMARY_CSV" 2>/dev/null; then
        echo "[SKIP] $instrument $resolution — already evaluated"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo ""
    echo "── Evaluating: $instrument ($resolution) phase${phase_num}  $(date +%H:%M:%S)"

    log_file="$FT_LOG_DIR/${instrument}_${resolution}_eval.log"

    # Run evaluation with compare_base flag
    if python evaluation.py \
        --checkpoint  "$ckpt_path" \
        --instrument  "$instrument" \
        --resolution  "$resolution" \
        --compare_base \
        > "$log_file" 2>&1; then

        # Parse accuracy values from the delta section of the log
        ft_h1=$(grep   "H1 :.*ft=" "$log_file" 2>/dev/null | tail -1 | grep -oP "ft=\K[0-9.]+" || echo "0")
        ft_h5=$(grep   "H5 :.*ft=" "$log_file" 2>/dev/null | tail -1 | grep -oP "ft=\K[0-9.]+" || echo "0")
        ft_h20=$(grep  "H20:.*ft=" "$log_file" 2>/dev/null | tail -1 | grep -oP "ft=\K[0-9.]+" || echo "0")
        base_h1=$(grep  "H1 :.*base=" "$log_file" 2>/dev/null | tail -1 | grep -oP "base=\K[0-9.]+" || echo "0")
        base_h5=$(grep  "H5 :.*base=" "$log_file" 2>/dev/null | tail -1 | grep -oP "base=\K[0-9.]+" || echo "0")
        base_h20=$(grep "H20:.*base=" "$log_file" 2>/dev/null | tail -1 | grep -oP "base=\K[0-9.]+" || echo "0")

        # Compute means, deltas, and recommendation (python handles float arithmetic)
        row=$(python3 - "$ft_h1" "$ft_h5" "$ft_h20" "$base_h1" "$base_h5" "$base_h20" << 'PYEOF'
import sys
ft   = [float(x) for x in sys.argv[1:4]]
base = [float(x) for x in sys.argv[4:7]]
ft_mean    = sum(ft)   / 3
base_mean  = sum(base) / 3
deltas     = [f - b for f, b in zip(ft, base)]
delta_mean = ft_mean - base_mean
# Keep fine-tuned unless mean accuracy clearly degraded (>0.1pp drop)
rec = "base" if delta_mean < -0.001 else "fine_tuned"
print(
    f"{ft[0]:.3f},{ft[1]:.3f},{ft[2]:.3f},{ft_mean:.3f},"
    f"{base[0]:.3f},{base[1]:.3f},{base[2]:.3f},{base_mean:.3f},"
    f"{deltas[0]:+.3f},{deltas[1]:+.3f},{deltas[2]:+.3f},{delta_mean:+.3f},"
    f"{rec}"
)
PYEOF
)

        echo "${instrument},${resolution},${phase_num},${row},${ckpt_path}" >> "$SUMMARY_CSV"

        delta_mean=$(echo "$row" | cut -d',' -f12)
        rec=$(echo "$row" | cut -d',' -f13)
        echo "   ✓ Done  delta_mean=${delta_mean}  use=${rec}"
        PASS=$((PASS + 1))

    else
        echo "   ✗ FAILED — see $log_file"
        echo "${instrument},${resolution},ERROR,,,,,,,,,,,,,,ERROR,${ckpt_path}" >> "$SUMMARY_CSV"
        FAIL=$((FAIL + 1))
    fi

done

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
echo "=========================================="
echo " Evaluation complete in ${ELAPSED} minutes"
echo " Passed : $PASS"
echo " Failed : $FAIL"
echo " Skipped: $SKIP"
echo " Summary: $SUMMARY_CSV"
echo "=========================================="

# ── Print summary table to terminal ───────────────────────────────────────────
echo ""
echo "RECOMMENDATION SUMMARY"
echo "──────────────────────────────────────────────────────────────────────"
printf "%-26s %-4s %-5s  %-8s %-8s %-8s  %s\n" \
    "Instrument" "Res" "Phase" "FT Mean" "Base" "Delta" "Use"
echo "──────────────────────────────────────────────────────────────────────"

tail -n +2 "$SUMMARY_CSV" | while IFS=',' read \
    instr res phase \
    ft_h1 ft_h5 ft_h20 ft_mean \
    base_h1 base_h5 base_h20 base_mean \
    dh1 dh5 dh20 dmean rec ckpt; do

    flag="✓"
    [ "$rec" = "base" ] && flag="⚠"
    [ "$rec" = "ERROR" ] && flag="✗"

    printf "%-26s %-4s %-5s  %-8s %-8s %-8s  %s %s\n" \
        "$instr" "$res" "$phase" "$ft_mean" "$base_mean" "$dmean" "$flag" "$rec"
done

echo ""
echo "Full details in: $SUMMARY_CSV"
echo "Individual logs in: $FT_LOG_DIR/"