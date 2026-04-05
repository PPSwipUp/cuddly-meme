#!/usr/bin/env bash
# ─── Stage 6: Batch Fine-Tuning ───────────────────────────────────────────────
# Runs finetune.py on every processed instrument automatically.
# Fine-tuned checkpoints are saved to models/fine_tuned/.
# Safe to re-run: already-completed instruments are skipped.
#
# Handles filename format: EXCHANGE_TICKER_RESOLUTION_STARTDATE_ENDDATE.npy
# e.g.  NYSE_AAPL_1D_20150101_20260326.npy
#        ─────────── ── ────────── ────────
#        instrument  res  start      end
#
# Usage:
#   bash run_finetune.sh
#   bash run_finetune.sh 2>&1 | tee logs/finetune/batch.log
# ─────────────────────────────────────────────────────────────────────────────

set -e

LOG_DIR="logs/finetune"
mkdir -p "$LOG_DIR"
mkdir -p "models/fine_tuned"

PASS=0
FAIL=0
SKIP=0
START=$(date +%s)

echo "=========================================="
echo " Stage 6 Batch Fine-Tuning"
echo " Started: $(date)"
echo "=========================================="

for npy_path in data/processed/*.npy; do
    filename=$(basename "$npy_path" .npy)

    # Split filename into parts on underscore
    # Format: EXCHANGE_TICKER_RESOLUTION[_STARTDATE_ENDDATE]
    # Resolution is always the 3rd underscore-delimited field (field index 2).
    # We use IFS to split into an array.
    IFS='_' read -ra parts <<< "$filename"

    # Need at least 3 parts: exchange, ticker, resolution
    if [[ ${#parts[@]} -lt 3 ]]; then
        continue
    fi

    resolution="${parts[2]}"

    # Reconstruct instrument as EXCHANGE_TICKER (first two parts only)
    instrument="${parts[0]}_${parts[1]}"

    # Only process known resolutions
    if [[ "$resolution" != "1D" && "$resolution" != "1W" &&           "$resolution" != "4H" && "$resolution" != "1H" ]]; then
        continue
    fi

    # Skip the S&P 500 correlation reference — it is not traded
    if [[ "$instrument" == *"GSPC"* ]]; then
        SKIP=$((SKIP + 1))
        continue
    fi

    # Skip _inspect CSV companion files (glob should only give .npy but be safe)
    if [[ "$filename" == *"_inspect"* ]]; then
        continue
    fi

    # Skip if a fine-tuned checkpoint already exists for this instrument+resolution.
    # finetune.py saves checkpoints as checkpoint_INSTRUMENT_RESOLUTION_phase*.pt
    # Check for any phase checkpoint matching this instrument+resolution.
    existing=$(ls models/fine_tuned/checkpoint_${instrument}_${resolution}_phase*.pt 2>/dev/null | head -1)
    if [[ -n "$existing" ]]; then
        echo "[SKIP] $instrument $resolution — checkpoint exists: $(basename $existing)"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo ""
    echo "── Fine-tuning: $instrument  ($resolution)  $(date +%H:%M:%S)"

    log_file="$LOG_DIR/${instrument}_${resolution}.log"

    if python finetune.py --instrument "$instrument" --resolution "$resolution"         >> "$log_file" 2>&1; then
        echo "   ✓ Done"
        PASS=$((PASS + 1))
    else
        echo "   ✗ FAILED — see $log_file"
        FAIL=$((FAIL + 1))
        # Don't exit on failure (set -e would kill the loop) — continue with next
    fi
done

END=$(date +%s)
ELAPSED=$(( (END - START) / 60 ))

echo ""
echo "=========================================="
echo " Batch complete in ${ELAPSED} minutes"
echo " Passed : $PASS"
echo " Failed : $FAIL"
echo " Skipped: $SKIP"
echo "=========================================="