"""
clean_logs.py — Clean up old files from logs/finetune/

Files in logs/finetune/ accumulate across runs:
  - {INSTRUMENT}_{RESOLUTION}_{DATE}_metrics.csv  — one per finetune run, date-stamped
  - {INSTRUMENT}_{RESOLUTION}.log                 — appended to on every run_finetune.sh run
  - {INSTRUMENT}_{RESOLUTION}_eval.log            — appended to on every run_eval_finetuned.sh run
  - finetune.log                                  — master log, appended to every run
  - batch.log                                     — from tee, appended to every run

This script:
  - Metrics CSVs: keeps only the NEWEST per instrument+resolution, deletes older ones
  - .log files: truncates them to empty (preserves the file so future runs can append)
    By default only metrics CSVs are cleaned. Pass --logs to also truncate log files.

Usage:
  python clean_logs.py              # dry run — shows what would be deleted
  python clean_logs.py --confirm    # actually deletes/truncates
  python clean_logs.py --logs       # dry run including log truncation
  python clean_logs.py --logs --confirm   # deletes old metrics + truncates logs
"""

import os
import glob
import argparse
import re
from collections import defaultdict
from datetime import datetime


LOG_DIR = "logs/finetune"


def parse_metrics_files(log_dir):
    """
    Group metrics CSVs by (instrument, resolution).
    Filename format: {EXCHANGE}_{TICKER}_{RESOLUTION}_{YYYYMMDD}_metrics.csv
    e.g. NYSE_AAPL_1D_20260404_metrics.csv
    """
    pattern = os.path.join(log_dir, "*_metrics.csv")
    files   = glob.glob(pattern)

    groups = defaultdict(list)
    for fpath in files:
        fname = os.path.basename(fpath)
        # Strip _metrics.csv suffix, then find the date (last 8-digit field)
        stem  = fname.replace("_metrics.csv", "")
        parts = stem.split("_")

        # Date is the last part — 8 digits YYYYMMDD
        if len(parts) < 4 or not re.match(r"^\d{8}$", parts[-1]):
            print(f"  [WARN] Could not parse date from: {fname} — skipping")
            continue

        date_str   = parts[-1]
        # Instrument+resolution is everything before the date
        inst_res   = "_".join(parts[:-1])

        try:
            date = datetime.strptime(date_str, "%Y%m%d")
        except ValueError:
            print(f"  [WARN] Bad date in: {fname} — skipping")
            continue

        groups[inst_res].append((date, fpath))

    return groups


def find_old_metrics(groups):
    """For each instrument+resolution group, return all files except the newest."""
    to_delete = []
    to_keep   = []
    for inst_res, entries in sorted(groups.items()):
        entries.sort(key=lambda x: x[0], reverse=True)  # newest first
        keep = entries[0]
        old  = entries[1:]
        to_keep.append((inst_res, keep[1], keep[0].strftime("%Y-%m-%d")))
        for date, fpath in old:
            to_delete.append((inst_res, fpath, date.strftime("%Y-%m-%d")))
    return to_keep, to_delete


def find_log_files(log_dir):
    """Return all .log files in the directory."""
    return sorted(glob.glob(os.path.join(log_dir, "*.log")))


def human_size(path):
    try:
        b = os.path.getsize(path)
        if b < 1024:        return f"{b}B"
        if b < 1024**2:     return f"{b/1024:.1f}KB"
        return f"{b/1024**2:.1f}MB"
    except OSError:
        return "?"


def main():
    parser = argparse.ArgumentParser(
        description="Clean up old files from logs/finetune/"
    )
    parser.add_argument(
        "--confirm", action="store_true",
        help="Actually delete/truncate files. Without this flag, runs in dry-run mode."
    )
    parser.add_argument(
        "--logs", action="store_true",
        help="Also truncate .log files to empty (keeps the files, clears contents)."
    )
    parser.add_argument(
        "--dir", default=LOG_DIR,
        help=f"Log directory to clean (default: {LOG_DIR})"
    )
    args = parser.parse_args()

    log_dir = args.dir
    dry_run = not args.confirm

    if not os.path.isdir(log_dir):
        print(f"Directory not found: {log_dir}")
        return

    if dry_run:
        print("DRY RUN — pass --confirm to actually delete/truncate\n")

    # ── Metrics CSVs ──────────────────────────────────────────────────────────
    groups              = parse_metrics_files(log_dir)
    to_keep, to_delete  = find_old_metrics(groups)

    total_metrics = sum(len(v) for v in groups.values())
    print(f"Metrics CSVs: {total_metrics} total across {len(groups)} instruments")
    print(f"  Keeping : {len(to_keep)}")
    print(f"  Deleting: {len(to_delete)}")
    print()

    if to_delete:
        print("Files to DELETE (old metrics CSVs):")
        for inst_res, fpath, date in sorted(to_delete):
            size = human_size(fpath)
            print(f"  [{date}]  {os.path.basename(fpath)}  ({size})")
        print()

        if not dry_run:
            deleted = 0
            for _, fpath, _ in to_delete:
                try:
                    os.remove(fpath)
                    deleted += 1
                except OSError as e:
                    print(f"  [ERROR] Could not delete {fpath}: {e}")
            print(f"✓ Deleted {deleted} old metrics CSV(s)")
    else:
        print("No old metrics CSVs to delete — already clean.")

    print()

    # ── Log files ─────────────────────────────────────────────────────────────
    log_files = find_log_files(log_dir)
    if args.logs:
        total_log_size = sum(os.path.getsize(f) for f in log_files if os.path.exists(f))
        size_str = human_size.__wrapped__(total_log_size) if hasattr(human_size, '__wrapped__') else f"{total_log_size/1024:.1f}KB"

        print(f"Log files: {len(log_files)} files")
        print("Files to TRUNCATE (contents cleared, file kept for future appends):")
        for fpath in log_files:
            print(f"  {os.path.basename(fpath)}  ({human_size(fpath)})")
        print()

        if not dry_run:
            truncated = 0
            for fpath in log_files:
                try:
                    open(fpath, "w").close()
                    truncated += 1
                except OSError as e:
                    print(f"  [ERROR] Could not truncate {fpath}: {e}")
            print(f"✓ Truncated {truncated} log file(s)")
    else:
        if log_files:
            total_size = sum(os.path.getsize(f) for f in log_files if os.path.exists(f))
            print(f"Log files: {len(log_files)} files  "
                  f"({total_size/1024:.1f}KB total) — pass --logs to truncate these too")

    # ── Summary ───────────────────────────────────────────────────────────────
    if dry_run and (to_delete or (args.logs and log_files)):
        print()
        print("Re-run with --confirm to apply changes:")
        cmd = "python clean_logs.py --confirm"
        if args.logs:
            cmd += " --logs"
        if args.dir != LOG_DIR:
            cmd += f" --dir {args.dir}"
        print(f"  {cmd}")


if __name__ == "__main__":
    main()