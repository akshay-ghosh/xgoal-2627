#!/usr/bin/env bash
# Downloads NHL shot data from MoneyPuck (peter-tanner.com mirror) into ./data/
# as shots_<startYear>_<endYear>.csv, matching the filenames nn_xgoals*.py expect.
#
# Usage:
#   ./download_data.sh                 # download all seasons 2007-2025, skip ones already on disk
#   ./download_data.sh 2022 2023 2024  # download only specific season start-years
#   ./download_data.sh --force 2025    # re-download even if the CSV already exists
#                                       # (needed for the current in-progress season, which
#                                       # MoneyPuck keeps appending shots to during the year)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
BASE_URL="https://peter-tanner.com/moneypuck/downloads"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
  shift
fi

if [[ $# -gt 0 ]]; then
  YEARS=("$@")
else
  YEARS=($(seq 2007 2025))
fi

mkdir -p "$DATA_DIR"

for year in "${YEARS[@]}"; do
  next_year=$((year + 1))
  out_csv="$DATA_DIR/shots_${year}_${next_year}.csv"

  if [[ -f "$out_csv" && "$FORCE" -eq 0 ]]; then
    echo "skip: $out_csv already exists (use --force to re-download)"
    continue
  fi

  url="$BASE_URL/shots_${year}.zip"
  tmp_dir="$(mktemp -d)"
  tmp_zip="$tmp_dir/shots_${year}.zip"

  echo "downloading $url"
  if ! curl -fsSL "$url" -o "$tmp_zip"; then
    echo "warning: failed to download $url, skipping"
    rm -rf "$tmp_dir"
    continue
  fi

  unzip -oq "$tmp_zip" -d "$tmp_dir"
  csv_file="$(find "$tmp_dir" -maxdepth 1 -iname '*.csv' | head -n 1)"

  if [[ -z "$csv_file" ]]; then
    echo "warning: no CSV found inside $url, skipping"
    rm -rf "$tmp_dir"
    continue
  fi

  mv "$csv_file" "$out_csv"
  rm -rf "$tmp_dir"
  echo "saved $out_csv"
done
