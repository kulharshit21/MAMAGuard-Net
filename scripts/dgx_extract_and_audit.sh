#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${MAMAGUARD_DATA_ROOT:-$PWD/data}"
ARCHIVE="${MAMAGUARD_BREAST_ARCHIVE:-$DATA_ROOT/downloads/Breast_Cancer_DCE-MRI_Data.7z}"
OUTPUT="${MAMAGUARD_BREAST_ROOT:-$DATA_ROOT/Breast_Cancer_DCE-MRI_Data}"
REPORT_DIR="${MAMAGUARD_REPORT_DIR:-$PWD/reports/generated}"
MIN_FREE_GIB="${MAMAGUARD_EXTRACT_MIN_FREE_GIB:-100}"

if [[ ! -f "$ARCHIVE" ]]; then
  echo "Archive not found: $ARCHIVE" >&2
  exit 66
fi

SEVEN_ZIP="$(command -v 7zz || command -v 7z || true)"
if [[ -z "$SEVEN_ZIP" ]]; then
  PYTHON_BIN="${MAMAGUARD_PYTHON:-python3}"
  exec "$PYTHON_BIN" scripts/extract_nested_archives.py \
    --data-root "$DATA_ROOT" \
    --archive "$ARCHIVE" \
    --output "$OUTPUT" \
    --report-dir "$REPORT_DIR" \
    --minimum-free-gib "$MIN_FREE_GIB"
fi

available_kib="$(df -Pk "$DATA_ROOT" | awk 'NR==2 {print $4}')"
required_kib="$((MIN_FREE_GIB * 1024 * 1024))"
if (( available_kib < required_kib )); then
  echo "Extraction requires at least ${MIN_FREE_GIB} GiB free; only $((available_kib / 1024 / 1024)) GiB is available." >&2
  exit 70
fi

mkdir -p "$OUTPUT" "$REPORT_DIR"
echo "Testing outer archive: $ARCHIVE"
"$SEVEN_ZIP" t -bso0 -bsp1 "$ARCHIVE"

outer_marker="$OUTPUT/.mamaguard_outer_extracted.ok"
if [[ ! -f "$outer_marker" ]]; then
  echo "Extracting outer archive into: $OUTPUT"
  "$SEVEN_ZIP" x -y -bso1 -bsp1 "-o$OUTPUT" "$ARCHIVE"
  printf '%s\n' "$ARCHIVE" > "$outer_marker"
else
  echo "Outer archive already marked as successfully extracted."
fi

extracted_count=0
while true; do
  zip_path="$(find "$OUTPUT" -type f -iname '*.zip' -print -quit)"
  [[ -n "$zip_path" ]] || break
  zip_hash="$(sha256sum "$zip_path" | awk '{print $1}')"
  destination="${zip_path%.*}"
  marker="$destination/.mamaguard_${zip_hash}.ok"
  echo "Testing nested archive: $zip_path"
  "$SEVEN_ZIP" t -bso0 -bsp1 "$zip_path"
  mkdir -p "$destination"
  if [[ ! -f "$marker" ]]; then
    echo "Extracting into: $destination"
    "$SEVEN_ZIP" x -y -bso1 -bsp1 "-o$destination" "$zip_path"
    if [[ -z "$(find "$destination" -mindepth 1 -type f -print -quit)" ]]; then
      echo "Extraction produced no files; preserving archive: $zip_path" >&2
      exit 74
    fi
    printf '%s\n' "$zip_hash" > "$marker"
  fi
  rm -- "$zip_path"
  extracted_count=$((extracted_count + 1))
done

echo "Successfully extracted and removed $extracted_count nested ZIP archives."
python scripts/profile_dataset_tree.py "$DATA_ROOT" \
  --output "$REPORT_DIR/dataset_tree_after_extraction.json"
find "$DATA_ROOT" -maxdepth 4 -type d -printf '%p\n' \
  > "$REPORT_DIR/dataset_directories_depth4.txt"
find "$DATA_ROOT" -type f \( -iname '*.zip' -o -iname '*.part' -o -iname '*.partial' \) \
  -printf '%p\t%s bytes\n' > "$REPORT_DIR/remaining_archives_or_partials.txt"

echo "Audit files written under: $REPORT_DIR"
echo "The original outer archive remains at: $ARCHIVE"
