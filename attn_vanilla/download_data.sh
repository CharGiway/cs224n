#!/usr/bin/env bash
#
# Download Multi30k (English–German) dataset for vanilla Transformer training.
# Data source: https://github.com/multi30k/dataset
#

set -euo pipefail
DATA_DIR="$(dirname "$0")/data"
mkdir -p "$DATA_DIR"

BASE_URL="https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw"

echo "==> Downloading Multi30k English–German data..."

for split in train val test_2016_flickr; do
    echo "  → ${split}"
    for lang in en de; do
        if [ "$lang" = "en" ]; then
            ext="en"
        else
            ext="de"
        fi
        curl -fSL --progress-bar "${BASE_URL}/train.${ext}.gz"   -o "${DATA_DIR}/train.${ext}.gz"  2>/dev/null || true
        curl -fSL --progress-bar "${BASE_URL}/val.${ext}.gz"     -o "${DATA_DIR}/val.${ext}.gz"    2>/dev/null || true
        curl -fSL --progress-bar "${BASE_URL}/test_2016_flickr.${ext}.gz" -o "${DATA_DIR}/test.${ext}.gz" 2>/dev/null || true
    done
done

# Handle the fact that URLs differ slightly — fallback approach
# Train
for lang in en de; do
    if [ ! -f "${DATA_DIR}/train.${lang}.gz" ] || [ "$(stat -f%z "${DATA_DIR}/train.${lang}.gz" 2>/dev/null || echo 0)" -eq 0 ]; then
        echo "  → Re-downloading train.${lang} from WMT16..."
        rm -f "${DATA_DIR}/train.${lang}.gz"
        curl -fSL --progress-bar "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/train.${lang}.gz" -o "${DATA_DIR}/train.${lang}.gz"
    fi
done

# Val
for lang in en de; do
    if [ ! -f "${DATA_DIR}/val.${lang}.gz" ] || [ "$(stat -f%z "${DATA_DIR}/val.${lang}.gz" 2>/dev/null || echo 0)" -eq 0 ]; then
        echo "  → Re-downloading val.${lang}..."
        rm -f "${DATA_DIR}/val.${lang}.gz"
        curl -fSL --progress-bar "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/val.${lang}.gz" -o "${DATA_DIR}/val.${lang}.gz"
    fi
done

# Test
for lang in en de; do
    if [ ! -f "${DATA_DIR}/test.${lang}.gz" ] || [ "$(stat -f%z "${DATA_DIR}/test.${lang}.gz" 2>/dev/null || echo 0)" -eq 0 ]; then
        echo "  → Re-downloading test_2016_flickr.${lang}..."
        rm -f "${DATA_DIR}/test.${lang}.gz"
        curl -fSL --progress-bar "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/test_2016_flickr.${lang}.gz" -o "${DATA_DIR}/test.${lang}.gz"
    fi
done

echo ""
echo "==> Decompressing..."
for f in "$DATA_DIR"/*.gz; do
    gunzip -kf "$f"
done

echo ""
echo "==> Done! Data files in ${DATA_DIR}/"
ls -lh "$DATA_DIR"/*.en "$DATA_DIR"/*.de 2>/dev/null || true

echo ""
echo "Sizes:"
wc -l "$DATA_DIR"/train.en "$DATA_DIR"/train.de \
      "$DATA_DIR"/val.en   "$DATA_DIR"/val.de \
      "$DATA_DIR"/test.en  "$DATA_DIR"/test.de 2>/dev/null || true
