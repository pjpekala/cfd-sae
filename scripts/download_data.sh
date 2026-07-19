#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:-"./data"}
mkdir -p "$DATA_DIR"

BASE="https://storage.googleapis.com/dm-meshgraphnets/cylinder_flow"
for split in train valid test; do
  echo "Downloading ${split}.tfrecord ..."
  curl -L --retry 5 --fail -o "$DATA_DIR/${split}.tfrecord" "$BASE/${split}.tfrecord"
done

echo "Done. Files in $DATA_DIR"
ls -lh "$DATA_DIR"
