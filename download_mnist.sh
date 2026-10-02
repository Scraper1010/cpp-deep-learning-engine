#!/bin/bash
mkdir -p data
BASE_URL="https://ossci-datasets.s3.amazonaws.com/mnist"

FILES=(
    "train-images-idx3-ubyte"
    "train-labels-idx1-ubyte"
    "t10k-images-idx3-ubyte"
    "t10k-labels-idx1-ubyte"
)

for file in "${FILES[@]}"; do
    if [ ! -f "data/$file" ]; then
        echo "Downloading $file.gz..."
        curl -s "$BASE_URL/$file.gz" -o "data/$file.gz"
        gunzip -f "data/$file.gz"
    fi
done
echo "MNIST dataset ready in data/"