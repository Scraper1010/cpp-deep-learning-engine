#!/usr/bin/env pwsh

New-Item -ItemType Directory -Force -Path "data" | Out-Null

$BASE_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"

$FILES = @(
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte"
)

foreach ($file in $FILES) {
    $outputFile = "data\$file"
    $gzFile = "$outputFile.gz"

    if (-not (Test-Path $outputFile)) {
        Write-Host "Downloading $file.gz..."

        Invoke-WebRequest `
            -Uri "$BASE_URL/$file.gz" `
            -OutFile $gzFile

        # Extract .gz
        $input = [System.IO.File]::OpenRead($gzFile)
        $gzip = New-Object System.IO.Compression.GzipStream(
            $input,
            [System.IO.Compression.CompressionMode]::Decompress
        )
        $output = [System.IO.File]::Create($outputFile)

        $gzip.CopyTo($output)

        $output.Close()
        $gzip.Close()
        $input.Close()

        Remove-Item $gzFile
    }
}

Write-Host "MNIST dataset ready in data/"