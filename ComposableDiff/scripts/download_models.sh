#!/bin/bash

cd ..

# Create a directory to store the downloaded files
mkdir -p models
mkdir -p models/clevr_pos_classifier_128
mkdir -p models/clevr_rel_classifier_128

# Function to download a file
download_file() {
	local url=$1
	local filename=$2
	echo "Downloading $filename..."
	curl -L "$url" -o "$filename"
	echo "Downloaded $filename"
}

# Download the models
download_file "https://www.dropbox.com/s/fpy5xlsxk3xdh4g/ema_0.9999_740000.pt" "models/clevr_pos.pt"
download_file "https://www.dropbox.com/s/i929gm43u7n2hj5/ema_0.9999_740000.pt" "models/clevr_rel.pt"
download_file "https://www.dropbox.com/s/kf86h2dp5zasqtq/544.tar?dl=0" "models/clevr_pos_classifier_128/544.tar"
download_file "https://www.dropbox.com/s/a0ya3yo4nfgg8c8/232.tar?dl=0" "models/clevr_rel_classifier_128/232.tar"
