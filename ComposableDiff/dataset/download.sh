#!/bin/bash

# Function to download a file
download_file() {
	local url=$1
	local filename=$2
	echo "Downloading $filename..."
	curl -L "$url" -o "$filename"
	echo "Downloaded $filename"
}


# https://gist.github.com/tanaikech/f0f2d122e05bf5f971611258c22c110f
download_gdrive_file() {
	local file_id=$1
	local filename=$2
	echo "Downloading $filename..."
	curl -L "https://drive.usercontent.google.com/download?id=$file_id&confirm=xxx" -o "$filename"
	echo "Downloaded $filename"
}

# CLEVR Relation Test Datasets
download_file "https://www.dropbox.com/s/bfj4wjb4ksic6z2/clevr_generation_1_relations.npz?dl=1" "test_clevr_rel_5000_1.npz"
download_file "https://www.dropbox.com/s/g59mscx6880j72h/clevr_generation_2_relations.npz?dl=1" "test_clevr_rel_5000_2.npz"
download_file "https://www.dropbox.com/s/nvc2mdsixi7vu3i/clevr_generation_3_relations.npz?dl=1" "test_clevr_rel_5000_3.npz"

# CLEVR 2D Position Test Datasets
download_file "https://www.dropbox.com/s/je1nbw463ic1urm/clevr_pos_5000_1.npz?dl=1" "test_clevr_pos_5000_1.npz"
download_file "https://www.dropbox.com/s/8svd75vw1j7wmzj/clevr_pos_5000_2.npz?dl=1" "test_clevr_pos_5000_2.npz"
download_file "https://www.dropbox.com/s/j3d32udsg1cewvc/clevr_pos_5000_3.npz?dl=1" "test_clevr_pos_5000_3.npz"
download_gdrive_file "1lGvaFF5SsOXcm2u7VaacSjIOKFYU4tqm" "test_clevr_pos_5000_4.npz"
download_gdrive_file "1BWUjVYhI0x0TyhxENoGGQi5-NjZrusOD" "test_clevr_pos_5000_5.npz"
