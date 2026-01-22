import kagglehub

# Download latest version
path = kagglehub.dataset_download("viseaonlab/flsea-stereo")

print("Path to dataset files:", path)