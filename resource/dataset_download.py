from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="commaai/commavq",
    repo_type="dataset",
    local_dir="resource/dataset",
    local_dir_use_symlinks=False,   # set True if you want symlinks
    # allow_patterns=["data-0000.tar.gz"],  # optionally limit to specific shards
)

print("Downloaded to:", local_dir)
