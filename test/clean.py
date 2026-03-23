from huggingface_hub import HfApi

api = HfApi()
REPO_ID = "SAT-oO/commavq_compression"

# 1. Get all files in the repository
repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="model")

# 2. Filter for files in the ROOT folder that start with "epoch_" and end with ".pt"
# The '/' check ensures we aren't deleting things inside subfolders (non-recursive)
files_to_delete = [
    f for f in repo_files 
    if f.startswith("step_") and f.endswith(".pt") and "/" not in f
]

if not files_to_delete:
    print("No matching files found in the root directory.")
else:
    print(f"Found {len(files_to_delete)} files to delete in the root.")
    deleted = 0
    failed = []

    # `delete_files(paths=...)` is not available in older huggingface_hub versions.
    # Use per-file delete_file for compatibility.
    for file in files_to_delete:
        print(f"Deleting: {file}")
        try:
            api.delete_file(
                path_in_repo=file,
                repo_id=REPO_ID,
                repo_type="model",
                commit_message=f"Cleanup: remove {file}",
            )
            deleted += 1
        except Exception as e:
            failed.append((file, str(e)))

    print(f"Deleted: {deleted}/{len(files_to_delete)}")
    if failed:
        print("Failed deletes:")
        for file, err in failed:
            print(f"  - {file}: {err}")
    else:
        print("Successfully cleaned up the root folder.")