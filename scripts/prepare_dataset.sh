
# Download training dataset
- export HF_TOKEN=<ADD_YOUR_TOKEN>
- hf download Skylion007/openwebtext --repo-type dataset
- arrange the data in train/val/test format
    ```
    # 1. Find the latest snapshot directory and save it to a variable
    SNAPSHOT_DIR=$(ls -td ~/.cache/huggingface/hub/datasets--Skylion007--openwebtext/snapshots/*/ | head -n 1)

    # 2. Verify and CD into the dynamic path
    echo "Moving into snapshot: $SNAPSHOT_DIR"
    cd "$SNAPSHOT_DIR"

    # 3. Create the target directories
    mkdir -p sample_train sample_val sample_test

    # 4. Move the files
    mv plain_text/train-{00000..00060}-of-00080.parquet sample_train/
    mv plain_text/train-{00061..00070}-of-00080.parquet sample_val/
    mv plain_text/train-{00071..00080}-of-00080.parquet sample_test/
    ```
- 