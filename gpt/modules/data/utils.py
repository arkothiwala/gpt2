import os
import pyarrow.parquet as pq
import tiktoken
import torch
import random
import time
import numpy as np
from tqdm import tqdm
tqdm.pandas()
# encoded_text = []

class DataUtils:
    
    @staticmethod
    def tokenize_data(raw_data_folder, output_binary_path, tokenizer, text_column_name='text'):
        """
        This function reads raw text data from parquet files in the specified folder, tokenizes the text using the provided tokenizer, and writes the tokenized data to a binary file. The tokenized data is stored as uint16 integers, which can be efficiently read later for training a GPT model.
        Args:
            raw_data_folder (str): The path to the folder containing the raw data in parquet format
            output_binary_path (str): The path where the output binary file will be saved
            tokenizer: The tokenizer to be used for encoding the text data
        """
        eot_token_id = tokenizer.eot_token
        # delete the output binary file if it already exists to avoid appending to old data
        os.remove(output_binary_path) if os.path.exists(output_binary_path) else None
        # Write to binary file in append mode, so that we can write the encoded data of each parquet file one by one without loading everything in memory at once.
        with open(output_binary_path, 'ab') as f_out:

            # For each parquet file, encode and write tokenized data to binary file
            for parquet_file in tqdm(sorted(os.listdir(raw_data_folder)), desc="Processing parquet files"):
                if parquet_file.endswith(".parquet"):
                    file_path = os.path.join(raw_data_folder, parquet_file)
                    print(f"Processing file: {file_path}")
                    pf = pq.ParquetFile(file_path)

                    # safer option compared to reading entire file at once
                    for row_group_index in range(pf.num_row_groups):
                        row_group = pf.read_row_group(row_group_index, columns=[text_column_name])
                        text_data = row_group[text_column_name].to_pylist()
                        encoded_text = tokenizer.encode_batch(text=text_data, num_threads=os.cpu_count(), allowed_special={"<|endoftext|>"})

                        # encoded_with_eot = [enc + [eot_token_id] for enc in encoded_text]
                        for enc in encoded_text:
                            enc.append(eot_token_id)

                        # np.concatenate will be fastest compared to itertools.chain and list comprehensions given the data is int only -> casting it to uint16 because GPT2 vocab size is 50k which is less than 65k limit for unsigned int16.
                        # using uint16 will restrict us to not store -100 as value for end_of_text tokens.
                        encoded_text_array = np.concatenate(encoded_text).astype(np.uint16)
                        # further processing like creating input-target pairs, batching, etc. can be done here
                        f_out.write(encoded_text_array.tobytes())

    def create_eot_index(tokenized_binary_path, binfile_obj_dtype, query_value, output_index_path):
        """
        This function reads the tokenized binary file, finds the offsets of the end-of-text (EoT) tokens, and saves these offsets to a separate binary file. The offsets are stored as int64 integers to accommodate large files.
        Args:
            tokenized_binary_path (str): The path to the tokenized binary file
            binfile_obj_dtype (dtype): The data type of the binary file object
            query_value (int): The value to query in the binary file
            output_index_path (str): The path where the EoT offsets binary file will be saved
        """
        # Map the tokenized binary file
        data = np.memmap(tokenized_binary_path, dtype=binfile_obj_dtype, mode='r')

        # Find all indices where the token equals the query value
        eot_offsets = np.where(data == query_value)[0]

        # Save the offsets as an int64 binary file
        # (int64 is required because an offset index can easily exceed 4 billion)
        eot_offsets.astype(np.int64).tofile(output_index_path)

    
    @staticmethod
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % 2**32
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def split_binary_file(input_path, val_size_mb=500, test_size_mb=500, bytes_per_token=2):
    # Calculate exact split sizes in bytes (must be a multiple of token size)
        val_size_bytes = (val_size_mb * 1024 * 1024) // bytes_per_token * bytes_per_token
        test_size_bytes = (test_size_mb * 1024 * 1024) // bytes_per_token * bytes_per_token
    
        total_bytes = os.path.getsize(input_path)
        train_size_bytes = total_bytes - val_size_bytes - test_size_bytes
        
        print(f"Total file size: {total_bytes / (1024**3):.2f} GB")
        print(f"Train size: {train_size_bytes / (1024**2):.2f} MB")
        print(f"Val size: {val_size_bytes / (1024**2):.2f} MB")
        print(f"Test size: {test_size_bytes / (1024**2):.2f} MB")

        base_dir = os.path.dirname(input_path)
        
        # Define output paths
        train_path = os.path.join(base_dir, "train.bin")
        val_path = os.path.join(base_dir, "val.bin")
        test_path = os.path.join(base_dir, "test.bin")
        
        chunk_size = 64 * 1024 * 1024 # 64MB buffer chunk
        
        with open(input_path, 'rb') as f_in:
            # 1. Write Train Data
            print("Writing train.bin...")
            bytes_written = 0
            with open(train_path, 'wb') as f_out:
                while bytes_written < train_size_bytes:
                    to_read = min(chunk_size, train_size_bytes - bytes_written)
                    chunk = f_in.read(to_read)
                    if not chunk: break
                    f_out.write(chunk)
                    bytes_written += len(chunk)
                    
            # 2. Write Validation Data
            print("Writing val.bin...")
            bytes_written = 0
            with open(val_path, 'wb') as f_out:
                while bytes_written < val_size_bytes:
                    to_read = min(chunk_size, val_size_bytes - bytes_written)
                    chunk = f_in.read(to_read)
                    if not chunk: break
                    f_out.write(chunk)
                    bytes_written += len(chunk)
                    
            # 3. Write Test Data
            print("Writing test.bin...")
            bytes_written = 0
            with open(test_path, 'wb') as f_out:
                while bytes_written < test_size_bytes:
                    to_read = min(chunk_size, test_size_bytes - bytes_written)
                    chunk = f_in.read(to_read)
                    if not chunk: break
                    f_out.write(chunk)
                    bytes_written += len(chunk)

        print("Successfully completed splitting binary data!")

# # Replace with your actual binary file path
# split_binary_file("path_to_your_fineweb_file.bin", val_size_mb=500, test_size_mb=500, bytes_per_token=2)


if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding(encoding_name="gpt2")
    output_dir = "assets/processed_data/finewebedu"
    os.makedirs(output_dir, exist_ok=True)
    split = "train"
    start_time = time.time()
    # DataUtils.tokenize_data(
    #     # raw_data_folder="/Users/ashutosh/personal/study/gpt/assets/raw_data", 
    #     raw_data_folder=os.path.expanduser(f"~/.cache/huggingface/hub/datasets--Skylion007--openwebtext/snapshots/b4325f019c648b1641a1784748667e8b74e5e064/{split}/"),
    #     output_binary_path=os.path.join(output_dir, f"openwebtext_{split}.bin"), 
    #     tokenizer=tokenizer
    # )
    # DataUtils.create_eot_index(
    #     tokenized_binary_path=os.path.join(output_dir, f"openwebtext_{split}.bin"),
    #     binfile_obj_dtype=np.uint16,
    #     query_value=tokenizer.eot_token,
    #     output_index_path=os.path.join(output_dir, f"openwebtext_{split}_eot_index.bin")
    # )

    DataUtils.tokenize_data(
        # raw_data_folder="/Users/ashutosh/personal/study/gpt/assets/raw_data", 
        # raw_data_folder=os.path.expanduser(f"~/.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/10BT/{split}/"),
        # raw_data_folder=os.path.expanduser(f"~/.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/{split}/"),
        # raw_data_folder=os.path.expanduser(f"~/.cache/huggingface/hub/datasets--Skylion007--openwebtext/snapshots/b4325f019c648b1641a1784748667e8b74e5e064/sample_{split}"),
        raw_data_folder=os.path.expanduser(f"~/.cache/huggingface/hub/datasets--HuggingFaceFW--fineweb-edu/snapshots/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/sample/{split}"),
        output_binary_path=os.path.join(output_dir, f"{split}.bin"), 
        tokenizer=tokenizer
    )
    DataUtils.create_eot_index(
        tokenized_binary_path=os.path.join(output_dir, f"{split}.bin"),
        binfile_obj_dtype=np.uint16,
        query_value=tokenizer.eot_token,
        output_index_path=os.path.join(output_dir, f"{split}_eot_index.bin")
    )
    end_time = time.time()
    print(f"Data tokenization and EoT index creation completed in {(end_time - start_time):.2f} seconds.")