import pandas as pd
import torch
import os
import tiktoken
import time
import argparse

class GPTDataUtils:

    @staticmethod
    def load_raw_data(path, columns=None, format='parquet'):
        if format == 'parquet':
            return pd.read_parquet(path, columns=columns)
        else:
            raise NotImplementedError()

class GPTDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data_path, num_threads=os.cpu_count()):
        # load the raw data
        dataloader_start_time = time.time()
        self.raw_data_df = GPTDataUtils.load_raw_data(path=raw_data_path)#"assets/raw_data")
        dataloader_end_time = time.time()
        # print(f"time to load raw data = {round(dataloader_end_time-dataloader_start_time, 4)} seconds")

        # load the pretrained tokenizer by gpt2
        self.tokenizer = tiktoken.get_encoding(encoding_name="gpt2")

        batch_encode_start_time = time.time()
        self.tokens = self.tokenizer.encode_batch(text=self.raw_data_df['text'], num_threads=num_threads)
        batch_encode_end_time = time.time()
        print(f"num_thread = {num_threads} \t| raw_data_load_time = {round(dataloader_end_time-dataloader_start_time, 4)} \t| tokenizer_batch_encode_time = {round(batch_encode_end_time-batch_encode_start_time, 4)}")
    def __len__(self):
        return len(self.raw_data_df)

    def __getitem__(self, index):
        x = self.tokens[index]
        y = self.tokens[index][1:] + [self.tokenizer._special_tokens.get('<|endoftext|>')]
        return x, y
    
if __name__ == '__main__':
    """ To run the benchmark, execute the following command in terminal:
    for i in {1..30}; do
        python benchmarking/tokeniser_batch_encode.py --num_threads $i
    done
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_threads', type=int, default=os.cpu_count(), help='Number of threads for tokenization')
    args = parser.parse_args()

    # print(f"Using {args.num_threads} threads for tokenization")

    start_time = time.time()
    gpt_dataset = GPTDataset("assets/raw_data", num_threads=args.num_threads)
    end_time = time.time()
    # print(f"time to load date = {round(end_time-start_time, 4)} seconds")


"""
command to run the benchmark:
```
for i in {1..30}; do
    python benchmarking/tokeniser_batch_encode.py --num_threads $i
done
```

Results:
num_thread = 1          | raw_data_load_time = 0.4322   | tokenizer_batch_encode_time = 52.5508
num_thread = 2          | raw_data_load_time = 0.5386   | tokenizer_batch_encode_time = 26.5676
num_thread = 3          | raw_data_load_time = 0.5085   | tokenizer_batch_encode_time = 17.6309
num_thread = 4          | raw_data_load_time = 0.55     | tokenizer_batch_encode_time = 14.6522
num_thread = 5          | raw_data_load_time = 0.4972   | tokenizer_batch_encode_time = 11.0302
num_thread = 6          | raw_data_load_time = 0.4777   | tokenizer_batch_encode_time = 9.8341
num_thread = 7          | raw_data_load_time = 0.494    | tokenizer_batch_encode_time = 8.8259
num_thread = 8          | raw_data_load_time = 0.5152   | tokenizer_batch_encode_time = 7.7873
num_thread = 9          | raw_data_load_time = 0.5119   | tokenizer_batch_encode_time = 7.367
num_thread = 10         | raw_data_load_time = 0.5085   | tokenizer_batch_encode_time = 7.4656
num_thread = 11         | raw_data_load_time = 0.5183   | tokenizer_batch_encode_time = 7.8553
num_thread = 12         | raw_data_load_time = 0.5131   | tokenizer_batch_encode_time = 8.2025
num_thread = 13         | raw_data_load_time = 0.5274   | tokenizer_batch_encode_time = 7.8393
num_thread = 14         | raw_data_load_time = 0.5216   | tokenizer_batch_encode_time = 7.8567
num_thread = 15         | raw_data_load_time = 0.5126   | tokenizer_batch_encode_time = 7.7051
num_thread = 16         | raw_data_load_time = 0.5081   | tokenizer_batch_encode_time = 8.04
num_thread = 17         | raw_data_load_time = 0.5155   | tokenizer_batch_encode_time = 10.8049
num_thread = 18         | raw_data_load_time = 0.528    | tokenizer_batch_encode_time = 9.507
num_thread = 19         | raw_data_load_time = 0.5085   | tokenizer_batch_encode_time = 10.695
num_thread = 20         | raw_data_load_time = 0.5115   | tokenizer_batch_encode_time = 11.4107
num_thread = 21         | raw_data_load_time = 0.5155   | tokenizer_batch_encode_time = 11.8891
num_thread = 22         | raw_data_load_time = 0.5481   | tokenizer_batch_encode_time = 14.4034
num_thread = 23         | raw_data_load_time = 0.5516   | tokenizer_batch_encode_time = 15.0238
num_thread = 24         | raw_data_load_time = 0.5264   | tokenizer_batch_encode_time = 13.4533
num_thread = 25         | raw_data_load_time = 0.4961   | tokenizer_batch_encode_time = 12.1849
num_thread = 26         | raw_data_load_time = 0.5096   | tokenizer_batch_encode_time = 11.8204
num_thread = 27         | raw_data_load_time = 0.5189   | tokenizer_batch_encode_time = 11.2252
num_thread = 28         | raw_data_load_time = 0.4994   | tokenizer_batch_encode_time = 15.714
num_thread = 29         | raw_data_load_time = 0.4914   | tokenizer_batch_encode_time = 12.4115
num_thread = 30         | raw_data_load_time = 0.5282   | tokenizer_batch_encode_time = 16.3009
"""