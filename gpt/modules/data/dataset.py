import time
import tiktoken
import torch
import pandas as pd
import numpy as np
import os
import itertools
from more_itertools import batched

class GPTDataUtils:

    @staticmethod
    def load_raw_data(path, columns=None, format='parquet'):
        if format == 'parquet':
            return pd.read_parquet(path, columns=columns)
        else:
            raise NotImplementedError()

class GPTDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data_path, num_threads=os.cpu_count(), min_seq_len=1, max_seq_len=512, min_start_idx=0):
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

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.min_start_idx = min_start_idx

    def __len__(self):
        return len(self.raw_data_df)

    def __getitem__(self, index):
        x = self.tokens[index]
        y = self.tokens[index][1:] + [self.tokenizer._special_tokens.get('<|endoftext|>')]
        seq_len = len(x)
        if seq_len < self.min_seq_len:
            raise ValueError(f"Sequence length {seq_len} is out of bounds [{self.min_seq_len}, {self.max_seq_len}]")
        
        # ensure uniform distribution of sequence lengths
        curr_seq_len = np.random.randint(low=self.min_seq_len, high=min(self.max_seq_len, seq_len)+1)
        start_idx = self.min_start_idx #np.random.randint(low=self.min_start_idx, high=seq_len-curr_seq_len+1)
        end_idx = min(start_idx + curr_seq_len, seq_len)
        # should we always use max_seq_len or should we truncate it like we are doing now?
        # I think we should not truncate it until end_of_file token is reached, because that is what the model will see during inference as well. 
        # Also, it will be good to have a mix of sequence lengths during training.
        return torch.tensor(x[start_idx:end_idx]), torch.tensor(y[start_idx:end_idx])
    
class GPTDatasetSequancePacking(torch.utils.data.Dataset):
    def __init__(self, raw_data_path, num_threads=os.cpu_count(), min_seq_len=1, max_seq_len=512, min_start_idx=0):
        # load the raw data
        dataloader_start_time = time.time()
        # This will become a bottleneck if the raw data is too large
        self.raw_data_df = GPTDataUtils.load_raw_data(path=raw_data_path)#"assets/raw_data")
        dataloader_end_time = time.time()
        # print(f"time to load raw data = {round(dataloader_end_time-dataloader_start_time, 4)} seconds")

        # load the pretrained tokenizer by gpt2
        self.tokenizer = tiktoken.get_encoding(encoding_name="gpt2")

        # MISTAKE - This would create a copy and then add <EOT> text
        # self.raw_data_df['text'] = self.raw_data_df['text'] + '<|endoftext|>'

        batch_encode_start_time = time.time()
        self.tokens = self.tokenizer.encode_batch(text=self.raw_data_df['text'], num_threads=num_threads,  allowed_special={"<|endoftext|>"})
        unique_last_tokens = set([tokens[-1] for tokens in self.tokens])
        print(f"Unique last tokens: {unique_last_tokens}")
        assert len(unique_last_tokens) == 1, "All sequences should end with <|endoftext|> token"
        # self.tokens_flattened = torch.tensor(list(itertools.chain.from_iterable(self.tokens)))
        # LEARNING - This is faster than using itertools.chain.from_iterable, because it avoids creating an intermediate list of all tokens, and directly creates a tensor from the generator expression.
        self.tokens_flattened = torch.tensor([x for sub in self.tokens for x in (*sub, 100)])

        batch_encode_end_time = time.time()
        print(f"num_thread = {num_threads} \t| raw_data_load_time = {round(dataloader_end_time-dataloader_start_time, 4)} \t| tokenizer_batch_encode_time = {round(batch_encode_end_time-batch_encode_start_time, 4)}")

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.min_start_idx = min_start_idx

    def __len__(self):
        return len(self.tokens_flattened) // self.max_seq_len

    def __getitem__(self, index):
        start_idx = index*self.max_seq_len
        end_idx = (index+1)*self.max_seq_len
        if end_idx >= len(self.tokens_flattened)-1:
            start_idx = len(self.tokens_flattened) - self.max_seq_len - 1
            end_idx = len(self.tokens_flattened) - 1
        x = self.tokens_flattened[start_idx:end_idx]
        # MISTAKE - didn't clone y tensor, which caused the original tokens_flattened tensor to be modified when we set the EOT_mask to -100
        y = self.tokens_flattened[start_idx+1:end_idx+1].clone()
        
        # mask the loss for the tokens which are after <|endoftext|> token, because those tokens are not actually seen by the model during training, and we don't want the model to learn from those tokens.
        # this will make it slow though due to finding EOT mask for each sequence
        EOT_mask = (x == self.tokenizer.eot_token)
        # print(f"EOT_Mask | x = {x[EOT_mask]} | y = {y[EOT_mask]}")
        y[EOT_mask] = torch.tensor(-100) # -100 is the default ignore index for CrossEntropyLoss in PyTorch
        return x, y


class GPTDatasetBinFile(torch.utils.data.Dataset):
    def __init__(self, file_path, context_length, binfile_dtype, eot_token, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_length = context_length
        self.eot_token = eot_token
        
        assert os.path.splitext(file_path)[1] == '.bin', f"file_path must point to a .bin file. invalid file_path = {file_path}"
        # read the binary file and create memmap object
        self.data = np.memmap(
            filename=file_path,
            dtype=binfile_dtype,
            mode='r'
        )
        
    def __len__(self):
        return (self.data.shape[0]-1) // self.context_length

    def __getitem__(self, index):
        # check_1 - idx should not be less than or eq. len
        assert index <= self.__len__()-1, "index out of bound"

        # assign start and end index
        x_start_idx = index*self.context_length
        x_end_idx = (index+1)*self.context_length
        y_start_idx = x_start_idx + 1
        y_end_idx = x_end_idx + 1

        # edge case handling where `data.shape[0]%context_length == 0`
        if y_end_idx >= self.data.shape[0]:
            x_start_idx -= 1
            x_end_idx -= 1
            y_start_idx -= 1
            y_end_idx -= 1

        # MISTAKE | PERFORMANCE_IMPACT - below approach reads twice -> causing 2x disk pressure, 2x memory because for both we are doing astype(np.int64).
        # x = self.data[x_start_idx:x_end_idx].astype(np.int64)
        # y = self.data[y_start_idx:y_end_idx].astype(np.int64)

        # x_tensor = torch.from_numpy(x)
        # y_tensor = torch.from_numpy(y)

        # This is better compared to above
        xy_data = torch.from_numpy(self.data[x_start_idx:y_end_idx].astype(np.int64))
        x_tensor, y_tensor = xy_data[:-1], xy_data[1:]


        # here we don't need to clone the y_tensor because we are doing .astype(np.int64) so original data in memmap is not modified.
        EOT_mask = (x_tensor == self.eot_token) # use the eot_token provided during initialization
        y_tensor[EOT_mask] = torch.tensor(-100) # -100 is the default ignore index for CrossEntropyLoss in PyTorch

        return x_tensor, y_tensor
