## GPT-2 from Scratch — The Paper‑Only Journey

### Description
To learn LLM pre-training, I attempted to implement GPT2 from scratch by referring only to the GPT1 and GPT2 papers. This work deliberately avoided implementation videos, blogs and tutorials to expose the kind of mistakes, hidden assumptions, naive ablation and training stability issues that only surface while translating research papers into a working code.

<br>

### Learnings
1. large scale has its own challenges and highlights inefficiencies which could have been overlooked on a smaller scale.
2. Develop low level code execution thinking/intuition to really understand what is happening under the hood. This helps early identify/prevent 
    1. OOM errors [multiprocessing in spawn mode pickles data. This triggers reading entire data as it treats memmap as standard numpy aaray] 
    2. slow execution [if you don't leverage multithreading in tokenization or use more threads than CPU cores] 
    3. data leaks 
    4. silent failures [not cloning y and then updating values at EOT masks]

### Dataset Preparation

#### Which dataset
- GPT2 was trained on the `WebText` data but the dataset wasn't released by OpenAI. 
- So `openwebtext` dataset available on the [huggingface](https://huggingface.co/datasets/Skylion007/openwebtext#plain_text) was chosen as the author claims to be open-source replication of the WebText dataset.
- The dataset is in a `List[str]` format

#### Loading the dataset
- There are two popular approaches:
    1. load the entire dataset in memory during `__init__` function itself [works when data fits in the memory] and tokenize in the __init__ function itself.
    2. read file and tokenize content in the `__getitem__` function - this is efficient for large scale data that doesn't fit in memory [often used in computer vision].

    <br>
    <details>
    <summary> First Attempt with first approach </summary>

    - [GPTDataset Permalink](https://github.com/arkothiwala/gpt2/blob/485df8462a08bf6d6b0fb2f6c5c66ea13a0f8fef/gpt/modules/data/dataset.py#L19)
        - Implements 1st approach where we load the entire dataset in memory and tokenize in the `__init__` step only.
        - For a given index in `__getitem__`, it returns a `substring` from the ith document [of arbitrary size].
            ```python
            class GPTDataset(torch.utils.data.Dataset):
            def __init__(self, raw_data_path, num_threads=os.cpu_count(), min_seq_len=32, max_seq_len=512):
                self.raw_data_df = GPTDataUtils.load_raw_data(path=raw_data_path)
                self.tokenizer = tiktoken.get_encoding(encoding_name="gpt2")
                self.tokens = self.tokenizer.encode_batch(text=self.raw_data_df['text'], num_threads=num_threads)

                self.min_seq_len = min_seq_len
                self.max_seq_len = max_seq_len

            def __len__(self):
                return len(self.raw_data_df)

            def __getitem__(self, index):
                x = self.tokens[index]
                y = self.tokens[index][1:] + [self.tokenizer._special_tokens.get('<|endoftext|>')]
                doc_seq_len = len(x)
                if doc_seq_len < self.min_seq_len:
                    raise ValueError(f"{index}th document is shorter than the required min_seq_len={self.min_seq_len}")
                
                # ensure uniform distribution of sequence lengths
                curr_seq_len = np.random.randint(low=self.min_seq_len, high=min(self.max_seq_len, doc_seq_len)+1)
                end_idx = min(curr_seq_len, doc_seq_len)
                return torch.tensor(x[:end_idx]), torch.tensor(y[:end_idx])
            ```
        - Turns out, mathematical advantages of varying input sequence length (prevention of attention dilution, better positional encoding generalization or scope of curriculum learning - training on shorter text initially and longer text later) are there, the hardware constraints weigh more as varying input sequence length
            - Introduces inefficiency due to padding - it wastes compute.
            - Modern compilers havily optimize execution graphs based on static tensor shapes. Dynamic shapes force the compiler to re-evaluate the memory allocation and execution plan on the fly, `which destroys throughput`


    </details>

    - **Problem** - Then how do you handle documents of varying length in model training?
    - **Solution** - Sequence packing - concatenate documents into `<DOC1>_<EOT>_<DOC2>_<EOT>_<DOC3>` format.

    <br>
    <details>
    <summary> Second Attempt with sequence packing </summary>

    - [GPTDatasetSequancePacking Permalink](https://github.com/arkothiwala/gpt2/blob/485df8462a08bf6d6b0fb2f6c5c66ea13a0f8fef/gpt/modules/data/dataset.py#L58)
    - This implements sequence packing but still loads entire data into memory

        ```python
        class GPTDatasetSequancePacking(torch.utils.data.Dataset):
            def __init__(self, raw_data_path, num_threads=os.cpu_count(), max_seq_len=512):
                self.raw_data_df = GPTDataUtils.load_raw_data(path=raw_data_path)
                self.tokenizer = tiktoken.get_encoding(encoding_name="gpt2")

                # BUG - creates copy
                self.raw_data_df['text'] += '<|endoftext|>'
                self.tokens = self.tokenizer.encode_batch(
                    text=self.raw_data_df['text'], 
                    num_threads=num_threads,  
                    allowed_special={"<|endoftext|>"}
                )
                self.tokens_flattened = torch.tensor([x for sub in self.tokens for x in (*sub, 100)])
                self.max_seq_len = max_seq_len

            def __len__(self):
                return len(self.tokens_flattened) // self.max_seq_len

            def __getitem__(self, index):
                start_idx = index*self.max_seq_len
                end_idx = (index+1)*self.max_seq_len
                if end_idx >= len(self.tokens_flattened)-1:
                    start_idx = len(self.tokens_flattened) - self.max_seq_len - 1
                    end_idx = len(self.tokens_flattened) - 1
                    
                x = self.tokens_flattened[start_idx:end_idx]
                # clone is important otherwise, value of x will change when we set y[EOT_mask]=-100 shown below
                y = self.tokens_flattened[start_idx+1:end_idx+1].clone()
                
                EOT_mask = (x == self.tokenizer.eot_token)
                y[EOT_mask] = torch.tensor(-100) # -100 is default ignore index for CrossEntropyLoss
                return x, y
        ```
    </details>

    - **Problem 1**: Tokenization is a slow process and it runs on CPU. It will starve a GPU if we do it during pre-training.
        - we shouldn't run it on GPU due to PCIe bottlenecks, string matching instead of matrix multiplication, dynamic memory allocation due to variable length string array
        - GPU may go idle if it needs to wait as workers are busy tokenizing content in the `__getitem__` function. This increases training duration and wastes GPU compute which your the costliest resource.
    - **Problem 2**: How to load larger-than-RAM data with RAM-like performance?
    - **Solution**: Two step solution
        1. tokenize the data and store it in seqeunce packing format `<DOC1>_<EOT>_<DOC2>_<EOT>_<DOC3>` in a binary file
        2. read the binary file in a streaming mode [instead of loading in memory by creating copy, directly read from disk]

    <br>
    <details>
    <summary> Third Attempt with pre-tokenization + streaming read </summary>
        
    - [GPTDatasetBinFile permalink](https://github.com/arkothiwala/gpt2/blob/485df8462a08bf6d6b0fb2f6c5c66ea13a0f8fef/gpt/modules/data/dataset.py#L110)
    </details>  
    
    - This gave a blazing fast reading speed while being able to load the whole data eventually in streaming mode.

    <details>
    <summary> General Learnings </summary>
    
    - Tokenization done in batch mode is a lot more efficient than being done sequentially. 
        - Benchmarking done with different num_threads parameter shows interesting results.
        - Despite my laptop having 12 cores. We got the best performance at num_thread = 8
        - For more details, one can take a look at the [tiktokenizer batch_encode benchmarking report](../benchmarking/readme.md)
    - initialising memmap in __init__ along with `num_workers>0` may create [issues](https://claude.ai/share/d0450442-b974-451e-bd21-dec8907e3464) depending on the worker process start method [fork, spawn or forkserver].
        - This can blow up the memory if the workers make in-memory copy due to type conversions or some other reasons.
        - Or cause synchronization issues due to race condition if one worker writes to the memmap
        - File discreptor may get exhausted if you are reading from multiple files with high no of workers.
    - [PyTorch Memmap Multiprocessing Trap](https://gemini.google.com/share/d075e619575a): When using DataLoader with num_workers > 0, initializing an np.memmap in the main process (like inside __init__ or __len__) causes the system to crash exactly at iter(dataloader). This happens because PyTorch uses pickle to send the dataset to worker processes, and pickle mistakenly reads the entire disk-backed memmap into RAM to serialize it. The Fix: Calculate dataset length using os.path.getsize() and strictly delay calling np.memmap until inside the worker-executed __getitem__ method so the file reference is never pickled.
    </details>
