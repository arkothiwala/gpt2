1. Dataset creation:
    - Implemented random length and random start for training data. Later switched to random length, fixed start [thinking model may not learn well if it starts at a random location]
    - We can use tiktokenizer encode_batch with optimal no of threads to reduce the encode time [did benchmarking as well - benchmarking/tokeniser_batch_encode.py]
    - in Dataset.init() I am creating copies of the data multiple times [] which would easily break with scale. Examples as following
        - self.raw_data_df['text'] = self.raw_data_df['text'] + '<|endoftext|>'
        - self.tokens = self.tokenizer.encode_batch(text=self.raw_data_df['text'], num_threads=num_threads,  allowed_special={"<|endoftext|>"})
        - self.tokens_flattened = torch.tensor(list(itertools.chain.from_iterable(self.tokens)))
    - BIN files - can be accessed w/o loading entire data in memory. This is crucial and important when we are training LLM on massive data. Earlier I was loading the entire parquet file in memory to prepare dataset and dataloader classes.
    - Can read data for x and y from bin file in a single operation then I can slice it
    - initialising memmap in __init__ can create [pickling issue](https://chatgpt.com/share/69b6ee7e-010c-800a-bd84-6cdaa9e7bc55)
    - [PyTorch Memmap Multiprocessing Trap](https://gemini.google.com/share/d075e619575a): When using DataLoader with num_workers > 0, initializing an np.memmap in the main process (like inside __init__ or __len__) causes the system to crash exactly at iter(dataloader). This happens because PyTorch uses pickle to send the dataset to worker processes, and pickle mistakenly reads the entire disk-backed memmap into RAM to serialize it. The Fix: Calculate dataset length using os.path.getsize() and strictly delay calling np.memmap until inside the worker-executed __getitem__ method so the file reference is never pickled.
1. Implemented masking when I shouldn't have
    - Implemeted dynamic length rnn masking [similar to done in timeseries V1 and V2], realised that this may not be correct. Checked whether one should use fixed length masking instead of dynamic given model context length is constant
    - Realised that for LLM pretraining we do fixed length **sequence packing**

    - I was applying masking to set prediction values to -100 where x was endoftext token => realised that I had applied EOT mask on list instead of tensor hence my code was breaking misrably
    - `y[EOT_mask] = -100` was being done w/o creating clone of the tensor. This would modify `tokens_flattened` in place.


MHA:
- I was initally thinking k_dim and v_dim as the dimensions on which we want embedding dimensions to project. It wasn't true. It is the dimensions in which one can expect K and V tensors to originally have shape. MHA internally projects them to match d_model dimensions
- I had applied scaling for MHA weights by torch.sqrt(1/self.n_layers) where as it should have been torch.sqrt(1/(2*self.n_layers)) because the network will have 2*self.n_layers number of residual blocks. Each block is increasing variance hence we need to normalise accordingly. Another way to think is FFN weight normalization is taking care of two residual connections - one before and one after the FFN
- Silly mistake -> I had not set `batch_first = True` explicitly and torch set `batch_first = False` by default
- In forward pass, I didn't pass x for all query, key and value arguments -> apperently they are different due to common API for MHA
- had not applied causal_mask which I realised later when I was preparing for the training.
- MHA returns two values -> post MHA output, and actual attention
- Identified a bug in torch -> if is_causal=True and need_weights=True [by default] => it would not apply causal mask and fail silently
    - option 1 - either set need_weights=False
    - [used this] option 2 - create causal mask and pass it in the attn_mask
- was doing `if self.MHA.in_proj_weight:` this throws error as boolean comparison goes for a toss when the tensor has a value -> should use `self.MHA.in_proj_weight is not None` instead because torch has a native support to check None
- I misunderstood the scaling factor for residual layers and was scaling all the weights -> we need to just scale the residual connection weights which are out projection layer from MHA and FFN.
- while applying scaling I did `self.MHA.out_proj.weight.data.mul_(scaling_factor)` -> doing weight.data.mul_ bypasses torch's autograd engine. this is discouraged in torch

Position Encoding:
- was returning numpy array instead of torch.tensor -> leading to issue when doing sum b/w tensor and ndarray
- was passing input X instead of input X's seq_len
- when creating torch.tensor() had to set dtype to float32 because torch by default set dtype to float32 v/s numpy sets it to float64 or double which makes operations incompatible
- was using fixed sinusoidal embeddings [as in attention is all you need] -> GPT2 is using learnd embeedings instead
- use register_buffer - I am calculating torch.triu for each forward pass for each layer -> this can make it super slow -> better to register at initialization and use it as required by the current seq_len

Embeddings:
- max_norm can be left None unless the training is unstable
- was creating another layer in the last layer to translate d_model to vocab_size. This will blow up the parameter space. GPT1 and GPT2 uses the same embedding metrics transpose for it.

Dropout:
- I was applying dropout w/o training v/s inference flag. This would have created dropout even in the inference mode [suboptimal]
- **NN.module manage bunch of such things which torch.nn.functional does not. ALWAYS be SUPER careful while using torch.nn.functional over equivalent torch.nn.Module**

Package:
- Repo wasn't a package earlier. Worker initialization was failing with `no module named gpt` error. This wasn't an issue in the main thread because it was initialized from the repo's root folder

Loss:
- cross_entropy_loss expects logits in (batch_size, categories, seq_len,...) format but we were providing it (batch_size, seq_len, categories...) causing failure in terms of expected shape mismatch error

Trainings:
- x.to(device) does not happen in_place -> need to assign it back
- `y_tensor[EOT_mask] = -100` was causing issue in the CrossEntropyLoss w/o any proper error message
- `y.clone()` needs to be done before setting `y[EOT_MASK] = -100` otherwise it updates the original tensor which cause out of index issue because there is no -100 index for embedidng layer
- -100 could be causing issue in the BCE calculation [failed even in CPU with following error]
    - `zsh: segmentation fault  python train.py --model_yaml`
    - `zsh: bus error  python train.py --model_yaml`
- using exp_config.get() can lead to breaking the code silently as bunch of parameters are passed as null and torch will have a default handling for it. use d[key] for such scenarios
- The issue is well known that cross entropy loss with logits and labels is unstable for classes > 50000 with MPS. So, I moved the logits to CPU for loss calculation and it worked [per row it is 100 MB data -> with batchsize of 8 -> it's 800MB data that needs to be transferred for each batch]
- gradient accumulation - due to which gradients are multipliers [n_accumulator_steps] of the desired gradients. We have to account for it and calculate gradients on a scaled micro batch loss

- Improving performance [290 hours to 90 hours total run time]:
    - turn off `need_weights=True` in MHA
    - add `is_causal=True` in MHA
    - model.compile() for inplace compilation
    - torch.set_float32_matmul_precision('high')


----------------------------------
Long due learnings
----------------------------------
- Optimizer - I misread optimiser config in GPT1 paper. Used Adam optmiser instead of AdamW. This was fine but the real mistake was using weight decay of 0.01 [good for AdamW but horrible for Adam]
- Training speed - Flash attention works only with FP16 [yet to figure out why so] but we were using FP32 in our code. Upon trying different techniques, didn't get the confidence that FP16 was actually being used
- Training speed - when using autocast we need scale and unscale only in FP16 but not in BF16.
- LR finder - select either (1) mid point of the steepest descent OR (2) 1/10th of the point when loss start to increase -> both of it indicated that I should use max_lr = 0.0001 only
- LR warmup steps - Ideally it should be 1 to 5% [for our data - 6-7B tokens with 128Mn param model with step size = 512 [as per chinchilla law, we should have used 2.5B tokens only], no of tokens per step is 0.5Mn and total steps are 13000] -> that translates in out warmup step to be close to 500-600 for an ideal start.
- Model Training - Logit Variance - When the model is trained, logits variance continue to increase as un learnt model logits start from uniform distribution. Once model starts to learn, it pushes correct logits up and start to supress the incorrect ones.
- Layers closer to residual network gets higher gradients as they are closer to the residual operation compared to the other counterparts [example - out projections in MHA and FFN have higher gradients compared to inprojection in MHA and linear expansion in FFN]
- L0 gradients are often 5-8x of L11 -> there is a gradual reduction -> If you see a huge disparity, something is going wrong.
- Importance of the right initialization - Initially my loss was starting with ~500 and came down to ~12 but this was wrong because theoretically random prediction would give ~11 loss as a starting point.
    - The issue - I wasn't doing initialization [mean=0, std=0.02] that was done in the GPT2 paper so the layers got initialized to the pytorch defaults.
    - Second BUG - my initialization was incorrect such that residual scaling was getting overwritten by linear layer scaling [0.02] -> this was happening because the forloop wasn't at layer/parameter level but it was at layer type level and there would be an overlap that would overwrite it
- Learnt something about variance - I was doubting whether the residual scaling should be baked into the standard deviation or you do 0.02 intialization and then multiply the weights with residual scaling, turned out the result is exactly same.
- Positional Encoding - This was the biggest trap. For a moment, I thought instead of learnable positional embeddings let's try using sinusoidal positional embeddings. It completely changed. While this didn't show any significant delta in the training loss, the delta in gradiant norm was very evident [one where gradients shaking massively v/s the other calm as still water] the gradient norm in the model training witg sinusoidal embeddings had gone to almost zero [~0.1-0.2].
    - reference runs for comparison - 20260324_143208, 20260324_111412.
    - tried adding layer norm after two embeddings are added but it didn't make the difference
- I wasn't using GeLU with tanh appoximation [default option in torch was None which multiplies with a constant value]
- I was confused about the 
- Logging - This has massive value
    - started logging gradient norm, logit stats [variance, min, max, mean], perplexity
    - started logging parameter level gradient norms to compare across transformer block layers and across types of parameters [in_proj, out pro, FFN exp, FFN proj, Layer Norm]
- Architecture creation/init - it is best if we create the layers in the order than model.named_modules and model.named_parameters will be in the same order.
- Torch is unstable - always use parameters explicitly and hand over less work to the torch. Examples as following:
    - if attn_mask is boolean, then depending on the torch version it may create different mask.
    - is_causal as just a flag was fine earlier but in the newer versions it also needs attn_mask
    - buggy and prone to silent failures - there was a bug in torch that if you pass `need_weights=True` and `is_causal=True` then it will actually not use causal mask w/o any warning or errors. This also happened because I was using torch 2.0.0 and the world had moved to much newer version. Still stable releases have bugs.
- Causal Mask - better create and pass the additive mask in the forward pass.
- CustomLayerNorm vs TorchLayerNorm - torch actually makes the fused layer dynamically if you use native torch layer norm instead of your own.
- Checkpointing - crucial to save the seeds with <>. Yet to figure out how to store dataloader state so that it gives the data it hasn't saved v/s the one already learnt. Dataloader is still unsolved problem for me [folks may have already done it]
- Gradient norm is important
- Even if you clip the norm, direction isn't solved -> so always try to keep the gradients under control [model learns little but gradients are high -> likely the direction is also wrong]
- Loss scaling for gradient accumulation - Loss should scaled according to the gradient accumulation steps otherwise the gradients become so high because loss.backward() add the gradients
- LR schedular - it is important because we need ramp up and cosine annealing. Critical part is what is max_lr [calculate it as per LR schedular] and warm_up steps [1-5% of the learning steps]