from async_timeout import timeout
import tiktoken
import torch
import yaml
import argparse
import numpy as np
from gpt.modules.models.gpt2 import GPT2Model
from gpt.modules.data.dataset import GPTDatasetBinFile
from gpt.modules.data.utils import DataUtils
from torch.utils.data import DataLoader
from tqdm import tqdm

import logging

# Set global level to DEBUG
logging.basicConfig(level=logging.DEBUG)

# Specific override for a noisy library
logging.getLogger('multiprocessing').setLevel(logging.INFO)

def validate_experiment_config(config):
    global_batch_size = config.get("training").get("global_batch_size")
    micro_batch_size = config.get("training").get("micro_batch_size")

    if global_batch_size % micro_batch_size != 0:
        raise ValueError("invalid config: global_batch_size % micro_batch_size must be 0")

    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_yaml", help="configs/<exp_config>.yaml file path")
    args = parser.parse_args()
    print(args)
    print(args.model_yaml)

    with open(args.model_yaml, "r") as f:
        exp_config = yaml.safe_load(f)
        print(exp_config)

    ################################################
    # Setting common variables
    device = torch.device("cuda") if torch.cuda.is_available() else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    device = 'cpu'
    cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    global_batch_size = exp_config.get("training").get("global_batch_size")
    ################################################

    assert validate_experiment_config(exp_config), "invalid experiment config"

    ########################################################################
    ############################# Model Config #############################
    ########################################################################
        
    if exp_config.get("model").get("type") == "gpt2":
        tokenizer = tiktoken.get_encoding(encoding_name=exp_config.get("tokenizer").get("type"))
        model = GPT2Model(
            d_model=exp_config.get("model").get("d_model"),
            n_heads=exp_config.get("model").get("n_heads"),
            n_layers=exp_config.get("model").get("n_layers"),
            vocab_size=exp_config.get("model").get("vocab_size"),
            context_length=exp_config.get("model").get("context_length"),
        )
    model.to(device)
    print("loaded total {} parameters".format(sum(p.numel() for p in model.parameters())))
    print(model.parameters)
    print(f"Model parameters device: {next(model.parameters()).device}")
    print(f"embedding parameters device: {next(model.get_submodule('embedding').parameters()).device}")


    ########################################################################
    ############################# Optim Config #############################
    ########################################################################


    if exp_config.get("optimizer").get("type") == "adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr = exp_config.get("optimizer").get("lr"),
            weight_decay = exp_config.get("optimizer").get("weight_decay"),
            betas=(
                exp_config.get("optimizer").get("beta1"),
                exp_config.get("optimizer").get("beta2")
            ),
        )
        
    ########################################################################
    ##################### Dataset and Dataloader setup #####################
    ########################################################################
        
    # Get the dataset
    train_ds = GPTDatasetBinFile(
        file_path=exp_config.get("data").get("train_bin_file_path"),
        context_length=exp_config.get("model").get("context_length"),
        binfile_dtype=np.uint16,
        eot_token=tokenizer.eot_token
    )
    valid_ds = GPTDatasetBinFile(
        file_path=exp_config.get("data").get("valid_bin_file_path"),
        context_length=exp_config.get("model").get("context_length"),
        binfile_dtype=np.uint16,
        eot_token=tokenizer.eot_token
    )
    test_ds = GPTDatasetBinFile(
        file_path=exp_config.get("data").get("test_bin_file_path"),
        context_length=exp_config.get("model").get("context_length"),
        binfile_dtype=np.uint16,
        eot_token=tokenizer.eot_token
    )
    print("Datasets loaded successfully")
    
    # Get the dataloaders
    train_dl = DataLoader(
        dataset=train_ds,
        batch_size=exp_config.get("training").get("micro_batch_size"),
        shuffle=True,
        # sampler: Sampler | Iterable | None = None,
        # batch_sampler: Sampler[Sequence] | Iterable[Sequence] | None = None,
        num_workers=exp_config.get("training").get("dataloader_num_workers"),
        # collate_fn = None,
        pin_memory = True if torch.cuda.is_available() else False,
        drop_last = False,
        timeout = 0,
        worker_init_fn = DataUtils.worker_init_fn,
        multiprocessing_context = None,
        generator = None,
        prefetch_factor = exp_config.get("data").get("prefetch_factor"),
        persistent_workers = True,
        pin_memory_device = "cuda" if torch.cuda.is_available() else ''
    )

    valid_dl = DataLoader(
        dataset=valid_ds,
        batch_size=exp_config.get("training").get("micro_batch_size"),
        shuffle=False,
        # sampler: Sampler | Iterable | None = None,
        # batch_sampler: Sampler[Sequence] | Iterable[Sequence] | None = None,
        num_workers=exp_config.get("training").get("dataloader_num_workers"),
        # collate_fn = None,
        pin_memory = True if torch.cuda.is_available() else False,
        drop_last = False,
        timeout = 0,
        worker_init_fn = DataUtils.worker_init_fn,
        multiprocessing_context = None,
        generator = None,
        prefetch_factor = exp_config.get("data").get("prefetch_factor"),
        persistent_workers = True,
        pin_memory_device = "cuda" if torch.cuda.is_available() else ''
    )

    test_dl = DataLoader(
        dataset=test_ds,
        batch_size=exp_config.get("training").get("micro_batch_size"),
        shuffle=False,
        # sampler: Sampler | Iterable | None = None,
        # batch_sampler: Sampler[Sequence] | Iterable[Sequence] | None = None,
        num_workers=exp_config.get("training").get("dataloader_num_workers"),
        # collate_fn = None,
        pin_memory = True if torch.cuda.is_available() else False,
        drop_last = False,
        timeout = 0,
        worker_init_fn = DataUtils.worker_init_fn,
        multiprocessing_context = None,
        generator = None,
        prefetch_factor = exp_config.get("data").get("prefetch_factor"),
        persistent_workers = True,
        pin_memory_device = "cuda" if torch.cuda.is_available() else ''
    )

    #########################################################################
    ################## dataloader testing and forward pass ##################
    #########################################################################

    batch = next(iter(train_dl))
    print(f"batch = {batch}")
    print(f"batch shape = {batch[0].shape}")
    batch[0] = batch[0].to(device)
    batch[1] = batch[1].to(device)

    print(model(x=torch.randint(low=1, high=model.vocab_size, size=(2,512), device=device)).shape)

    #########################################################################
    ############################# training loop #############################
    #########################################################################

    for epoch in tqdm(range(exp_config.get("training").get("epochs")), desc="epoch progress"):
        # set model in the training model
        model.train()
        
        # zero_grad
        total_accumulated = 0
        global_batch_loss = torch.tensor(0.0, requires_grad=False)
        optimizer.zero_grad()
        
        # iterate through the micro_batch_size
        for batch_x_train, batch_y_train in tqdm(train_dl, desc="epoch's batch progress"):
            batch_size = batch_x_train.shape[0]
            total_accumulated += batch_size

            # print(f"pushing batch_x_train to {device}")

            # MISTAKE - I wasn't assigning it back to the variable leading to `RuntimeError: Placeholder storage has not been allocated on MPS device!`
            batch_x_train = batch_x_train.to(device=device)
            batch_y_train = batch_y_train.to(device=device)

            # print(f"batch_x_train.max() = {batch_x_train.max().max()}")

            # forward pass
            batch_logits = model(batch_x_train)
            assert batch_logits.device == batch_y_train.device, f"batch_logits device {batch_logits.device} and batch_y_train device {batch_y_train.device} are not the same"
            # print(f"batch_logits.shape = {batch_logits.shape}")
            # print(f"batch_y_train.shape = {batch_y_train.shape}")
            # print(f"batch_y_train.max() = {batch_y_train.max().max()} | batch_y_train.min() = {batch_y_train.min()}")
            # micro_batch_loss = cross_entropy_loss(input=batch_logits.permute(0,2,1).contiguous(), target=batch_y_train)
            micro_batch_loss = cross_entropy_loss(input=batch_logits.view(-1, batch_logits.size(-1)), target=batch_y_train.view(-1))
            # print(f"micro_batch_loss.shape = {micro_batch_loss.shape}")
            print(f"micro_batch_loss = {micro_batch_loss}")
            global_batch_loss += micro_batch_loss
            micro_batch_loss.backward()

            # handle gradient accumulation
            if total_accumulated % global_batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
                total_accumulated = 0
                global_batch_loss = torch.tensor(0.0, requires_grad=False)
                print(f"global_batch_loss = {global_batch_loss}")

        
        # update weights as of the last batch of the epoch
        optimizer.step()
        optimizer.zero_grad()
        total_accumulated = 0
        global_batch_loss = torch.tensor(0.0, requires_grad=False)

        # # run validation after n_epoch interval
        # if epoch % exp_config.get("training").get("valid_epoch_interval") == 0:
        #     model.eval()
        #     # run batch predictions

            
        # # do forward pass, backward pass, accumulate gradient and update weights when global_batch_size is met

