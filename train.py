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
import os
from datetime import datetime
import logging
import math
torch.set_float32_matmul_precision('high')

# Remove basicConfig to avoid conflict with custom handlers
# logging.basicConfig(level=logging.DEBUG)

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

    # Logging config
    now = datetime.now()
    log_dir = os.path.join("training_runs", now.strftime("%Y%m%d_%H%M%S"))
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "model.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # default level

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file_path)

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    logger.info(args)
    logger.info(f"Loading config from: {args.model_yaml}")

    with open(args.model_yaml, "r") as f:
        exp_config = yaml.safe_load(f)
        logger.debug(f"Config: {exp_config}")

    ################################################
    # Setting common variables
    device = torch.device("cuda") if torch.cuda.is_available() else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    # device = 'cpu'
    cross_entropy_loss = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
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
    logger.info("loaded total {} parameters".format(sum(p.numel() for p in model.parameters())))
    logger.debug(model.parameters)
    logger.debug(f"Model parameters device: {next(model.parameters()).device}")
    logger.debug(f"embedding parameters device: {next(model.get_submodule('embedding').parameters()).device}")

    
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
    logger.info("Datasets loaded successfully")
    
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
        persistent_workers = False,
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
        persistent_workers = False,
        pin_memory_device = "cuda" if torch.cuda.is_available() else ''
    )

    #########################################################################
    ################## dataloader testing and forward pass ##################
    #########################################################################

    batch = next(iter(train_dl))
    logger.debug(f"batch = {batch}")
    logger.debug(f"batch shape = {batch[0].shape}")
    batch[0] = batch[0].to(device)
    batch[1] = batch[1].to(device)

    with torch.no_grad():
        logger.debug(model(x=torch.randint(low=1, high=model.vocab_size, size=(2,512), device=device)).shape)

    ########################################################################
    #################### Optim + LR Scheduler Config #######################
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

    total_optimizer_steps = len(train_ds) // exp_config.get("training").get("global_batch_size")
    cosine_ann_steps = total_optimizer_steps - exp_config.get("scheduler").get("warmup_steps")
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer=optimizer, 
                start_factor=exp_config.get("scheduler").get("warmup_start_factor"), 
                end_factor=exp_config.get("scheduler").get("warmup_end_factor"), 
                total_iters=exp_config.get("scheduler").get("warmup_steps")
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, 
                T_max=cosine_ann_steps
            )
        ],
        milestones=[exp_config.get("scheduler").get("warmup_steps")],
    )

    #########################################################################
    ############################# training loop #############################
    #########################################################################

    # LEARNINGS: Unlike GPT1 where they trained the model for 100 epochs, modern LLMs are mostly trained for 1 epochs to avoid overfitting.

    # for epoch in tqdm(range(exp_config.get("training").get("epochs")), desc="epoch progress"):
    # set model in the training model
    model.train()
    model.compile() if torch.cuda.is_available() else model
    
    # zero_grad
    total_accumulated = 0
    global_batch_loss = 0 #torch.tensor(0.0, requires_grad=False)
    optimizer.zero_grad()
    
    # iterate through the micro_batch_size
    global_step = 0
    for batch_idx, (batch_x_train, batch_y_train) in enumerate(tqdm(train_dl, desc="epoch's batch progress")):
        batch_size = batch_x_train.shape[0]
        total_accumulated += batch_size

        # print(f"pushing batch_x_train to {device}")

        # MISTAKE - I wasn't assigning it back to the variable leading to `RuntimeError: Placeholder storage has not been allocated on MPS device!`
        batch_x_train = batch_x_train.to(device=device)
        batch_y_train = batch_y_train.to(device=device) if torch.cuda.is_available() else batch_y_train

        # print(f"batch_x_train.max() = {batch_x_train.max().max()}")

        # forward pass
        batch_logits = model(batch_x_train)
        # print(f"batch_logits.shape = {batch_logits.shape}")
        # print(f"batch_y_train.shape = {batch_y_train.shape}")
        # print(f"batch_y_train.max() = {batch_y_train.max().max()} | batch_y_train.min() = {batch_y_train.min()}")
        # micro_batch_loss = cross_entropy_loss(input=batch_logits.permute(0,2,1).contiguous(), target=batch_y_train)
        batch_logits = batch_logits.to(device='cpu') if not torch.cuda.is_available() else batch_logits
        # batch_y_train = batch_y_train.to(device='cpu')
        assert batch_logits.device == batch_y_train.device, f"batch_logits device {batch_logits.device} and batch_y_train device {batch_y_train.device} are not the same"
        micro_batch_loss = cross_entropy_loss(input=batch_logits.view(-1, batch_logits.size(-1)), target=batch_y_train.view(-1))
        # print(f"micro_batch_loss.shape = {micro_batch_loss.shape}")
        logger.debug(f"micro_batch_loss = {micro_batch_loss}")
        global_batch_loss += micro_batch_loss.detach()*batch_size

        # We devide micro_batch_loss by accumulation_steps to get scaled loss because the gradients will keep accumulating for accumulation_steps up to number of micro batches times before we do an optimizer step
        # this is equivalent to doing global batch size gradient accumulation in small chunks
        accumulation_steps = global_batch_size // batch_size
        micro_batch_loss_scaled = micro_batch_loss / accumulation_steps
        micro_batch_loss_scaled.backward()

        # handle gradient accumulation
        if total_accumulated % global_batch_size == 0:
            global_batch_loss = global_batch_loss / global_batch_size
            try:
                perplexity = math.exp(global_batch_loss.item())
            except OverflowError:
                perplexity = float('inf')
            logger.info(f"Step {global_step} | global_batch_loss = {global_batch_loss} | perplexity = {perplexity} | lr = {optimizer.param_groups[0]['lr']}")
            optimizer.step()
            lr_scheduler.step()

            global_step += 1
            if global_step % exp_config.get("training").get("checkpoint_interval") == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{global_step}.pt")
                logger.info(f"Saving checkpoint to {checkpoint_path}")
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'global_step': global_step,
                    'batch_idx': batch_idx, # Index in the current epoch/dataloader
                    'torch_rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                }
                if torch.cuda.is_available():
                    checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
                torch.save(checkpoint, checkpoint_path)

            optimizer.zero_grad()
            total_accumulated = 0
            global_batch_loss = 0

    
    # update weights as of the last batch of the epoch
    if total_accumulated > 0:
        optimizer.step()
        lr_scheduler.step()

        optimizer.zero_grad()
        total_accumulated = 0
        global_batch_loss = 0

    # # run validation after n_epoch interval
    # if epoch % exp_config.get("training").get("valid_epoch_interval") == 0:
    #     model.eval()
    #     # run batch predictions

        
    # # do forward pass, backward pass, accumulate gradient and update weights when global_batch_size is met

