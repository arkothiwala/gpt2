import tiktoken
import torch
import yaml
import argparse
import numpy as np
import wandb
from gpt.modules.models.gpt2 import GPT2Model
from gpt.modules.data.dataset import GPTDatasetBinFile
from gpt.modules.data.utils import DataUtils
from gpt.modules.utils.logger import CustomLogger
from torch.utils.data import DataLoader
from gpt.modules.optimizer.lr_finder import run_lr_finder
from torch.amp import autocast, GradScaler
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm
import os
from datetime import datetime
import logging
import math
import random
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

def get_modulewise_grad_stats(model, norm_type=2, prefix="grad/"):
    d = {}

    def process_name(name):
        return (
            name.replace("transformer_layers.", "L")
                .replace("linear_expansion", "exp")
                .replace("linear_projection", "proj")
                .replace("layer_norm", "ln")
                .replace("out_proj","out")
        )

    for name, module in model.named_modules():
        total_grad_sq = 0.0
        total_param_sq = 0.0
        total_params = 0

        for param in module.parameters(recurse=False):
            if param.grad is None:
                continue

            g = param.grad.data
            w = param.data

            total_grad_sq += g.norm(2).item() ** 2
            total_param_sq += w.norm(2).item() ** 2
            total_params += param.numel()

        if total_params == 0:
            continue

        grad_norm = total_grad_sq ** 0.5
        weight_norm = total_param_sq ** 0.5

        # ✅ Key metrics
        grad_rms = grad_norm / (total_params ** 0.5)
        update_ratio = grad_norm / (weight_norm + 1e-12)

        base = prefix + process_name(name)

        d[base + "/norm"] = grad_norm              # raw (size dependent)
        d[base + "/rms"] = grad_rms                # ✅ comparable across layers
        d[base + "/update_ratio"] = update_ratio   # ✅ learning speed

    return d

def get_parameterwise_grad_stats(model, prefix="grad/"):
    d = {}

    def process_name(name):
        return (
            name.replace("transformer_layers.", "L")
                .replace("linear_expansion", "exp")
                .replace("linear_projection", "proj")
                .replace("layer_norm", "ln")
                .replace(".weight", ".w")
                .replace(".bias", ".b")
                .replace("out_proj","out")
        )

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        g = param.grad.data
        w = param.data

        numel = param.numel()

        grad_norm = g.norm(2).item()
        weight_norm = w.norm(2).item()

        # ✅ Comparable metrics
        grad_rms = grad_norm / (numel ** 0.5)
        update_ratio = grad_norm / (weight_norm + 1e-12)

        base = prefix + process_name(name)

        d[base + "/norm"] = grad_norm
        d[base + "/rms"] = grad_rms
        d[base + "/updt_rto"] = update_ratio

    return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_yaml", help="configs/<exp_config>.yaml file path")
    parser.add_argument("--checkpoint_path", default=None, help="checkpoint path to resume training from")
    args = parser.parse_args()

    # Experiment config and logger
    now = datetime.now()
    exp_run_time = now.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("training_runs", exp_run_time)
    os.makedirs(exp_dir, exist_ok=True)
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = CustomLogger.get_logger(base_dir=exp_dir)
    
    
    #########################################################################
    ############################## Load Config ##############################
    #########################################################################
    
    logger.info(args)
    logger.info(f"Loading config from: {args.model_yaml}")

    with open(args.model_yaml, "r") as f:
        exp_config = yaml.safe_load(f)
        logger.debug(f"Config: {exp_config}")

    # assign autocast_dtype based on device capabilities
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        autocast_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        autocast_dtype = torch.float16
    else:
        autocast_dtype = None
    
    #########################################################################
    ############################ load checkpoint ############################
    #########################################################################

    if args.checkpoint_path is not None:
        logger.info(f"Resuming training from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, weights_only=False)
        exp_config['training']['global_batch_size'] = checkpoint.get('global_batch_size', exp_config['training']['global_batch_size'])
        exp_config['training']['micro_batch_size'] = checkpoint.get('micro_batch_size', exp_config['training']['micro_batch_size'])
        exp_config['checkpoint'] = {
            'path': args.checkpoint_path,
            'global_step': checkpoint.get('global_step', 0),
            'batch_idx': checkpoint.get('batch_idx', 0),
            'scheduler_state_dict': checkpoint.get('scheduler_state_dict', None),
        }
        logger.debug(f"Updated Config after loading checkpoint: {exp_config}")

    ################################################
    # Wandb Init
    ################################################
    wandb.init(
        project="gpt2-from-scratch",
        config=exp_config,
        name=f"{exp_run_time}",
    )

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
            logger=logger
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
        # num_workers=exp_config.get("training").get("dataloader_num_workers"),
        # collate_fn = None,
        # pin_memory = True if torch.cuda.is_available() else False,
        drop_last = True,
        # timeout = 0,
        # worker_init_fn = DataUtils.worker_init_fn,
        # multiprocessing_context = None,
        # generator = None,
        # prefetch_factor = exp_config.get("data").get("prefetch_factor"),
        # persistent_workers = True,
        # pin_memory_device = "cuda" if torch.cuda.is_available() else ''
    )

    valid_dl = DataLoader(
        dataset=valid_ds,
        batch_size=exp_config.get("validation").get("micro_batch_size"),
        shuffle=False,
        # sampler: Sampler | Iterable | None = None,
        # batch_sampler: Sampler[Sequence] | Iterable[Sequence] | None = None,
        num_workers=exp_config.get("training").get("dataloader_num_workers"),
        # collate_fn = None,
        pin_memory = True if torch.cuda.is_available() else False,
        drop_last = True,
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
        drop_last = True,
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
        input_ids = torch.randint(low=1, high=model.vocab_size, size=(2,512), device=device)
        logger.debug(model(x=input_ids).shape)
        
        # PROFILER to check and confirm if flash attention is being used or not.
        from torch.profiler import profile, ProfilerActivity

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
                output = model(input_ids)

        # Look for flash attention kernels
        logger.info("checking profiler events for attention kernels")
        for event in prof.key_averages():
            if "attention" in event.key.lower() or "flash" in event.key.lower() or "sdpa" in event.key.lower():
                logger.info(event.__dict__)#.key, event.cuda_time_total)

    ########################################################################
    #################### Optim + LR Scheduler Config #######################
    ########################################################################


    if exp_config.get("optimizer").get("type") == "adamw":
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr = exp_config.get("optimizer").get("lr"),
            weight_decay = exp_config.get("optimizer").get("weight_decay"),
            betas=(
                exp_config.get("optimizer").get("beta1"),
                exp_config.get("optimizer").get("beta2")
            ),
        )

    total_optimizer_steps = len(train_ds) // exp_config.get("training").get("global_batch_size")
    logger.info(f"Total optimizer steps per epoch: {total_optimizer_steps}")
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

    if args.checkpoint_path is not None:
        if checkpoint.get('scheduler_state_dict', None) is not None:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # set RNG states
            torch.set_rng_state(checkpoint['torch_rng_state'])
            if torch.cuda.is_available() and 'cuda_rng_state' in checkpoint and checkpoint['cuda_rng_state'] is not None:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
            if 'numpy_rng_state' in checkpoint and checkpoint['numpy_rng_state'] is not None:
                np.random.set_state(checkpoint['numpy_rng_state'])
            if 'python_rng_state' in checkpoint and checkpoint['python_rng_state'] is not None:
                random.setstate(checkpoint['python_rng_state'])
            
            logger.info(f"Loaded LR scheduler state from checkpoint: {args.checkpoint_path}")
            logger.info(f"Resuming training from global step: {checkpoint.get('global_step', 'N/A')} and batch index: {checkpoint.get('batch_idx', 'N/A')}")
        else:
            logger.warning(f"No LR scheduler state found in checkpoint: {args.checkpoint_path}. Starting LR scheduler from scratch.")

    #########################################################################
    ############################# training loop #############################
    #########################################################################

    # LEARNINGS: Unlike GPT1 where they trained the model for 100 epochs, modern LLMs are mostly trained for 1 epochs to avoid overfitting.

    # for epoch in tqdm(range(exp_config.get("training").get("epochs")), desc="epoch progress"):
    # set model in the training model
    model.train()
    model.compile() if torch.cuda.is_available() else model
    grad_scaler = GradScaler() if torch.cuda.is_available() else None
    # run_lr_finder(
    #     model=model, 
    #     optimizer=optimizer, 
    #     dataloader=valid_dl, 
    #     device=device, 
    #     micro_batch_size=exp_config.get("training").get("micro_batch_size"),
    #     global_batch_size=exp_config.get("training").get("global_batch_size"),
    #     start_lr=exp_config.get("lr_finder").get("start_lr"), 
    #     end_lr=exp_config.get("lr_finder").get("end_lr"), 
    #     num_steps=exp_config.get("lr_finder").get("num_steps"),
    #     max_grad_norm=exp_config.get("training").get("max_grad_norm")
    # )
    
    logger.info(f"torch.backends.cuda.flash_sdp_enabled={torch.backends.cuda.flash_sdp_enabled()}")
    
    # zero_grad
    total_accumulated = 0
    global_batch_loss = 0 #torch.tensor(0.0, requires_grad=False)
    global_logits_variance = 0
    global_logits_max = 0
    global_logits_min = 0
    global_logits_mean = 0
    optimizer.zero_grad(set_to_none=True)
    
    # iterate through the micro_batch_size
    global_step = 0 if args.checkpoint_path is None else checkpoint["global_step"]
    batch_idx_start = 0 if args.checkpoint_path is None else checkpoint["batch_idx"]
    logger.info(f"Starting training loop from global step {global_step}, batch index {batch_idx_start}, lr {optimizer.param_groups[0]['lr']}, global batch size {global_batch_size}, micro batch size {exp_config.get('training').get('micro_batch_size')}")
    # MISTAKE - to do resumable training, I was just advancing enumerate index to batch_idx_start but it doesn't advance the dataloader [NOT-DESIRED]
    train_iter = iter(train_dl)
    if batch_idx_start > 0:
        logger.info(f"Advancing dataloader to batch index {batch_idx_start} to resume training")
        for _ in tqdm(range(batch_idx_start), desc="Advancing dataloader progress"):
            next(train_iter)
    for batch_idx, (batch_x_train, batch_y_train) in enumerate(tqdm(train_iter, desc="epoch's batch progress"), start=batch_idx_start):
        batch_size = batch_x_train.shape[0]
        total_accumulated += batch_size

        # print(f"pushing batch_x_train to {device}")

        # MISTAKE - I wasn't assigning it back to the variable leading to `RuntimeError: Placeholder storage has not been allocated on MPS device!`
        batch_x_train = batch_x_train.to(device=device)
        batch_y_train = batch_y_train.to(device=device) if torch.cuda.is_available() else batch_y_train
        
        logger.debug(f"batch_x_train.dtype = {batch_x_train.dtype} | batch_y_train.dtype = {batch_y_train.dtype}")

        # print(f"batch_x_train.max() = {batch_x_train.max().max()}")
        
        # We devide micro_batch_loss by accumulation_steps to get scaled loss because the gradients will keep accumulating for accumulation_steps up to number of micro batches times before we do an optimizer step
        # this is equivalent to doing global batch size gradient accumulation in small chunks
        accumulation_steps = global_batch_size // batch_size

        # forward pass
            
        if autocast_dtype in [torch.float16, torch.bfloat16]:
            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                with autocast(device_type=device.type, dtype=autocast_dtype):
                    batch_logits = model(batch_x_train)
                    micro_batch_loss = cross_entropy_loss(input=batch_logits.view(-1, batch_logits.size(-1)), target=batch_y_train.view(-1))
                    micro_batch_loss_scaled = micro_batch_loss / accumulation_steps
        else:
            batch_logits = model(batch_x_train)
            # micro_batch_loss = cross_entropy_loss(input=batch_logits.permute(0,2,1).contiguous(), target=batch_y_train)
            # batch_logits = batch_logits.to(device='cpu') if not torch.cuda.is_available() else batch_logits
            # batch_y_train = batch_y_train.to(device='cpu')
            micro_batch_loss = cross_entropy_loss(input=batch_logits.view(-1, batch_logits.size(-1)), target=batch_y_train.view(-1))
            micro_batch_loss_scaled = micro_batch_loss / accumulation_steps
            
        if autocast_dtype == torch.float16:
            micro_batch_loss_scaled = grad_scaler.scale(micro_batch_loss_scaled)
        micro_batch_loss_scaled.backward()
        assert batch_logits.device == batch_y_train.device, f"batch_logits device {batch_logits.device} and batch_y_train device {batch_y_train.device} are not the same"
        # print(f"micro_batch_loss.shape = {micro_batch_loss.shape}")
        logger.debug(f"micro_batch_loss = {micro_batch_loss}")

        global_batch_loss += micro_batch_loss.detach().item()
        global_logits_variance += batch_logits.var(dim=-1).mean().item()
        global_logits_max += batch_logits.max(dim=-1).values.mean().item()
        global_logits_min += batch_logits.min(dim=-1).values.mean().item()
        global_logits_mean += batch_logits.mean(dim=-1).mean().item()
        

        # handle gradient accumulation
        if total_accumulated % global_batch_size == 0:
            ####################################################################################
            ############################# Log Metrics to WandB #################################
            ####################################################################################
            global_batch_loss = global_batch_loss / accumulation_steps
            global_logits_variance = global_logits_variance / accumulation_steps
            global_logits_max = global_logits_max / accumulation_steps
            global_logits_min = global_logits_min / accumulation_steps
            global_logits_mean = global_logits_mean / accumulation_steps
            try:
                perplexity = math.exp(global_batch_loss)
            except OverflowError:
                perplexity = float('inf')
            unclipped_grad_norm_early = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
            unclipped_grad_norm = torch.nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None])
            logger.info(f"Step {global_step} | global_batch_loss = {global_batch_loss} | perplexity = {perplexity} | lr = {optimizer.param_groups[0]['lr']} | unclipped_grad_norm = {unclipped_grad_norm}")
            
            paramwise_grad_stats = get_parameterwise_grad_stats(model)
            paramwise_total_grad_norm = torch.linalg.vector_norm(torch.tensor(list(paramwise_grad_stats.values())), ord=2)
            
            wandb.log({
                "train/loss": global_batch_loss.item() if isinstance(global_batch_loss, torch.Tensor) else global_batch_loss,
                "train/perplexity": perplexity,
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/grad_norm": unclipped_grad_norm.item() if isinstance(unclipped_grad_norm, torch.Tensor) else unclipped_grad_norm,
                "train/step": global_step,
                "train/logits_variance": global_logits_variance,
                "train/logits_max": global_logits_max,
                "train/logits_min": global_logits_min,
                "train/logits_mean": global_logits_mean,
                **paramwise_grad_stats,
                "train/derived_grad_norm": paramwise_total_grad_norm.item() if isinstance(paramwise_total_grad_norm, torch.Tensor) else paramwise_total_grad_norm

            })
            
            ####################################################################################
            ################ Optimizer.step() | LR_scheduler.step() | Reset ####################
            ####################################################################################
            # Unscales the gradients of optimizer's assigned params in-place
            if autocast_dtype == torch.float16:
                grad_scaler.unscale_(optimizer)
                
            # clip the gradients
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=exp_config.get("training").get("max_grad_norm"), error_if_nonfinite=False)
            
            # take optimizer step and update the lr scheduler
            if autocast_dtype == torch.float16:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            total_accumulated = 0
            global_batch_loss = 0
            global_logits_variance = 0
            global_logits_max = 0
            global_logits_min = 0
            global_logits_mean = 0
            
            ####################################################################################
            ################## save checkpoints after each checkpoint_interval #################
            ####################################################################################
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
                    'python_rng_state': random.getstate(),
                }
                if torch.cuda.is_available():
                    checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state_all()
                torch.save(checkpoint, checkpoint_path)

            ####################################################################################
            ################### run validation after each validation_interval ##################
            ####################################################################################
            if global_step % exp_config.get("training").get("validation_interval") == 0:
                logger.info(f"Running validation at global step {global_step}")
                valid_step = 0
                global_valid_loss = 0
                gloabl_valid_steps = exp_config.get("validation").get("global_batch_size") // exp_config.get("validation").get("micro_batch_size")
                model.eval() # set to eval mode before running validation loop
                for valid_batch_idx, (batch_x_valid, batch_y_valid) in enumerate(tqdm(valid_dl, desc="validation batch progress"), start=0):
                    batch_x_valid = batch_x_valid.to(device=device)
                    batch_y_valid = batch_y_valid.to(device=device) if torch.cuda.is_available() else batch_y_valid

                    with torch.no_grad():
                        if autocast_dtype in [torch.float16, torch.bfloat16]:
                            with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
                                with autocast(device_type=device.type, dtype=autocast_dtype):
                                    valid_logits = model(batch_x_valid)
                                    valid_loss = cross_entropy_loss(input=valid_logits.view(-1, valid_logits.size(-1)), target=batch_y_valid.view(-1))
                        else:
                            valid_logits = model(batch_x_valid)
                            valid_loss = cross_entropy_loss(input=valid_logits.view(-1, valid_logits.size(-1)), target=batch_y_valid.view(-1))
                        
                        global_valid_loss += valid_loss.item() / gloabl_valid_steps

                    valid_step += 1
                    if valid_step >= gloabl_valid_steps:
                        global_valid_perplexity = math.exp(global_valid_loss) if global_valid_loss < 700 else float('inf')
                        logger.info(f"Gloabal Step {global_step} | valid_loss = {global_valid_loss} | valid_perplexity = {global_valid_perplexity}")
                        wandb.log({
                            "validation/loss": global_valid_loss,
                            "validation/perplexity": global_valid_perplexity, # to avoid overflow in exp
                            "validation/step":global_step
                        })
                        break
                model.train()  # set back to train mode after validation
    wandb.finish()
        
    # # do forward pass, backward pass, accumulate gradient and update weights when global_batch_size is met

