import torch
import torch.nn.functional as F
import math
import wandb
def run_lr_finder(
    model,
    optimizer,
    dataloader,
    device,
    micro_batch_size,
    global_batch_size,
    start_lr=1e-6,
    end_lr=1.0,
    num_steps=400,  # number of OPTIMIZER steps
    beta=0.98,
    max_grad_norm=1.0
):
    """
    LR range test for GPT-style training.

    Args:
        micro_batch_size: batch size per forward pass
        global_batch_size: effective batch size (after accumulation)
        num_steps: number of optimizer steps (NOT micro-steps)

    Returns:
        lrs, losses
    """

    assert global_batch_size % micro_batch_size == 0, \
        "global_batch_size must be divisible by micro_batch_size"

    grad_accum_steps = global_batch_size // micro_batch_size

    print(f"Using grad_accum_steps = {grad_accum_steps}")

    model.train()
    
    start_lr = float(start_lr)
    end_lr = float(end_lr)

    # LR schedule setup (per optimizer step)
    lr = start_lr
    mult = (end_lr / start_lr) ** (1 / num_steps)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    avg_loss = 0.0
    best_loss = float("inf")

    losses = []
    lrs = []

    optimizer.zero_grad()

    optimizer_step = 0
    micro_step = 0
    global_batch_cumm_loss = 0.0
    global_logits_variance = 0.0
    data_iter = iter(dataloader)

    while optimizer_step < num_steps:
        batch_x, batch_y = next(data_iter)
        inputs = batch_x.to(device, non_blocking=True)
        targets = batch_y.to(device, non_blocking=True)

        # Forward
        logits = model(inputs)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        
        global_batch_cumm_loss += loss.item()
        global_logits_variance += logits.var(dim=-1).mean().item()

        # Normalize for accumulation
        loss = loss / grad_accum_steps
        loss.backward()

        micro_step += 1

        # Perform optimizer step
        if micro_step % grad_accum_steps == 0:
            global_batch_loss = global_batch_cumm_loss / grad_accum_steps

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            optimizer_step += 1
            
            avg_loss = beta * avg_loss + (1 - beta) * global_batch_loss
            smoothed_loss = avg_loss / (1 - beta ** (optimizer_step))

            # Record BEFORE optimizer step
            losses.append(smoothed_loss)
            lrs.append(lr)

            # Early stopping on divergence
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            if math.isnan(smoothed_loss) or (smoothed_loss > 4 * best_loss):
                print(f"Stopping early at step {optimizer_step}, LR={lr:.2e}")
                break

            # Update LR AFTER optimizer step
            lr *= mult
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            if optimizer_step % 1 == 0:
                print(
                    f"OptStep {optimizer_step:04d} | "
                    f"LR {lr:.2e} | "
                    f"Loss {smoothed_loss:.4f}"
                )
                wandb.log({"lr": lr, "loss": smoothed_loss, "optimizer_step": optimizer_step, "grad_norm": grad_norm, "logit_variance": global_logits_variance / grad_accum_steps})
            global_batch_cumm_loss = 0.0
            global_logits_variance = 0.0

    return lrs, losses