[ ] during loss calculation ignore loss for EOT token's prediction [i.e. -100]

## Training
- all tensors must be on the same device
- should be able to compare doing training in CPU v.s. GPU
- Do inference to see how is the model doing
- Add a profiler to understand the bottleneck

## Training [2026-03-17]
- [done] add LR scheduler [ramp up to LR=2.5e-4 and then cosine annealing to 0]
- weight initialization of N(0,0.02) to be updated
- add modified version L2 regularization w=0.01 on all non-bias or gain weights
- [done] update gradients because in gloabal loss calculation -> we are summing up