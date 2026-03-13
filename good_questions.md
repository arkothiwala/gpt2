1. Why did GPT2 move layer norm from post-layers [as in GPT1] to pre-layer?
2. Why do we divide Q@K_Transpose with sqrt(D_k)?
3. Which residual network wrights initializations do we need to scale by sqrt(2*n_layers)? why? and why not by sqrt(n_layers)?
4. what do we need to take care off when calculating variance in torch?
5. what is the difference between causal mask and padding mask?
6. For which applications K and V dimensions aren't necessarilu needs to be same as Q dimension?
7. Why He initialization is done with sqrt(2/fan_in) and not sqrt(1/fan_in) as one would expect? [activation aware v/s unaware initialization]? What is fan_in and how is it different from n_in in linear layer? For which layers fan_in ~= n_in?
8. How would you decide what should be the value of a max_norm for a learnable embedding layer?
9. Why do glorot initialization initialize by sqrt(2/(n_in + n_out))?
10. When to use batch_norm vs layer_norm? Why do we use layer_norm in the transformers? What would happen if we don't use layer_norm?
11. 