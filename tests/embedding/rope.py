import unittest
import torch
import math
from gpt.modules.embedding.rope import RotaryPositionalEmbeddings
# --- 2. The Tests ---
class TestRoPE(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.rope = RotaryPositionalEmbeddings()
    
    def setUp(self):
        # Standard test parameters
        self.batch_size = 2
        self.seq_len = 10
        self.dim = 64 # Must be even
        self.rtol = 1e-4 # Tolerance for float comparisons
        self.atol = 1e-5

    def test_01_shape_preservation(self):
        """Basic: Output shape must match input shape."""
        print("="*100)
        print("test_01_shape_preservation")
        print("="*100)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        out = self.rope(x)
        print(f"x.shape = {x.shape}")
        print(f"x_rope.shape = {out.shape}")
        self.assertEqual(x.shape, out.shape)

    def test_02_norm_preservation(self):
        """Math Property: Rotation must not change the length (norm) of the vector."""
        print("="*100)
        print("test_02_norm_preservation")
        print("="*100)
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        out = self.rope(x)
        
        norm_in = torch.norm(x, dim=-1)
        norm_out = torch.norm(out, dim=-1)
        
        print(f"x_norm = {norm_in.abs().sum()}")
        print(f"x_rope_norm = {norm_out.abs().sum()}")
        
        # Check if norms are equal within tolerance
        diff = (norm_in - norm_out).abs().max()
        self.assertTrue(diff < self.rtol, f"Norm changed by {diff} (should be 0)")

    def test_03_linearity(self):
        """Math Property: RoPE(a + b) == RoPE(a) + RoPE(b)."""
        print("="*100)
        print("test_03_linearity")
        print("="*100)
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        y = torch.randn(self.batch_size, self.seq_len, self.dim)
        
        out_sum = self.rope(x + y)
        sum_out = self.rope(x) + self.rope(y)
        print(f"f(x+y) = {out_sum.abs().sum()}")
        print(f"f(x)+f(y) = {out_sum.abs().sum()}")
        
        diff = (out_sum - sum_out).abs().max()
        self.assertTrue(diff < self.rtol, f"Linearity violated by {diff}")

    def test_04_relative_distance_invariance(self):
        """
        The Golden Rule: Dot product (Attention Score) should depend only on relative distance.
        score(pos 0, pos k) should equal score(pos t, pos t+k)

        NOTE - Essential condition for this to hold true is input embedding at both timestamp is same
        example sentence - My name is Ashutosh. My height is 178 cm.
        # Here, we should compare embeddings of 'My'->'is' in first and 2nd statement only becaue the embedding will be same and location delta (b/w 'my' and 'is') is 2 for both sentences.
        """
        
        print("="*100)
        print("test_04_relative_distance_invariance")
        print("="*100)
        k = 5 # distance
        t = 2 # shift
        
        # Create a single vector 'v'
        v = torch.randn(1, 1, self.dim) 
        
        # We need to manually simulate being at different positions
        # We'll construct a sequence of length t+k+1
        # and place 'v' at specific spots to rotate it.
        seq_len_needed = t + k + 1
        
        # Create a dummy full sequence to pass to the function
        dummy_input = torch.zeros(1, seq_len_needed, self.dim)
        
        # Case 1: Q at pos 0, K at pos k
        # To test this, we put 'v' at pos 0 and 'v' at pos k in the input
        dummy_input[0, 0, :] = v
        dummy_input[0, k, :] = v
        
        out = self.rope(dummy_input)
        q_0 = out[0, 0, :]
        k_k = out[0, k, :]
        score_1 = torch.dot(q_0, k_k)
        
        # Case 2: Q at pos t, K at pos t+k
        # Put 'v' at pos t and pos t+k
        dummy_input.zero_()
        dummy_input[0, t, :] = v
        dummy_input[0, t+k, :] = v
        
        out_shifted = self.rope(dummy_input)
        q_t = out_shifted[0, t, :]
        k_tk = out_shifted[0, t+k, :]
        score_2 = torch.dot(q_t, k_tk)

        print(f"Case 1: Q at pos 0, K at pos k | score = {score_1}")
        print(f"Case 2: Q at pos t, K at pos t+k | score = {score_2}")

        # Comparison
        diff = (score_1 - score_2).abs()
        self.assertTrue(diff < self.rtol, 
                        f"Relative distance failed. \nScore(0, {k})={score_1:.4f}\nScore({t}, {t+k})={score_2:.4f}")

    def test_05_frequency_decay(self):
        """
        Math Property: High frequencies (start of vector) should rotate fast.
        Low frequencies (end of vector) should rotate slow.
        """
        print("="*100)
        print("test_05_frequency_decay")
        print("="*100)

        x = torch.ones(1, 100, self.dim) # Sequence length 100
        out = self.rope(x)
        
        # Index 0 (Highest Freq, theta=1): Should vary wildly across sequence
        start_variance = out[0, :, 0].var()
        
        # Index -1 (Lowest Freq, theta=small): Should vary very little (almost constant)
        # Note: In sliced, the last element is the pair of the middle element. 
        # So we check index dim/2 - 1 (last freq, first component)
        last_freq_idx = (self.dim // 2) - 1
        end_variance = out[0, :, last_freq_idx].var()
        
        print(f"\nHigh Freq Variance: {start_variance:.4f}")
        print(f"Low Freq Variance:  {end_variance:.4f}")
        
        self.assertTrue(start_variance > end_variance * 50, 
                        "High frequency dimensions are not rotating significantly faster than low freq ones.")

if __name__ == '__main__':
    unittest.main()