import sys
import numpy as np
from gpt.modules.embedding.sinusoidal import SinusoidalPositionalEmbeddings

def test_unit_magnitude_check():
    """
    1. Unit Magnitude Check (Per Frequency Pair)
    Ensure each (2i, 2i+1) dimension pair forms a unit circle.
    Invariant: PE(pos, 2i)^2 + PE(pos, 2i+1)^2 = 1
    """
    n_dim = 128
    model = SinusoidalPositionalEmbeddings(n_dim)
    seq_len = 100
    pos_encodings = model.forward(seq_len) # (seq_len, n_dim)
    
    # Check multiple positions
    for p in [0, 1, 10, 50, seq_len-1]:
        pe = pos_encodings[p]
        for i in range(0, n_dim, 2):
            magnitude_sq = pe[i]**2 + pe[i+1]**2
            assert np.isclose(magnitude_sq, 1.0, atol=1e-6), f"Position {p}, Pair {i//2} failed magnitude check: {magnitude_sq}"

def test_dot_product_shift_invariance():
    """
    2. Dot Product Relative Shift Invariance Test
    Ensure dot products depend only on relative distance.
    Invariant: PE(p) . PE(q) = PE(p+k) . PE(q+k)
    """
    n_dim = 128
    model = SinusoidalPositionalEmbeddings(n_dim)
    # Generate a large enough range
    pos_encodings = model.forward(200)
    
    p, q, k = 10, 20, 50
    dot1 = np.dot(pos_encodings[p], pos_encodings[q])
    dot2 = np.dot(pos_encodings[p+k], pos_encodings[q+k])
    
    assert np.isclose(dot1, dot2, atol=1e-6), f"Dot product not shift invariant: {dot1} vs {dot2}"

def test_pairwise_dot_product_identity():
    """
    3. Pairwise Dot Product Identity Test
    Validate the cosine identity: PE(p) . PE(q) = sum_i cos(omega_i * (p - q))
    """
    n_dim = 128
    model = SinusoidalPositionalEmbeddings(n_dim)
    pos_encodings = model.forward(100)
    
    p, q = 10, 25
    dot_product = np.dot(pos_encodings[p], pos_encodings[q])
    
    # Manual computation of omega_i
    # Note: the implementation uses idx / (10000**(np.arange(n_dim)/n_dim))
    # For pairs (2i, 2i+1), they SHOULD share the same freq.
    # We'll check if the identity holds.
    # In sinusoidal PE, omega_i = 1 / (10000**(2*i/n_dim))
    omega = 1.0 / (10000**(np.arange(0, n_dim, 2) / n_dim))
    expected_dot = np.sum(np.cos(omega * (p - q)))
    
    assert np.isclose(dot_product, expected_dot, atol=1e-6), f"Dot product identity failed: {dot_product} vs {expected_dot}"

def test_linear_shift_operator():
    """
    4. Linear Shift Operator Test (Rotation Property)
    PE(p + delta) = R(delta) PE(p)
    """
    n_dim = 128
    model = SinusoidalPositionalEmbeddings(n_dim)
    pos_encodings = model.forward(100)
    
    p = 10
    delta = 5
    pe_p = pos_encodings[p]
    pe_p_delta = pos_encodings[p + delta]
    
    omega = 1.0 / (10000**(np.arange(0, n_dim, 2) / n_dim))
    
    for i in range(len(omega)):
        w = omega[i]
        d = delta
        # Rotation matrix for this pair
        # [ cos(wd)  sin(wd) ] [ sin(wp) ] = [ sin(wp)cos(wd) + cos(wp)sin(wd) ] = [ sin(w(p+d)) ]
        # [ -sin(wd) cos(wd) ] [ cos(wp) ]   [ -sin(wp)sin(wd) + cos(wp)cos(wd) ]   [ cos(w(p+d)) ]
        
        # In this implementation: pe[2i] is sine, pe[2i+1] is cosine
        v_p = np.array([pe_p[2*i], pe_p[2*i+1]])
        R = np.array([
            [np.cos(w*d), np.sin(w*d)],
            [-np.sin(w*d), np.cos(w*d)]
        ])
        v_rotated = R @ v_p
        v_actual = np.array([pe_p_delta[2*i], pe_p_delta[2*i+1]])
        
        assert np.allclose(v_rotated, v_actual, atol=1e-6), f"Rotation property failed for pair {i}"

def test_frequency_pair_equality():
    """
    5. Frequency Pair Equality Test
    Ensure even and odd dimensions share identical frequencies.
    """
    n_dim = 128
    model = SinusoidalPositionalEmbeddings(n_dim)
    
    # Based on the implementation:
    # get_sine_wave uses 10000**(np.arange(self.n_dim)/self.n_dim)
    # get_cos_wave uses the same.
    # However, the pair (2i, 2i+1) should use ONLY the even dim freq.
    
    # Let's inspect the waves directly for a position p=1
    pe = model.forward(2)[1] # pos 1
    
    for i in range(0, n_dim, 2):
        # If they share same freq w: pe[2i] = sin(w), pe[2i+1] = cos(w)
        # Then arcsin(pe[2i]) and arccos(pe[2i+1]) should be related
        magnitude_sq = pe[i]**2 + pe[i+1]**2
        assert np.isclose(magnitude_sq, 1.0, atol=1e-6), f"Freq mismatch: pair {i//2} doesn't form unit circle"

def test_constant_norm_per_pair():
    """
    6. Constant Norm Per Pair Across Positions
    Ensure magnitude does not oscillate across positions.
    """
    n_dim = 128
    model = SinusoidalPositionalEmbeddings(n_dim)
    seq_len = 500
    pos_encodings = model.forward(seq_len)
    
    for i in range(0, n_dim, 2):
        magnitudes = pos_encodings[:, i]**2 + pos_encodings[:, i+1]**2
        variance = np.var(magnitudes)
        assert np.isclose(variance, 0.0, atol=1e-12), f"Magnitude variance is high for pair {i//2}: {variance}"

def test_orthogonality_across_frequencies():
    """
    7. Orthogonality Check Across Different Frequencies
    Validate that different frequency pairs have low cross-correlation.
    """
    n_dim = 128
    model = SinusoidalPositionalEmbeddings(n_dim)
    seq_len = 1000
    pos_encodings = model.forward(seq_len)
    
    # Check correlation between pair 0 and pair 5
    pair_a = pos_encodings[:, 0:2].flatten()
    pair_b = pos_encodings[:, 10:12].flatten()
    
    correlation = np.corrcoef(pair_a, pair_b)[0, 1]
    # We expect some correlation but it shouldn't be 1 or -1 for distinct frequencies
    assert abs(correlation) < 0.5, f"High correlation between different frequency pairs: {correlation}"

def test_long_sequence_stability():
    """
    8. Long-Sequence Stability Test
    Validate extrapolation consistency for very large indices.
    """
    n_dim = 128
    model = SinusoidalPositionalEmbeddings(n_dim)
    large_seq_len = 100000
    # Just compute a few distant positions to avoid memory issues if forward always returns (seq_len, n_dim)
    # Note: current implementation forward(seq_len) returns (seq_len, n_dim)
    # For large seq_len this might be slow/memory intensive.
    # Let's test reaching a large index.
    
    try:
        pe_large = model.forward(large_seq_len)[large_seq_len-1]
        assert not np.any(np.isnan(pe_large)), "NaN values in large sequence"
        assert not np.any(np.isinf(pe_large)), "Inf values in large sequence"
        
        # Check magnitude at large index
        for i in range(0, n_dim, 2):
            magnitude_sq = pe_large[i]**2 + pe_large[i+1]**2
            assert np.isclose(magnitude_sq, 1.0, atol=1e-6)
    except MemoryError:
        pytest.skip("Skipping large sequence test due to memory limits")

def test_batch_consistency():
    """
    9. Batch Consistency Test
    Ensure vectorized and scalar computation match.
    (This is implicitly covered by how forward is implemented using np.arange)
    """
    n_dim = 128
    model = SinusoidalPositionalEmbeddings(n_dim)
    
    full_pe = model.forward(10)
    
    # If the implementation supported single index queries, we'd compare.
    # Current implementation always returns full matrix.
    # We can check if forward(5) matches first 5 rows of forward(10)
    pe_5 = model.forward(5)
    assert np.allclose(full_pe[:5], pe_5)

def test_dtype_consistency():
    """
    10. Device & dtype Consistency Test
    """
    n_dim = 128
    model = SinusoidalPositionalEmbeddings(n_dim)
    pe = model.forward(10)
    assert pe.dtype == np.float64 or pe.dtype == np.float32


def run_tests():
    tests = [
        test_unit_magnitude_check,
        test_dot_product_shift_invariance,
        test_pairwise_dot_product_identity,
        test_linear_shift_operator,
        test_frequency_pair_equality,
        test_constant_norm_per_pair,
        test_orthogonality_across_frequencies,
        test_long_sequence_stability,
        test_batch_consistency,
        test_dtype_consistency
    ]
    
    passed = 0
    failed = 0
    results = []
    
    print("Running Sinusoidal Positional Embedding Tests...")
    print("=" * 50)
    
    for test in tests:
        test_name = test.__name__
        try:
            test()
            print(f"✅ {test_name} PASSED")
            passed += 1
            results.append((test_name, "PASSED", None))
        except Exception as e:
            print(f"❌ {test_name} FAILED")
            # print(traceback.format_exc())
            failed += 1
            results.append((test_name, "FAILED", str(e)))
            
    print("=" * 50)
    print(f"Summary: {passed} passed, {failed} failed")
    
    if failed > 0:
        print("\nFailure Details:")
        for name, status, error in results:
            if status == "FAILED":
                print(f"- {name}: {error}")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    run_tests()
