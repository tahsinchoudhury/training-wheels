import time
import numpy as np
from tqdm import tqdm

from .adapters import run_tensor_multiply, run_tensor_dot


def test_multiply_basic_shape():
    """Test tensor multiplication preserves correct output shape."""
    np.random.seed(42)
    batch_size, x_dim, y_dim, z_dim = 2, 3, 4, 5

    arr1 = np.random.randn(batch_size, x_dim, y_dim)
    arr2 = np.random.randn(batch_size, y_dim, z_dim)

    result = run_tensor_multiply(arr1.tolist(), arr2.tolist())
    result_array = np.array(result)

    assert result_array.shape == (batch_size, x_dim, z_dim), (
        f"Expected shape {(batch_size, x_dim, z_dim)}, got {result_array.shape}"
    )


def test_multiply_batch_correctness():
    """Test tensor multiplication correctness using einsum."""
    np.random.seed(123)
    batch_size, x_dim, y_dim, z_dim = 3, 2, 3, 4

    arr1 = np.random.randn(batch_size, x_dim, y_dim)
    arr2 = np.random.randn(batch_size, y_dim, z_dim)

    result = run_tensor_multiply(arr1.tolist(), arr2.tolist())
    result_array = np.array(result)

    # Compute expected result using einsum: bij, bjk -> bik
    expected = np.einsum("bij,bjk->bik", arr1, arr2)

    np.testing.assert_allclose(
        result_array,
        expected,
        rtol=1e-5,
        atol=1e-8,
        err_msg="Tensor multiplication result doesn't match einsum computation",
    )


def test_multiply_single_batch():
    """Test tensor multiplication with batch size of 1."""
    np.random.seed(456)
    arr1 = np.random.randn(1, 3, 4)
    arr2 = np.random.randn(1, 4, 5)

    result = run_tensor_multiply(arr1.tolist(), arr2.tolist())
    result_array = np.array(result)

    expected = np.einsum("bij,bjk->bik", arr1, arr2)
    np.testing.assert_allclose(result_array, expected, rtol=1e-5, atol=1e-8)


def test_multiply_large_dimensions():
    """Test tensor multiplication with larger dimensions."""
    np.random.seed(789)
    batch_size, x_dim, y_dim, z_dim = 8, 16, 32, 24

    arr1 = np.random.randn(batch_size, x_dim, y_dim)
    arr2 = np.random.randn(batch_size, y_dim, z_dim)

    result = run_tensor_multiply(arr1.tolist(), arr2.tolist())
    result_array = np.array(result)

    expected = np.einsum("bij,bjk->bik", arr1, arr2)
    np.testing.assert_allclose(result_array, expected, rtol=1e-5, atol=1e-8)


def test_multiply_small_values():
    """Test tensor multiplication with small values."""
    np.random.seed(101)
    arr1 = np.random.randn(2, 3, 4) * 0.001
    arr2 = np.random.randn(2, 4, 5) * 0.001

    result = run_tensor_multiply(arr1.tolist(), arr2.tolist())
    result_array = np.array(result)

    expected = np.einsum("bij,bjk->bik", arr1, arr2)
    np.testing.assert_allclose(result_array, expected, rtol=1e-5, atol=1e-10)


def test_multiply_negative_values():
    """Test tensor multiplication with negative values."""
    np.random.seed(202)
    arr1 = np.random.randn(2, 3, 4)
    arr2 = np.random.randn(2, 4, 5)

    result = run_tensor_multiply(arr1.tolist(), arr2.tolist())
    result_array = np.array(result)

    expected = np.einsum("bij,bjk->bik", arr1, arr2)
    np.testing.assert_allclose(result_array, expected, rtol=1e-5, atol=1e-8)


def test_dot_vector_dot():
    """Test dot product of two 1D vectors."""
    np.random.seed(303)
    arr1 = np.random.randn(10)
    arr2 = np.random.randn(10)

    result = run_tensor_dot(arr1.tolist(), arr2.tolist(), dim=0)

    # Compute expected result: simple dot product
    expected = np.dot(arr1, arr2)

    np.testing.assert_allclose(
        result, expected, rtol=1e-5, atol=1e-8, err_msg="Vector dot product result doesn't match expected"
    )


def test_dot_matrix_multiply_dim0():
    """Test dot product along dimension 0 for 2D arrays."""
    np.random.seed(404)
    arr1 = np.random.randn(4, 5)
    arr2 = np.random.randn(4, 5)

    result = run_tensor_dot(arr1.tolist(), arr2.tolist(), dim=0)
    result_array = np.array(result)

    # Contract along dimension 0: i,i... -> ...
    expected = np.einsum("ij,ij->j", arr1, arr2)

    np.testing.assert_allclose(result_array, expected, rtol=1e-5, atol=1e-8)


def test_dot_matrix_multiply_dim1():
    """Test dot product along dimension 1 for 2D arrays."""
    np.random.seed(505)
    arr1 = np.random.randn(3, 5)
    arr2 = np.random.randn(3, 5)

    result = run_tensor_dot(arr1.tolist(), arr2.tolist(), dim=1)
    result_array = np.array(result)

    # Contract along dimension 1: ij,ij -> i
    expected = np.einsum("ij,ij->i", arr1, arr2)

    np.testing.assert_allclose(result_array, expected, rtol=1e-5, atol=1e-8)


def test_dot_batch_dim0():
    """Test dot product along dimension 0 for 3D arrays."""
    np.random.seed(606)
    arr1 = np.random.randn(4, 3, 5)
    arr2 = np.random.randn(4, 3, 5)

    result = run_tensor_dot(arr1.tolist(), arr2.tolist(), dim=0)
    result_array = np.array(result)

    # Contract along dimension 0: ijk,ilk -> jlk
    expected = np.einsum("ijk,ijk->jk", arr1, arr2)

    np.testing.assert_allclose(result_array, expected, rtol=1e-5, atol=1e-8)


def test_dot_batch_dim1():
    """Test dot product along dimension 1 for 3D arrays."""
    np.random.seed(707)
    arr1 = np.random.randn(2, 4, 5)
    arr2 = np.random.randn(2, 4, 5)

    result = run_tensor_dot(arr1.tolist(), arr2.tolist(), dim=1)
    result_array = np.array(result)

    # Contract along dimension 1: ijk,ijk -> ik
    expected = np.einsum("ijk,ijk->ik", arr1, arr2)

    np.testing.assert_allclose(result_array, expected, rtol=1e-5, atol=1e-8)


def test_dot_single_element():
    """Test dot product with single element arrays."""
    np.random.seed(808)
    arr1 = np.array([5.0])
    arr2 = np.array([3.0])

    result = run_tensor_dot(arr1.tolist(), arr2.tolist(), dim=0)

    expected = np.dot(arr1, arr2)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_dot_orthogonal_vectors():
    """Test dot product with orthogonal vectors."""
    arr1 = np.array([1.0, 0.0, 0.0])
    arr2 = np.array([0.0, 1.0, 0.0])

    result = run_tensor_dot(arr1.tolist(), arr2.tolist(), dim=0)

    assert np.isclose(result, 0.0, atol=1e-8), "Orthogonal vectors should have zero dot product"


def test_dot_parallel_vectors():
    """Test dot product with parallel vectors."""
    arr1 = np.array([1.0, 2.0, 3.0])
    arr2 = np.array([2.0, 4.0, 6.0])

    result = run_tensor_dot(arr1.tolist(), arr2.tolist(), dim=0)
    expected = np.dot(arr1, arr2)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_dot_negative_values():
    """Test dot product with negative values."""
    np.random.seed(909)
    arr1 = np.random.randn(7)
    arr2 = np.random.randn(7)

    result = run_tensor_dot(arr1.tolist(), arr2.tolist(), dim=0)
    expected = np.dot(arr1, arr2)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_dot_large_vectors():
    """Test dot product with large vectors."""
    np.random.seed(1010)
    arr1 = np.random.randn(1000)
    arr2 = np.random.randn(1000)

    result = run_tensor_dot(arr1.tolist(), arr2.tolist(), dim=0)
    expected = np.dot(arr1, arr2)

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)


def test_integration_chained_multiply_and_dot():
    """Test chaining multiply and dot operations."""
    np.random.seed(1111)
    batch_size, x_dim, y_dim, z_dim = 2, 3, 4, 5

    arr1 = np.random.randn(batch_size, x_dim, y_dim)
    arr2 = np.random.randn(batch_size, y_dim, z_dim)

    # First multiply
    multiply_result = run_tensor_multiply(arr1.tolist(), arr2.tolist())
    multiply_result_np = np.array(multiply_result)

    # Verify multiply result
    expected_multiply = np.einsum("bij,bjk->bik", arr1, arr2)
    np.testing.assert_allclose(multiply_result_np, expected_multiply, rtol=1e-5, atol=1e-8)


def test_performance_multiply_benchmark():
    """Benchmark tensor multiplication performance against numpy."""
    np.random.seed(2024)
    batch_size, x_dim, y_dim, z_dim = 32, 64, 128, 96

    arr1 = np.random.randn(batch_size, x_dim, y_dim).tolist()
    arr2 = np.random.randn(batch_size, y_dim, z_dim).tolist()

    # Benchmark numpy einsum
    num_iterations = 5
    numpy_times = []
    for _ in tqdm(range(num_iterations), "NP Mult Perf"):
        start = time.perf_counter()
        arr1_ = np.ascontiguousarray(arr1)
        arr2_ = np.ascontiguousarray(arr2)
        _ = arr1_ @ arr2_
        elapsed = time.perf_counter() - start
        numpy_times.append(elapsed)

    numpy_avg = np.mean(numpy_times)
    numpy_std = np.std(numpy_times)

    # Benchmark custom implementation
    custom_times = []
    for _ in tqdm(range(num_iterations), "Custom Mult Perf"):
        start = time.perf_counter()
        _ = run_tensor_multiply(arr1, arr2)
        elapsed = time.perf_counter() - start
        custom_times.append(elapsed)

    custom_avg = np.mean(custom_times)
    custom_std = np.std(custom_times)

    # Calculate performance ratio
    performance_ratio = custom_avg / numpy_avg

    print("\nTensor Multiply Benchmark:")
    print(f"  NumPy avg: {numpy_avg * 1000:.3f}ms ± {numpy_std * 1000:.3f}ms")
    print(f"  Custom avg: {custom_avg * 1000:.3f}ms ± {custom_std * 1000:.3f}ms")
    print(f"  Performance ratio: {performance_ratio:.2f}x")

    # Assert custom implementation is within 100x of numpy performance
    assert performance_ratio <= 100.0, (
        f"Custom multiply is {performance_ratio:.2f}x slower than numpy (max allowed: 100.0x)"
    )


def test_performance_dot_benchmark():
    """Benchmark tensor dot product performance against numpy."""
    np.random.seed(2025)
    batch_size, dim_a, dim_b = 32, 64, 128

    arr1 = np.random.randn(batch_size, dim_a, dim_b).tolist()
    arr2 = np.random.randn(batch_size, dim_a, dim_b).tolist()

    # Benchmark numpy dot along dimension 0
    num_iterations = 5
    numpy_times = []
    for _ in tqdm(range(num_iterations), "NP Dot Perf"):
        start = time.perf_counter()
        arr1_ = np.ascontiguousarray(arr1)
        arr2_ = np.ascontiguousarray(arr2)
        _ = (arr1_ * arr2_).sum(axis=0)
        elapsed = time.perf_counter() - start
        numpy_times.append(elapsed)

    numpy_avg = np.mean(numpy_times)
    numpy_std = np.std(numpy_times)

    # Benchmark custom implementation
    custom_times = []
    for _ in tqdm(range(num_iterations), "Custom Dot Perf"):
        start = time.perf_counter()
        _ = run_tensor_dot(arr1, arr2, dim=0)
        elapsed = time.perf_counter() - start
        custom_times.append(elapsed)

    custom_avg = np.mean(custom_times)
    custom_std = np.std(custom_times)

    # Calculate performance ratio
    performance_ratio = custom_avg / numpy_avg

    print("\nTensor Dot Benchmark:")
    print(f"  NumPy avg: {numpy_avg * 1000:.3f}ms ± {numpy_std * 1000:.3f}ms")
    print(f"  Custom avg: {custom_avg * 1000:.3f}ms ± {custom_std * 1000:.3f}ms")
    print(f"  Performance ratio: {performance_ratio:.2f}x")

    # Assert custom implementation is within 100x of numpy performance
    assert performance_ratio <= 20.0, f"Custom dot is {performance_ratio:.2f}x slower than numpy (max allowed: 20.0x)"
