import numpy as np

def histogram_elements(arr, num_bins):
    """
    Groups elements of an array into histogram bins.

    This is different from a standard histogram, which counts elements per bin.
    This function returns the actual elements that fall into each bin.

    Args:
        arr (np.ndarray): The input array.
        num_bins (int): The number of bins to create.

    Returns:
        list: A list of numpy arrays, where each array contains the elements
              from the input array for a corresponding bin.
    """
    # Sort the array in-place to conserve memory.
    arr.sort()

    # Determine bin edges. Using the sorted array makes finding min/max trivial.
    bin_edges = np.linspace(arr[0], arr[-1], num_bins + 1)

    # Find the indices where the array should be split.
    # np.searchsorted is highly efficient on sorted arrays.
    # We search for the internal bin edges (excluding the min and max).
    split_points = np.searchsorted(arr, bin_edges[1:-1])

    # Split the sorted array into bins.
    # np.split returns a list of arrays. For a large input array, these
    # will be views, not copies, making it memory-efficient.
    binned_elements = np.split(arr, split_points)

    return binned_elements

if __name__ == '__main__':
    # --- Example Usage ---
    # NOTE: The user's array is 10**9 elements.
    # We use a smaller array for this demonstration.
    # The principle is the same and is efficient for large arrays.
    print("--- Demonstrating with a smaller array (1,000,000 elements) ---")
    # Create a sample array with a normal distribution
    sample_arr = np.random.randn(1_000_000)
    num_bins = 10

    print(f"Input array shape: {sample_arr.shape}")
    print(f"Number of bins: {num_bins}")

    # Get the binned elements
    # Pass a copy if you want to preserve the original unsorted array
    binned_data = histogram_elements(sample_arr.copy(), num_bins)

    print(f"\nOutput is a list of {len(binned_data)} arrays (one for each bin).")

    # --- Verification ---
    # 1. Check that the total number of elements is conserved.
    total_elements_in_bins = sum(len(b) for b in binned_data)
    print(f"Original number of elements: {len(sample_arr)}")
    print(f"Total elements in all bins: {total_elements_in_bins}")
    assert len(sample_arr) == total_elements_in_bins

    # 2. Check the contents of a few bins
    print("\n--- Verifying bin contents ---")
    # Get bin edges for verification
    bin_edges = np.linspace(np.min(sample_arr), np.max(sample_arr), num_bins + 1)

    for i, bin_array in enumerate(binned_data):
        if len(bin_array) > 0:
            min_in_bin = np.min(bin_array)
            max_in_bin = np.max(bin_array)
            lower_edge = bin_edges[i]
            upper_edge = bin_edges[i+1]
            print(f"Bin {i:2d}: {len(bin_array):7d} elements. "
                  f"Range: [{min_in_bin:8.4f}, {max_in_bin:8.4f}]. "
                  f"Expected edge: [{lower_edge:8.4f}, {upper_edge:8.4f}]")
            # All elements in the bin should be >= the lower edge
            assert np.all(bin_array >= lower_edge)
            # All elements in the bin should be <= the upper edge
            # Note: due to floating point precision, the very last element of the last bin
            # might be exactly the upper edge. Other bins should be < upper_edge.
            if i < num_bins - 1:
                assert np.all(bin_array < upper_edge)
            else: # Last bin
                assert np.all(bin_array <= upper_edge)
    print("\nVerification successful.")

    # --- Note on the large array ---
    print("\n--- Note for your 1,000,000,000 element array ---")
    print("The provided function `histogram_elements` is designed to be memory-efficient.")
    print("It sorts the array in-place and uses views instead of copies where possible.")
    print("It should be able to handle your large array, provided it fits into your machine's RAM (approx. 8GB for 10^9 float64s).")
