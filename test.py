import numpy as np

# Based on your description, ts_bin_H0_epsilon is a 2D array where each row
# is a sample from the null hypothesis and each column is a bin.
# For demonstration, we'll create a placeholder for it.
nbins = 10  # Assuming 10 bins for the example
ts_bin_H0_epsilon = np.random.randn(199936, nbins)

def pvalue_array_eps(dat):
    """
    Calculates p-values for a single 1D array of test statistics against a
    2D reference distribution.

    This function is analogous to your `pvalue_grid_eps` but is specifically
    for a single 1D `dat` array of shape (nbins,).

    Args:
        dat: A 1D numpy array of observed test statistics, shape (nbins,).

    Returns:
        A 1D numpy array of p-values, one for each bin, shape (nbins,).
    """
    # Center the reference distribution by subtracting the mean of each bin
    eps_t_mean = np.mean(ts_bin_H0_epsilon, axis=0)
    eps_t_ref = ts_bin_H0_epsilon - eps_t_mean

    # Compare the centered reference distribution (N_samples, nbins) with
    # the observed data `dat` (nbins,). `dat` is broadcasted across all samples.
    # The result is a boolean array of shape (N_samples, nbins).
    # We sum along axis=0 to count how many samples in `eps_t_ref` were
    # greater than or equal to `dat` for each bin.
    counts = np.sum(eps_t_ref >= dat, axis=0)

    # Calculate the p-value for each bin
    return (counts + 1) / (len(eps_t_ref) + 1)

# --- Example Usage ---
if __name__ == '__main__':
    # Example 1: An array of observed test statistics for each bin.
    observed_statistics = np.array([0.1, -0.5, 1.2, 2.5, -1.8, 0.3, 0.9, -0.1, 3.1, 0.0])
    p_values = pvalue_array_eps(observed_statistics)

    print("--- Example ---")
    print(f"Number of bins: {len(observed_statistics)}")
    print(f"Observed statistics: {np.round(observed_statistics, 2)}")
    print(f"Resulting p-values per bin: {np.round(p_values, 4)}")
    print("\nNote: A larger statistic (e.g., 3.1) results in a smaller p-value, as expected.")