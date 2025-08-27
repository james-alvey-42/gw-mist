

import numpy as np

def precompute_centered_null(ts_bin_H0_epsilon):
    """
    Performs the one-time centering of the null distribution samples.

    This function takes the raw samples from a null hypothesis distribution,
    computes the mean of these samples, and subtracts it, effectively
    centering the distribution around zero. This is a useful pre-processing
    step before calculating empirical p-values.

    Args:
        ts_bin_H0_epsilon (np.ndarray): A 2D NumPy array where each row is a
                                        sample from the null hypothesis and
                                        each column is a different bin or feature.

    Returns:
        np.ndarray: The centered null distribution, with the same shape as the input.
    """
    eps_t_mean = np.mean(ts_bin_H0_epsilon, axis=0)
    return ts_bin_H0_epsilon - eps_t_mean


def calculate_pvalue_from_ref(observed_stats, centered_null_dist):
    """
    Calculates empirical p-values against a pre-computed centered null distribution.

    This function compares a 1D array of observed test statistics against a
    pre-computed null distribution to calculate empirical p-values. The p-value
    for each statistic is the proportion of samples in the null distribution
    that are at least as extreme as the observed statistic.

    Args:
        observed_stats (np.ndarray): A 1D NumPy array of observed test statistics.
        centered_null_dist (np.ndarray): The pre-computed centered null distribution
                                         (output of precompute_centered_null).

    Returns:
        np.ndarray: A 1D NumPy array of corresponding p-values.
    """
    # Count how many null statistics are greater than or equal to the observed statistics.
    # The `observed_stats` array is broadcasted across all rows of `centered_null_dist`.
    counts = np.sum(centered_null_dist >= observed_stats, axis=0)

    # Calculate the empirical p-value using the formula (counts + 1) / (n_samples + 1)
    # to avoid p-values of zero.
    n_samples = len(centered_null_dist)
    p_values = (counts + 1) / (n_samples + 1)

    return p_values

# --- Example Usage ---
if __name__ == '__main__':
    # This block will only run when the script is executed directly
    # It serves as a demonstration of how to use the functions.

    # 1. Create some dummy data for the null hypothesis distribution
    #    (e.g., 1000 samples, 50 bins/features per sample)
    ts_bin_H0_epsilon = np.random.randn(1000, 50)

    # 2. Pre-compute the centered null distribution (do this once)
    centered_null = precompute_centered_null(ts_bin_H0_epsilon)
    print(f"Shape of centered null distribution: {centered_null.shape}")

    # 3. Create a dummy 1D array of observed statistics
    my_observation = np.random.randn(50) + 1.5  # An observation that is slightly shifted
    print(f"Shape of observed stats: {my_observation.shape}")


    # 4. Calculate the p-values for the observation
    p_values = calculate_pvalue_from_ref(my_observation, centered_null)

    print(f"\nFirst 5 observed stats: {my_observation[:5]}")
    print(f"First 5 corresponding p-values: {p_values[:5]}")
    print(f"Shape of resulting p-values: {p_values.shape}")


