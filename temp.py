import numpy as np

def calculate_sample_entropy(subsequence, m, r):
    N = len(subsequence)
    B = 0.0
    A = 0.0

    for i in range(N - m):
        for j in range(i + 1, N - m + 1):
            if max(abs(subsequence[i:i + m] - subsequence[j:j + m])) <= r:
                A += 1

                if j < N - m:
                    B += 1

    return -np.log(A / B)

def calculate_composite_multiscale_entropy(time_series, m, r, scale):
    entropy_values = []
    subsequence_length = len(time_series) // scale

    for i in range(scale):
        start = i * subsequence_length
        end = start + subsequence_length
        subsequence = time_series[start:end]

        entropy = calculate_sample_entropy(subsequence, m, r)
        entropy_values.append(entropy)

    return np.mean(entropy_values)

def calculate_refined_composite_multiscale_entropy(time_series, m, r, scale):
    composite_entropies = []
    for i in range(1, scale + 1):
        composite_entropy = calculate_composite_multiscale_entropy(time_series, m, r, i)
        composite_entropies.append(composite_entropy)

    return np.mean(composite_entropies)

# Example usage
time_series = np.random.rand(1000)  # Replace with your own time series data
m = 2  # Length of subsequences
r = 0.5  # Tolerance threshold
scale = 5  # Number of scales

rcmse = calculate_refined_composite_multiscale_entropy(time_series, m, r, scale)
print("RCMSE:", rcmse)
