import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    Load data from a file where each row contains space-separated numbers.
    Returns a list of numpy arrays.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            row = np.array([float(x) for x in line.strip().split()])
            data.append(row)
    
    if len(data) != 7:
        raise ValueError("Input file must contain exactly 7 rows.")
    
    return data

def plot_data(data):
    """
    Plot 7 individual rows + 2 aggregated sums.
    """
    plt.figure(figsize=(12, 7))
    
    # Plot individual rows
    for i, row in enumerate(data):
        x = np.arange(len(row))
        plt.plot(x, row, label=f'Row {i+1}')
    
    # Sum of first 3 rows
    min_len_first = min(len(data[i]) for i in range(3))
    first_three_sum = sum(data[i][:min_len_first] for i in range(3))
    x1 = np.arange(min_len_first)
    plt.plot(x1, first_three_sum, linestyle='--', linewidth=2, label='Sum Rows 1-3')
    
    # Sum of last 4 rows
    min_len_last = min(len(data[i]) for i in range(3, 7))
    last_four_sum = sum(data[i][:min_len_last] for i in range(3, 7))
    x2 = np.arange(min_len_last)
    plt.plot(x2, last_four_sum, linestyle='--', linewidth=2, label='Sum Rows 4-7')
    
    # Labels and legend
    plt.xlabel('Time Index')
    plt.ylabel('Value')
    plt.title('Row Trends and Aggregated Sums')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = "loss.txt"  # <-- replace with your file path
    data = load_data(file_path)
    plot_data(data)