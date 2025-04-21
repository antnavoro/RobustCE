import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from load_dataset import load_data
import pandas as pd
import matplotlib.cm as cm


modelType = 'logistic' # 'logistic', 'probit', 'linear' or 'Poisson'
database = {'logistic': 'breast_cancer', 'probit': 'breast_cancer', 'linear': 'communities_and_crime', 'Poisson': 'Seoul_bike_sharing_demand'}
dataset = database[modelType]
file_data0 = f'{modelType}_kappa_0.csv'
file_data1 = f'{modelType}_kappa_1.csv'
# Load the CSV file
percentiles = [10, 20, 30, 40, 50]

def plot_heatmap(a, B, row_labels, col_labels, per, i):
    """
    Plots a heatmap of B - a by columns, with row and column labels.

    Parameters:
    - a: 1D reference vector.
    - B: 2D matrix where each row is a vector b_i.
    - row_labels: List of labels for the rows (each b_i).
    - col_labels: List of labels for the columns.
    """
    # Convert a and B to NumPy arrays
    a = np.array(a)
    B = np.array(B)

    # Check that dimensions match
    if B.shape[1] != len(a):
        raise ValueError("Dimensions of B and a do not match in number of columns.")
    
    # Check that the lengths of the labels are correct
    if len(row_labels) != B.shape[0]:
        raise ValueError("Number of row labels does not match number of rows in B.")
    if len(col_labels) != B.shape[1]:
        raise ValueError("Number of column labels does not match number of columns in B.")

    # Compute B - a (subtract row-wise)
    diff_matrix = B - a
    if all(isinstance(label, (int, float)) for label in row_labels):
        rounded_row_labels = [round(label, 1) for label in row_labels]

    # Create the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        diff_matrix,
        vmin=-1, vmax=1,
        cmap="coolwarm",
        annot=False,
        fmt=".2f",
        linewidths=0.5,
        xticklabels=col_labels,
        yticklabels=rounded_row_labels
    )
    plt.xlabel("Features")
    plt.ylabel("Maximum x-distance")

    # Rotate column labels
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'paper/heatmap_{modelType}_{per}_{i}.png', dpi=150)

    # plt.show()


def calculate_distances(x_max, x_0):
    # Convert to numpy arrays for easier vectorized operations
    x_max = np.array(x_max)
    x_0 = np.array(x_0)
    # Calculate L1 distance (sum of absolute differences)
    L1_distance = np.sum(np.abs(x_max - x_0))
    # Calculate L0 distance (number of non-zero differences)
    L0_distance = np.sum(np.abs(x_max - x_0)>1e-4)
    return L1_distance + L0_distance  # Return the sum of both distances


def load_results_from_csv(filename):
    # Load the data from the CSV file
    df = pd.read_csv(filename)
    
    # Reconstruct the vectors (x0, x_max, a0, a1) from the column names
    results_dict = {}
    
    for _, row in df.iterrows():
        # Reconstruct the vectors and other values
        percentile = row['percentile']
        x0 = [row[f'x0_{i}'] for i in range(len([col for col in df.columns if col.startswith('x0_')]))]
        x_max = [row[f'x_max_{i}'] for i in range(len([col for col in df.columns if col.startswith('x_max_')]))]
        omega = [row[f'omega_{i}'] for i in range(len([col for col in df.columns if col.startswith('omega')]))]
        beta = [row[f'beta_{i}'] for i in range(len([col for col in df.columns if col.startswith('beta')]))]
        epsilon = row['epsilon']
        lower_bound = row['lower_bound']
        upper_bound = row['upper_bound']
        
        # Create the result tuple
        result_tuple = (percentile, x0, epsilon, x_max, lower_bound, upper_bound, omega, beta)
        
        if str(int(percentile)) not in results_dict:
            results_dict[str(int(percentile))] = []

        results_dict[str(int(percentile))].append(result_tuple)
    
    return results_dict

def get_results(data, percentile):
    percentile = str(percentile)
    epsilons = np.array([result[2] for result in data[percentile]])
    lower_bounds = np.array([result[4] for result in data[percentile]])
    upper_bounds = np.array([result[5] for result in data[percentile]])
    d_x = [calculate_distances(result[1],result[3]) for result in data[percentile]]
    return epsilons, lower_bounds, upper_bounds, d_x


data0 = load_results_from_csv(file_data0)
data1 = load_results_from_csv(file_data1)


#Filter and plot the data for each x0
#plt.figure(figsize=(10, 6))  # Set the size of the figure
colors = cm.rainbow(np.linspace(0, 1, len(percentiles)))
i=0
for percentile in percentiles:  # Loop through unique x0 values
    color = colors[i]
    i+=1
    epsilons0, lower_bounds0, upper_bounds0, d_x0 = get_results(data0, percentile)
    epsilons1, lower_bounds1, upper_bounds1, d_x1 = get_results(data1, percentile)
    plt.plot(epsilons0, lower_bounds0, '-o', label=f'per = {percentile}',color=color)
    plt.plot(epsilons1, lower_bounds1, linestyle='--', marker='o', markerfacecolor='none', color=color)
    plt.fill_between(epsilons0, lower_bounds0, lower_bounds1, where=(lower_bounds0 > lower_bounds1), interpolate=True, color=color, alpha=0.3)

# Customize the plot
plt.xlabel('Maximum x-distance')
plt.ylabel('Objective function value')
#plt.xlim([0,6])
#plt.ylim([0,1.2])
plt.legend()  # Show the legend
plt.grid(True)  # Optional: Add grid for better readability
plt.savefig(f'paper/pareto_{modelType}.png', dpi=150)
# Show the plot
#plt.show()



columns = load_data(dataset)[-2].columns
for percentile in percentiles:
    percentile = str(percentile)
    x_max = [result[3][1:] for result in data0[percentile]]
    plot_heatmap(data0[percentile][0][1][1:], x_max, epsilons0, columns, percentile, 0)

    x_max = [result[3][1:] for result in data1[percentile]]
    plot_heatmap(data1[percentile][0][1][1:], x_max, epsilons0, columns, percentile, 1)

