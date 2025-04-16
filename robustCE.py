## Necessary libraries
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.optimize
import time
import sys
from load_dataset import load_data
from scipy.stats import norm
import pandas as pd
import concurrent.futures

## Parameters
modelType = 'Poisson' # 'logistic', 'probit', 'linear' or 'Poisson'
database = {'logistic': 'breast_cancer', 'probit': 'breast_cancer', 'linear': 'communities_and_crime', 'Poisson': 'Seoul_bike_sharing_demand'}
dataset = database[modelType]

np.random.seed(42)

X, Y, X_original, omega0 = load_data(dataset)
mean_X = np.mean(X, axis=0) # Columnwise mean
std_X = np.std(X, axis=0) # Columnwise std deviation
#var_X = np.var(X, axis=0) # Columnwise variance
mean_Y = np.mean(Y) # Mean of Y
std_Y = np.std(Y) # Std deviation of Y
#var_Y = np.var(Y) # Variance of Y
max_Y = np.max(Y) # Maximum of Y
min_Y = np.min(Y) # Minimum of Y
k, n = X.shape
print(k, n)

lambda_ = 1e-3  # Regularization parameter. Taken in units of phi
kappa = 1  # Maximum probability distance.
epsilons = [1.2*i for i in range(0, 6, 1)]  # Epsilon (maximum x-distance) values to test


## Parameters for the linear restrictions on omega
alpha = 1-2*std_X/mean_X  # alpha_j = 0 for all j to deactivate the restriction
nu = 1+2*std_X/mean_X  # nu_j = n for all j to deactivate the restriction
#iota = 0.5  # alpha_j = 0 for all j to deactivate the restriction
#xi = 1.5  # nu_j = n for all j to deactivate the restriction
gamma_1 = 1-std_Y/mean_Y  # gamma_1 = 0 to deactivate the restriction
gamma_2 = 1+std_Y/mean_Y  # gamma_2 = n to deactivate the restriction
#tau_1 = 0.5  # tau_1 = 0 to deactivate the restriction
#tau_2 = 1.5  # tau_2 = n to deactivate the restriction

## Basis of the k-dimensional simplex
def create_orthonormal_basis(k):
    # Step 1: Initialize a list to store the orthonormal basis vectors
    basis = []
    # Step 2: Create the basis of vectors
    for l in range(1, k):
        vector = np.zeros(k)
        vector[:l] = 1
        vector[l] = -l
        vector[l+1:] = 0
        # Normalize the vector to make it unit length
        vector /= np.sqrt(l**2 + l)
        basis.append(vector)
    return basis
basis = create_orthonormal_basis(k)

## Generate the list of standard deviations for our Gaussian VNS
def create_sigmas():
    num_elements = 15  # Number of elements in the array
    start = np.log10(1e-9)  # Base 10 logarithm of the inferior limit
    end = np.log10(2)  # Base 10 logarithm of the superior limit
    sigmas = np.logspace(start, end, num=num_elements)[::-1]  # Reverse using slicing
    return sigmas
sigmas = create_sigmas()

## Define the gradient, hessian, and inverse link function for the selected model
if modelType == 'logistic':
    def sigmoid(t):
        t = np.clip(t, -700,700) # Avoid overflow
        return 1 / (1 + np.exp(-t))

    def g_inv(t):
        return sigmoid(t)

    def gradient_log_likelihood(beta, omega):
        return np.dot(X.T, omega*(Y - sigmoid(np.dot(X, beta)))) - lambda_*beta

    def hessian_log_likelihood(beta, omega):
        p = sigmoid(np.dot(X, beta))
        return -X.T @ np.diag(omega * p * (1 - p)) @ X - lambda_*np.eye(n)
    
elif modelType == 'linear':
    def g_inv(t):
        return t
    
    def gradient_log_likelihood(beta, omega):
        return np.dot(X.T, omega*(Y-np.dot(X, beta))) - lambda_*beta
    
    def hessian_log_likelihood(beta, omega):
        return -X.T @ np.diag(omega) @ X - lambda_*np.eye(n)
    
elif modelType == 'Poisson':
    def g_inv(t):
        return np.exp(t)
    
    def gradient_log_likelihood(beta, omega):
        t = np.dot(X, beta)
        t = np.clip(t, -700,700) # Avoid overflow
        t = np.exp(t)
        return np.dot(X.T, omega*(Y-t)) - lambda_*beta
    
    def hessian_log_likelihood(beta, omega):
        t = np.dot(X, beta)
        t = np.clip(t, -700,700) # Avoid overflow
        t = np.exp(t)
        return - X.T @ np.diag(omega * t) @ X - lambda_*np.eye(n)
    
elif modelType == 'probit':
    def g_inv(t):
        return norm.cdf(t)
    
    def gradient_log_likelihood(beta, omega):
        t = np.dot(X, beta)
        pdf = norm.pdf(t)
        cdf = norm.cdf(t)
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6) # Avoid division by zero
        return np.dot(X.T, omega * pdf/(cdf*(1-cdf))* (Y-cdf)) - lambda_*beta
    
    def hessian_log_likelihood(beta, omega):
        t = np.dot(X, beta)
        pdf = norm.pdf(t)
        cdf = norm.cdf(t)
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6) # Avoid division by zero
        return - X.T @ np.diag(omega * pdf / (1-cdf)**2 * ((-t*(1-cdf)+pdf) + Y / cdf ** 2 * (t * cdf * (1-cdf) + (1-2*cdf)*pdf))) @ X - lambda_*np.eye(n)
    
else:
    print('Invalid model type')
    sys.exit()


def KL_divergence(p,q):
    mask = p > 0  # Filter only values where p_i > 0
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

err = 0 # Global variable to store the number of times grad=0 did not converge

def beta_given_omega(omega,beta0):
    global err
    err_loc = 0
    # Solve grad = 0
    result = scipy.optimize.root(gradient_log_likelihood, beta0, args=(omega), tol=1e-12, jac=hessian_log_likelihood)
    # Check if the optimization converged
    while not result.success:
        err_loc += 1
        if err_loc > 10000:
            print('Convergence error in the grad(L) = 0 equation')
            sys.exit()
        beta0 = np.random.normal(0,np.random.uniform(0, 1),n) # Try again with random initialization
        result = scipy.optimize.root(gradient_log_likelihood, beta0, args=(omega), tol=1e-12, jac=hessian_log_likelihood)
    err += err_loc

    return result.x  # Return the optimized beta

def get_x0(l, per):
    ## Return the x0s for the largest y_pred <= q_per, percentile of the predictions
    beta = beta_given_omega(omega0, np.full(n, 0))
    Y_pred = g_inv(np.dot(X, beta))
    
    # Compute the 'per' percentile of the predictions
    q_per = np.percentile(Y_pred, per)
    
    # Filter the indices where the prediction is below or equal to the percentile
    indices_below = np.where(Y_pred <= q_per)[0]
    
    # Order the indices by the predicted value
    indices_sorted = indices_below[np.argsort(Y_pred[indices_below])][-l:]
    
    # Select the corresponding rows of X, Y, and Y_pred
    X_subset = X[indices_sorted, :]
    Y_subset = Y[indices_sorted] # Only for debugging purposes
    Y_pred_subset = Y_pred[indices_sorted] # Only for debugging purposes
    
    return X_subset, Y_subset, Y_pred_subset

def omega_restrictions(omega, kappa):
    if omega.min() < 0:
        return False
    if KL_divergence(omega,omega0) > kappa:
        return False    
    
    ## Restrictions on the mean
    new_mean = np.sum([omega[i]*X[i,:] for i in range(k)], axis=0)
    new_mean_y = np.sum([omega[i]*Y[i] for i in range(k)])
    a = np.logical_and(alpha*mean_X <= new_mean, new_mean <= nu*mean_X).all()
    b = gamma_1*mean_Y <= new_mean_y <= gamma_2*mean_Y
    return a and b
 

    ## Restrictions on the variance
    #new_variance = np.sum([omega[i] * X[i, :] ** 2 for i in range(k)], axis=0) - new_mean ** 2
    #c = np.logical_and(iota * var_X <= new_variance, new_variance <= xi * var_X).all()
    #d = tau_1*var_Y <= np.sum([omega[i]*Y[i]**2 for i in range(k)]) - new_mean_y ** 2 <= tau_2*var_Y

    #return a and b and c and d 

def inner_solution(x, kappa, initial_omega=None):
    if kappa == 0: # If kappa is 0, return the only beta possible
        beta = beta_given_omega(omega0, np.full(n,1e-6))
        return omega0, beta, np.dot(beta,x)
    else:
        global err
        if initial_omega is None:
            omega_new = omega0.copy()
        else:
            omega_new = initial_omega.copy()
        beta_new = beta_given_omega(omega_new, np.full(n,1e-6))
        beta_new_ = beta_new.copy() # For display purposes only
        value_new = np.dot(x,beta_new)
        iterations = 0
        no_improvement = 0
        no_feasibility = 0
        no_advance = 0
        t_lim=10*60 # 600 seconds = 10 minutes
        start_time = time.time()
        while no_advance < 2500:
            no_advance +=1
            elapsed_time = time.time() - start_time
            no_feasibility += 1
            no_improvement = 0
            if elapsed_time > t_lim:
                formatted_value = f"{value:.5f}" if abs(value) < 10 else f"{value:.0f}" # For display purposes only
                print(f"\rInner iter.: {iterations:05d} | Value: {formatted_value} | Iter. w/o improv./feas.: {no_improvement:02d}/{no_feasibility:03d} | Beta_dif: {np.linalg.norm(beta_new-beta_new_):.5f} | Time: {elapsed_time:.1f} | Err: {err:2d}")
                return omega_new, beta_new, value_new
            iterations += 1
            value = g_inv(value_new)
            formatted_value = f"{value:.5f}" if abs(value) < 10 else f"{value:.0f}" # For display purposes only
            sys.stdout.write(f"\rInner iter.: {iterations:05d} | Value: {formatted_value} | Iter. w/o improv./feas.: {no_improvement:02d}/{no_feasibility:03d} | Beta_dif: {np.linalg.norm(beta_new-beta_new_):.5f} | Time: {elapsed_time:.1f} | Err: {err:2d}")
            sys.stdout.flush()  # Ensure it updates immediately
            for sigma in sigmas:
                omega_old = omega_new.copy()
                addition = np.random.normal(0, sigma, size=k-1)
                for j, v in enumerate(basis):
                    omega_old += addition[j] * v
                if omega_restrictions(omega_old, kappa):
                    no_feasibility = 0
                    no_improvement +=1
                    beta_old = beta_given_omega(omega_old, beta_new)
                    value_old = np.dot(x,beta_old)
                    if value_old < value_new:
                        omega_new = omega_old.copy()
                        beta_new_ = beta_new.copy() # For display purposes only
                        beta_new = beta_old.copy()
                        value_new = value_old.copy()
                        no_improvement = 0
                        no_advance = 0
                        break
        formatted_value = f"{value:.5f}" if abs(value) < 10 else f"{value:.0f}" # For display purposes only
        print(f"\rInner iter.: {iterations:05d} | Value: {formatted_value} | Iter. w/o improv./feas.: {no_improvement:02d}/{no_feasibility:03d} | Beta_dif: {np.linalg.norm(beta_new-beta_new_):.5f} | Time: {elapsed_time:.1f} | Err: {err:2d}")
        return  omega_new, beta_new, value_new

def outer_solution(per, x0, xinit, epsilon, kappa, num_iterations):
    model = gp.Model("Outer")

    beta0 = beta_given_omega(omega0, np.full(n,1e-6))
    print('Initial solution: ', x0) # To show the columns names as well, np.array2string(np.array(list(zip(X_original.columns, x0))), separator=", "))
    print('Initial t value: ', np.dot(x0,beta0))
    print('Initial value: ', g_inv(np.dot(x0,beta0)))
    xsol = xinit.copy()
    start_time = time.time()
    a = inner_solution(xsol, kappa)
    end_time = time.time()
    # Calculate and print the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    beta = a[1]
    x_max = xsol.copy()
    t_max = np.dot(xsol,beta)
    print('Initial worst t value: ', np.dot(xsol,beta))
    print('Initial worst value: ', g_inv(np.dot(xsol,beta)))
    x = [] # List of all (normalized) variables
    y = [] # List of (not normalized) integer variables
    categorical_groups = {}
    for i, col in enumerate(X_original.columns):        
        if "_categorical_" in col: # If it is a categorical variable (this string is added in load_dataset.py)
            j = int(col.split("_categorical_")[-1])
            if j not in categorical_groups:
                categorical_groups[j] = []
            categorical_groups[j].append(i)

        unique_values = np.unique(X_original[col])
        
        if np.all(np.isin(unique_values, [0, 1])):  # If it is binary
            x.append(model.addVar(vtype=GRB.BINARY, name=f"x_{i}"))
        
        elif np.issubdtype(X_original[col].dtype, np.integer):  # If it is integer
            yi = model.addVar(vtype=GRB.INTEGER, lb=X_original[col].min(), ub=X_original[col].max(), name=f"y_{i}")
            y.append(yi)
            xi = model.addVar(lb=-GRB.INFINITY, name=f"x_{i}")
            x.append(xi)
            model.addConstr(xi==(yi-X_original[col].min())/(X_original[col].max()-X_original[col].min()))
        
        else:  # If it is continuous
            x.append(model.addVar(lb=0, ub=1, name=f"x_{i}")) # Normalized to [0,1]

    for j, indices in enumerate([categorical_groups[j] for j in sorted(categorical_groups.keys())]):
        # Restrictions for the dummy variables associated with the same categorical variable
        max_dummy_sum = X_original.iloc[:, indices].sum(axis=1).max()  
        min_dummy_sum = X_original.iloc[:, indices].sum(axis=1).min()
        model.addConstr(gp.quicksum(x[i] for i in indices) <= max_dummy_sum, f"categorical_{j}_max")
        model.addConstr(gp.quicksum(x[i] for i in indices) >= min_dummy_sum, f"categorical_{j}_min")

    x = np.array(x)
    model.addConstr(x[0] == 1) # Intercept is fixed to 1

    # Activation variables for the l0 norm
    z = model.addVars(n, vtype=GRB.BINARY, name="z")

    # Absolute value variables for the l1 norm
    abs_diff = model.addVars(n, lb=0, name="abs_diff")

    # Define absolute value constraints
    for i in range(n):
        model.addConstr(abs_diff[i] >= x[i] - x0[i])
        model.addConstr(abs_diff[i] >= -(x[i] - x0[i]))

    # Activation constraints
    big_M = np.ones(n) # The dataset is normalised, so the maximum difference is 1
    model.addConstrs((abs_diff[i]<=z[i]*big_M[i] for i in range(n)), name="active")

    # Distance constraints
    model.addConstr(gp.quicksum(abs_diff[i] for i in range(n)) + gp.quicksum(z[i] for i in range(n)) <= epsilon, "x-distance")

    t = model.addVar(lb=-GRB.INFINITY, name="t")
    model.setObjective(t, GRB.MAXIMIZE)
    for j in range(1,num_iterations+1):
        start_time = time.time()
        print('\nOuter iteration: ', j)
        model.addConstr(t <= gp.quicksum(beta[i] * x[i] for i in range(n)), f"restriccion_{j}")
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            print('WARNING: GUROBI did not converge')
        xsol = [a.X for a in x]
        a = inner_solution(xsol, kappa, initial_omega=a[0])
        beta = a[1]
        if np.dot(xsol,beta) > t_max:
            x_max = xsol.copy()
            t_max = np.dot(xsol,beta)
        lower_bound = g_inv(t_max)
        upper_bound = g_inv(t.X)
        print('Status:', model.status)
        print('Objective value t: ', t.X)
        print('t_max: ', t_max)
        print('Solution: ', xsol) 
        beta0 = beta_given_omega(omega0, np.zeros(n))
        print('Worst value: ', g_inv(np.dot(xsol,beta0)))
        print('Worst value range: ', lower_bound, upper_bound)
        #print('Worst value range 2: ', g_inv(np.dot(xsol,beta)), upper_bound)
        gap = (upper_bound - lower_bound)/(max_Y - min_Y)
        print('Gap: ', gap)
        print('l1-distance: ', np.sum(np.abs(xsol-x0)))
        print('l2-distance: ', np.linalg.norm(xsol-x0)**2)
        print('linf-distance: ', np.max(np.abs(xsol-x0)))
        print('l0-distance: ', np.sum(np.abs(xsol - x0) > 1e-4))
        print('l1+l0: ', np.sum(np.abs(xsol-x0)) + np.sum(np.abs(xsol - x0) > 1e-4))
        print('omega-distance: ', KL_divergence(a[0],omega0))
        if gap < 1e-4:
            print('Convergence reached')
            print('Best solution so far:', x_max) # To show column names as well: np.array2string(np.array(list(zip(X_original.columns, x_max))), separator=", "))
            print('Best value interval so far:', lower_bound, upper_bound)
            print('\n\n\n\n')
            return x0, epsilon, x_max, lower_bound, upper_bound, a[0], a[1], per
        end_time = time.time()
        # Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
    print('Maximum number of iterations ', num_iterations, ' reached.')
    print('Best solution so far: ', x_max)
    print('Best value interval so far:', lower_bound, upper_bound)
    print('\n\n\n\n')
    return x0, epsilon, x_max, lower_bound, upper_bound, a[0], a[1], per

### Experiments:

def run_experiment(per, x0, xinit, epsilon):
    print(f'Experiment with x0 = {x0}, epsilon = {epsilon}')
    return outer_solution(per, x0, xinit, epsilon, kappa, 10)  # kappa is fixed to 1

def run_experiment_for_x0(x0_):
    results = []
    per, x0 = x0_  # Unpack the tuple
    xinit = x0
    for epsilon in epsilons:
        results+=[run_experiment(per, x0, xinit, epsilon)]
        xinit = results[-1][2]  # Update xinit with the last solution
    return results  # Return all results for a given x0

def experiment(x0s):
    # Collect results for all x0s
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(run_experiment_for_x0, x0s))  # Parallelize over x0
    
    # Flatten the results into individual components
    flattened_results = []
    for result in results:
        for res in result:
            # Prepare the result for the current experiment as a dictionary
            result_dict = {}
            # Extract epsilon, lower_bound, upper_bound, percentile
            result_dict['percentile'] = res[7]
            result_dict['epsilon'] = res[1]
            result_dict['lower_bound'] = res[3]
            result_dict['upper_bound'] = res[4]
            # Extract x0 components dynamically
            for i, val in enumerate(res[0]):
                result_dict[f'x0_{i}'] = val
            
            # Extract x_max components dynamically
            for i, val in enumerate(res[2]):
                result_dict[f'x_max_{i}'] = val
            
            # Extract omega components dynamically
            for i, val in enumerate(res[5]):
                result_dict[f'omega_{i}'] = val
            
            # Extract beta components dynamically
            for i, val in enumerate(res[6]):
                result_dict[f'beta_{i}'] = val            
            
            # Add the dictionary for this result to the list of all results
            flattened_results.append(result_dict)
    
    # Convert flattened results into DataFrame
    df = pd.DataFrame(flattened_results)
    return df

def save_results(df, filename):
    df.to_csv(filename, index=False)
    print(f'Results saved to {filename}')


if __name__ == "__main__":
    percentiles = range(10,51,10)  # Percentiles to consider
    per_dict = {per: get_x0(1, per)[0][0] for per in percentiles}  # Get the x0s for the selected percentiles
    x0s = [(per,per_dict[per]) for per in percentiles]
    all_results = experiment(x0s)
    filename = modelType + f"_kappa_{kappa}_fix.csv"
    save_results(all_results, filename)
