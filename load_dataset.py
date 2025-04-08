import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import sys


def elliminate_duplicates(df):
    # Eliminate duplicate rows and calculate frequency vector
    df_uniques = df.drop_duplicates()
    frequencies = df.groupby(list(df.columns)).size()
    print(len(frequencies[frequencies > 1].values))
    print(sum(frequencies[frequencies > 1].values))
    omega0 = frequencies / len(df)
    w0_series = df_uniques.apply(lambda row: omega0[tuple(row)], axis=1)
    omega0 = np.array(w0_series)
    return df_uniques, omega0


def load_data(dataset):
    if dataset == 'breast_cancer':
        file_data = "breast-cancer-wisconsin.data"  # Path to local dataset

        # Column names
        columns = ["ID", "Clump_Thickness", "Uniformity_Cell_Size", "Uniformity_Cell_Shape", 
                "Marginal_Adhesion", "Single_Epithelial_Cell_Size", "Bare_Nuclei", 
                "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class"]

        # Read dataset
        df = pd.read_csv(file_data, names=columns, header=None, na_values="?")

        # Drop rows with missing values
        df.dropna(inplace=True)

        # Convert to appropriate types
        df["Bare_Nuclei"] = df["Bare_Nuclei"].astype(int)
        df["Class"] = df["Class"].replace({2: 1, 4: 0}) # 1 = benign, 0 = malignant

        # Remove ID column (not useful for ML)
        df.drop(columns=["ID"], inplace=True)

        # There are duplicates in the dataset, so we need to eliminate them
        df, omega0 = elliminate_duplicates(df) # Eliminate duplicates and calculate omega0

        # Separate features and labels
        X = df.drop(columns=["Class"])
        Y = df["Class"].values

    elif dataset == 'communities_and_crime':
        file_data = "communities.data" # Path to local dataset
        file_names = "communities.names"

        # Read data
        df = pd.read_csv(file_data, header=None, na_values="?")

        # Read column names from the .names file
        with open(file_names) as f:
            lines = f.readlines()
        columns = [line.split()[1] for line in lines if line.startswith("@attribute")]

        # Assign columns to dataframe
        df.columns = columns

        non_predictive_columns = ['state', 'county', 'community', 'communityname', 'fold']
        df.drop(columns=non_predictive_columns, inplace=True)

        # Drop rows with missing values
        na_counts = df.isna().sum()

        # Get columns with more than 1 NaNs and drop them
        cols_to_drop = na_counts[na_counts > 1].index
        df.drop(columns=cols_to_drop, inplace=True)

        # Get columns with at most 1 NaNs and drop rows where they are NaN
        cols_to_check = na_counts[na_counts <= 1].index
        df.dropna(subset=cols_to_check, inplace=True)
        #print(f"DataFrame shape: {df.shape} (rows, columns)")
        
        # Separate features and labels
        X = df.drop(columns=["ViolentCrimesPerPop"])
        Y = 1-df["ViolentCrimesPerPop"].values # 1 = lower crime rate, 0 = higher crime rate

        omega0 = np.ones(len(Y)) / len(Y) # No duplicates in this dataset
        X_original = X.copy()


    elif dataset == 'Seoul_bike_sharing_demand':

        file_data = "SeoulBikeData.csv" # Path to local dataset

        df = pd.read_csv(file_data,encoding="latin1")
        df["Functioning Day"] = df["Functioning Day"].map({"Yes": 1, "No": 0})
        df["Holiday"] = df["Holiday"].map({"Holiday": 1, "No Holiday": 0})
        df.drop(columns=['Date'],inplace=True)
        categorical_columns = ["Seasons"]
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)  # Avoid "dummy variable trap"
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
        elliminate_duplicates(df) # Eliminate duplicates and calculate omega0
        for j, col in enumerate(categorical_columns):
            df.columns = [name + f"_categorical_{j}" if name.startswith(col + "_") else name for name in df.columns]

        X = df.drop(columns=["Rented Bike Count"])
        Y = df["Rented Bike Count"].values        
        
        omega0 = np.ones(len(Y)) / len(Y) # No duplicates in this dataset

    else:
        print('Invalid dataset')
        sys.exit()

    X_original = X.copy()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)  # Values between 0 and 1

    # Add intercept term
    X_original = pd.concat([pd.Series(1, index=X_original.index, name='Intercept'), X_original], axis=1)
    X = np.c_[np.ones(X.shape[0]), X]

    return X, Y, X_original, omega0

