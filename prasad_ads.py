#Import Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from lmfit import Model

def read_clean_transpose_csv(csv_file_path):
    """
    Reads data from a CSV file, cleans the data, and returns the original,
    cleaned, and transposed data.

    Parameters:
    - csv_file_path (str): Path to the CSV file.

    Returns:
    - original_data (pd.DataFrame): Original data read from the CSV.
    - cleaned_data (pd.DataFrame): Data after cleaning and imputation.
    - transposed_data (pd.DataFrame): Transposed data.
    """

    # Read the data from the CSV file
    original_data = pd.read_csv(csv_file_path)

    # Replace non-numeric values with NaN
    original_data.replace('..' , np.nan , inplace = True)

    # Select relevant columns
    columns_of_interest = [
        "Forest area (% of land area) [AG.LND.FRST.ZS]" ,
        "Adjusted net national income per capita (annual % growth) [NY.ADJ.NNTY.PC.KD.ZG]" ,
        "Agriculture, forestry, and fishing, value added (% of GDP) [NV.AGR.TOTL.ZS]" ,
        "Annual freshwater withdrawals, total (% of internal resources) [ER.H2O.FWTL.ZS]" ,
        "Arable land (% of land area) [AG.LND.ARBL.ZS]"
    ]

    # Create a SimpleImputer instance with strategy='mean'
    imputer = SimpleImputer(strategy = 'mean')

    # Apply imputer to fill missing values
    cleaned_data = original_data.copy()
    cleaned_data[columns_of_interest] = imputer.fit_transform(cleaned_data[columns_of_interest])

    # Transpose the data
    transposed_data = cleaned_data.transpose()

    return original_data , cleaned_data , transposed_data


def exponential_growth_model(x, a, b):
    """
        Exponential growth model function.

        Parameters:
        - x (array-like): Input values (time points).
        - a (float): Amplitude parameter.
        - b (float): Growth rate parameter.

        Returns:
        - array-like: Exponential growth model values.
        """
    return a * np.exp(b * np.array(x))

def curvefitPlot():
    """
        Plot the actual data, fitted curve, and confidence interval.

        Parameters:
        - time_data (array-like): Time points.
        - debt_service_data (array-like): Actual data values.
        - result (lmfit.model.ModelResult): Result of the curve fitting.

        Returns:
        None
        """

    plt.scatter(time_data , debt_service_data , label = 'Actual Data')
    plt.plot(time_data , result.best_fit , label = 'Exponential Growth Fit')
    plt.fill_between(time_data , result.eval_uncertainty() , -result.eval_uncertainty() ,
                     color = 'gray' , alpha = 0.2 ,
                     label = '95% Confidence Interval')
    plt.xlabel('Time')
    plt.ylabel('Forest area (% of land area) [AG.LND.FRST.ZS]')
    plt.title('Curve Fit for Debt Service Over Time')
    plt.legend()
    plt.show()



csv_file_path = '7937fb55-cc55-494d-ae13-2e964c893251_Data.csv'
original_data , cleaned_data , transposed_data = read_clean_transpose_csv(csv_file_path)

# Normalize the data
scaler = StandardScaler()
columns_of_interest = [
        "Forest area (% of land area) [AG.LND.FRST.ZS]" ,
        "Adjusted net national income per capita (annual % growth) [NY.ADJ.NNTY.PC.KD.ZG]" ,
        "Agriculture, forestry, and fishing, value added (% of GDP) [NV.AGR.TOTL.ZS]" ,
        "Annual freshwater withdrawals, total (% of internal resources) [ER.H2O.FWTL.ZS]" ,
        "Arable land (% of land area) [AG.LND.ARBL.ZS]"
    ]
df_normalized = scaler.fit_transform(cleaned_data[columns_of_interest])

# Apply KMeans clustering
kmeans = KMeans(n_clusters = 3 , random_state = 42)
cleaned_data['Cluster'] = kmeans.fit_predict(df_normalized)

# Extract cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

# Assuming df_normalized contains the normalized data used for clustering
silhouette_avg = silhouette_score(df_normalized , cleaned_data['Cluster'])
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Visualize the clusters and cluster centers
plt.scatter(cleaned_data["Adjusted net national income per capita (annual % growth) [NY.ADJ.NNTY.PC.KD.ZG]"] ,
            cleaned_data["Arable land (% of land area) [AG.LND.ARBL.ZS]"] ,
            c = cleaned_data['Cluster'] , cmap = 'viridis' ,  label = 'Data Points')
plt.scatter(cluster_centers[: , 1], cluster_centers[: , 4] ,
            marker = 'X' , s = 200 , c = 'red' , label = 'Cluster Centers')
plt.title('Clustering of Countries with Cluster Centers')
plt.xlabel('Adjusted Net National Income Growth (%)')
plt.ylabel('Arable Land (% of Land Area)')
plt.legend()
plt.show()

# Extract relevant data
time_data = cleaned_data['Time']
debt_service_data = cleaned_data['Arable land (% of land area) [AG.LND.ARBL.ZS]']

# Define the modified exponential growth model


# Create an lmfit Model
model = Model(exponential_growth_model)

# Set initial parameter values
params = model.make_params(a = 1 , b = 0.001)

# Fit the model to the data
result = model.fit(debt_service_data , x = time_data , params = params)
curvefitPlot()

# Generate time points for prediction
future_years = [2024 , 2025 , 2026]

# Predict values for the future years using the fitted model
predicted_values = result.eval(x = np.array(future_years))

# Display the predicted values
for year , value in zip(future_years , predicted_values):
    print(f"Predicted value for {year}: {value:.2f}")
