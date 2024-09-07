import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Load and clean the data
def load_data(file_path):
    #Load the dataset and clean column names by stripping extra spaces.
    data = pd.read_csv(file_path)
    #data = data.sample(n=1000, replace=True) #for small DataSet
    data.columns = data.columns.str.strip()
    return data

#Display basic info about the dataset
def display_basic_info(data):
    rows, cols = data.shape
    print(f'Number of Rows: {rows}')
    print(f'Number of Cols: {cols}')
    print(f'Total cells: {rows * cols}')
    print(data.head())  #Quick glance at the data
    print(data.info())  #Overview of data types and missing values
    print(f'Overview of Columns:\n')
    print(data.describe().transpose())  #Summary stats for numerical columns)

# Check for duplicate/missing/infinite values and clean the data
def data_cleaning(data):
    duplicate_vals  = data[data.duplicated()]
    print(f'Number of duplicates: {len(duplicate_vals)}')

    missing_values = data.isnull().sum()
    inf_counts = {col: np.sum(data[col].isin([np.inf, -np.inf])) for col in data.columns}

    print("Columns with missing values:")
    print(missing_values[missing_values > 0])

    print("\nColumns with infinite values:")
    for col, count in inf_counts.items():
        if count > 0:
            print(f"{col} has {count} infinite values.")

    # Replace infinite values with NaN for numeric columns
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Replace NaN with the mean only for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    return data


#visualize the distribution of labels - check balance
def plot_label_distribution(data):
    print(data['Label'].value_counts())
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Label', data=data)
    plt.title('Distribution of Label Classes')
    plt.xticks(rotation=45)
    plt.show()

#Check how many unique values each column has (could help identify categorical variables)
def check_unique_values(data):
    print("\nUnique values in each column:")
    print(data.nunique())

#Detect outliers using the IQR method and return percentage of outliers
def detect_outliers(data):
    outliers_percentage = {}
    for col in data.select_dtypes(include=[np.number]).columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        num_outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()

        # Calculate percentage of outliers
        outliers_percentage[col] = 100 * num_outliers / len(data[col])

    # Plot outliers percentage as a bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(outliers_percentage.keys()), y=list(outliers_percentage.values()))
    plt.xticks(rotation=90)
    plt.title("Percentage of Outliers in Each Feature")
    plt.xlabel("Features")
    plt.ylabel("Percentage of Outliers")
    plt.tight_layout()
    plt.show()

    return outliers_percentage

#Handle categorical columns
def handle_categorical_columns(data):
    # Get categorical columns (usually 'object' types)
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Label encode 'Label' if it's not numerical already
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])

    # For 'Destination Port' (a high-cardinality column), use Target Encoding
    if 'Destination Port' in categorical_columns:
        encoder = TargetEncoder(cols=['Destination Port'])
        data = encoder.fit_transform(data, data['Label'])

    return data

#Perform PCA
def perform_pca(data):
    # Drop the target column and standardize the features
    features = data.drop(columns=['Label'])
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Perform PCA and retain enough components to explain 95% of the variance
    pca = PCA(n_components=0.95)
    pca_features = pca.fit_transform(scaled_features)

    # Display how many components were kept
    print(f"Number of components retained: {pca.n_components_}")

    # Plot cumulative explained variance
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by Number of Principal Components')
    plt.show()

    # Create DataFrame for PCA results
    pca_data = pd.DataFrame(pca_features, columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)])
    pca_data['Label'] = data['Label'].values  # Add target variable back

    return pca_data, pca

#correlations
def generate_correlation_heatmap(data):
    # Create a correlation matrix including 'Label'
    corr_matrix = data.corr()

    # Print the correlation of all features with 'Label'
    print("\nCorrelation of features with Label:")
    label_correlation = corr_matrix['Label'].sort_values(ascending=False)
    print(label_correlation)

    # Identify high correlation features
    high_corr_features = label_correlation[label_correlation.abs() > 0.4]
    print("\nFeatures with high correlation with Label (absolute correlation > 0.5):")
    print(high_corr_features)

    # Plot the correlation heatmap
    plt.figure(figsize=(20, 16))  # Increase figure size
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5)
    plt.title('Correlation Matrix including Label')
    plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
    plt.yticks(rotation=0)  # Rotate y-axis labels for better readability
    plt.show()

if __name__ == '__main__':
    #Load and clean data
    file_path = 'dataset.csv'
    data = load_data(file_path)

    #Display basic info
    display_basic_info(data)

    #Check for missing and infinite values
    data = data_cleaning(data)

    #Visualize label distribution
    plot_label_distribution(data)

    #Check unique values in each column
    check_unique_values(data)

    #Detect outliers and show percentages
    outliers_percentage = detect_outliers(data)

    #Handle categorical columns
    data = handle_categorical_columns(data)

    #Save the cleaned data before PCA
    data.to_csv("cleaned_data_before_pca.csv", index=False)

    #Perform PCA
    pca_data, pca = perform_pca(data)

    #Save the PCA-transformed data
    pca_data.to_csv("cleaned_data_with_pca.csv", index=False)

    #Generate correlation heatmap
    generate_correlation_heatmap(data)
