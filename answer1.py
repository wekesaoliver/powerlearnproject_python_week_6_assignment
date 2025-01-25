import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
file_path = 'D:\POWERLEARNPROJECT\PLP_pythonadv\powerlearnproject_python_week_6_assignment/iris.csv'
data = pd.read_csv(file_path)

# Step 2: Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Step 3: Explore the structure of the dataset
print("\nDataset Information:")
print(data.info())

print("\nSummary Statistics:")
print(data.describe())

# Step 4: Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Step 5: Clean the dataset
# Option 1: Drop rows with missing values
data_cleaned = data.dropna()

# Option 2: Fill missing values (e.g., with mean for numerical columns)
# data_cleaned = data.fillna(data.mean())

print("\nDataset after cleaning:")
print(data_cleaned.info())

# Step 6: Basic Data Analysis
# Example: Count plot of a categorical column (replace 'Department' with an actual column)
if 'Department' in data_cleaned.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=data_cleaned, x='Department')
    plt.title('Count Plot of Department')
    plt.show()

# Example: Pairplot for numerical columns
numerical_columns = data_cleaned.select_dtypes(include=['float64', 'int64']).columns
if len(numerical_columns) > 1:
    sns.pairplot(data_cleaned[numerical_columns])
    plt.show()

# Step 7: Findings/Observations
# Example observation: Print correlations
if len(numerical_columns) > 1:
    correlation_matrix = data_cleaned[numerical_columns].corr()
    print("\nCorrelation Matrix:")
    print(correlation_matrix)
    
    # Heatmap of the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix Heatmap')
    plt.show()
