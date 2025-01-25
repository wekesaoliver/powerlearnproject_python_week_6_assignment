import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "D:\POWERLEARNPROJECT\PLP_pythonadv\powerlearnproject_python_week_6_assignment/iris.csv"  
df = pd.read_csv(file_path)

sns.set_style("whitegrid")

# 1. Bar Chart - Average petal length per species
plt.figure(figsize=(8, 5))
sns.barplot(x="species", y="petal_length", data=df, ci=None, palette="viridis")
plt.title("Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 2. Histogram - Distribution of Sepal Length
plt.figure(figsize=(8, 5))
sns.histplot(df["sepal_length"], bins=20, kde=True, color="blue")
plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# 3. Scatter Plot - Sepal Length vs Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(x="sepal_length", y="petal_length", hue="species", data=df, palette="deep")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species")
plt.show()
