import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Load dataset
df = pd.read_csv(r'C:\Users\CHINTAPALLI SIVA\Downloads\phone_usage_india_reduced.csv')

# Show basic info about the dataset
print("Head of the dataset:")
print(df.head())  # Display the first few rows

print("\nShape of the dataset:")
print(df.shape)  # Display the shape (rows, columns)

print("\nInfo of the dataset:")
print(df.info())  # Display information about the dataset

# Select numerical columns
num_cols = df.select_dtypes(include=np.number).columns

# Objective 1: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Objective 1: Correlation Heatmap", fontsize=16)
plt.show()

# Objective 2: Histograms for Numerical Columns 
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Objective 2: Histograms for Numerical Columns", fontsize=20)

# Histogram for the first numerical column
sns.histplot(df[num_cols[0]], kde=True, ax=axes[0, 0])
axes[0, 0].set_title(f'Distribution of {num_cols[0]}')

# Histogram for the second numerical column
sns.histplot(df[num_cols[1]], kde=True, ax=axes[0, 1])
axes[0, 1].set_title(f'Distribution of {num_cols[1]}')

# Histogram for the third numerical column
sns.histplot(df[num_cols[2]], kde=True, ax=axes[1, 0])
axes[1, 0].set_title(f'Distribution of {num_cols[2]}')

# Histogram for the fourth numerical column
sns.histplot(df[num_cols[3]], kde=True, ax=axes[1, 1])
axes[1, 1].set_title(f'Distribution of {num_cols[3]}')

# Adjust layout for titles and spacing
plt.tight_layout(pad=3.0, rect=[0, 0.03, 1, 0.95])
# Adjust for the main title space
plt.show()

# Objective 3: Boxplots for Numerical Columns 
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Objective 3: Boxplots for Numerical Columns", fontsize=20)

# Boxplot for the first numerical column
sns.boxplot(x=df[num_cols[0]], ax=axes[0, 0])
axes[0, 0].set_title(f'Boxplot of {num_cols[0]}')

# Boxplot for the second numerical column
sns.boxplot(x=df[num_cols[1]], ax=axes[0, 1])
axes[0, 1].set_title(f'Boxplot of {num_cols[1]}')

# Boxplot for the third numerical column
sns.boxplot(x=df[num_cols[2]], ax=axes[1, 0])
axes[1, 0].set_title(f'Boxplot of {num_cols[2]}')

# Boxplot for the fourth numerical column
sns.boxplot(x=df[num_cols[3]], ax=axes[1, 1])
axes[1, 1].set_title(f'Boxplot of {num_cols[3]}')

# Adjust layout for titles and spacing
plt.tight_layout(pad=3.0, rect=[0, 0.03, 1, 0.95])  # Adjust for the main title space
plt.show()

# Objective 4: Violin Plots for Numerical Columns
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Objective 4: Violin Plots for Numerical Columns", fontsize=20)

# Violin plot for the first numerical column
sns.violinplot(x=df[num_cols[0]], ax=axes[0, 0])
axes[0, 0].set_title(f'Violin Plot of {num_cols[0]}')

# Violin plot for the second numerical column
sns.violinplot(x=df[num_cols[1]], ax=axes[0, 1])
axes[0, 1].set_title(f'Violin Plot of {num_cols[1]}')

# Violin plot for the third numerical column
sns.violinplot(x=df[num_cols[2]], ax=axes[1, 0])
axes[1, 0].set_title(f'Violin Plot of {num_cols[2]}')

# Violin plot for the fourth numerical column
sns.violinplot(x=df[num_cols[3]], ax=axes[1, 1])
axes[1, 1].set_title(f'Violin Plot of {num_cols[3]}')

# Adjust layout for titles and spacing
plt.tight_layout(pad=3.0, rect=[0, 0.03, 1, 0.95])  # Adjust for the main title space
plt.show()

# Objective 5: Pairplot for Numerical Columns
sns.pairplot(df.select_dtypes(include=np.number))
plt.suptitle("Objective 5: Pairplot of Numerical Columns", fontsize=16)
plt.show()
