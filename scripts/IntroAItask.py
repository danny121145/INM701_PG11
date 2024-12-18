import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# ----------------------
# 1. Load Dataset
# ----------------------
df = pd.read_csv('C:/Users/danie/INM701_PG11/diabetes.csv')
print("Dataset Overview:")
print(df.head())  # Display first rows
print("\nMissing Values:\n", df.isnull().sum())  # Check for missing values
print("\nSummary Statistics:")
print(df.describe())  # Display summary statistics

# ----------------------
# 2. Initial Visualizations (Before Cleaning)
# ----------------------
plt.figure(figsize=(12, 10))
df.hist(figsize=(12, 10), bins=15, color='skyblue', edgecolor='black')
plt.suptitle("Feature Distributions (Before Cleaning)", fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for i, column in enumerate(df.columns[:-1], 1):  # Skip target column 'Outcome'
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[column], color='orange')
    plt.title(f'Boxplot of {column} (Before Cleaning)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (Before Cleaning)")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Outcome', data=df, palette='Set2', hue='Outcome')
plt.title("Class Distribution (Before Cleaning)")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()

# ----------------------
# 3. Data Cleaning
# ----------------------
columns_with_zeros = ['BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in columns_with_zeros:
    median = df[df[col] != 0][col].median()
    df[col] = df[col].replace(0, median)

# ----------------------
# 4. Visualizations After Cleaning
# ----------------------
plt.figure(figsize=(12, 10))
df.hist(figsize=(12, 10), bins=15, color='lightgreen', edgecolor='black')
plt.suptitle("Feature Distributions (After Cleaning)", fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.boxplot(y=df[column], color='purple')
    plt.title(f'Boxplot of {column} (After Cleaning)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap (After Cleaning)")
plt.show()

# ----------------------
# 5. Data Splitting (60-20-20)
# ----------------------
X = df.drop('Outcome', axis=1)  # Features
y = df['Outcome']  # Target

# First split: Training (60%) and Temp (40% for Validation + Testing)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Second split: Validation (50% of Temp) and Testing (50% of Temp)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training Set: {X_train.shape}")
print(f"Validation Set: {X_val.shape}")
print(f"Testing Set: {X_test.shape}")

# ----------------------
# 6. Apply SMOTE (Training Set Only)
# ----------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Visualize class distribution after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_smote, palette='coolwarm', hue=y_train_smote)
plt.title("Class Distribution After SMOTE (Training Set)")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.show()

# ----------------------
# 7. Feature Scaling
# ----------------------
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("Data preparation complete!")


