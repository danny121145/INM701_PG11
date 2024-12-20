# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# ----------------------
# 1. Load the Dataset
# ----------------------
df = pd.read_csv('C:/Users/danie/INM701_PG11/data.csv')  # Replace with your actual file path

# ----------------------
# 2. Drop Unnecessary Columns
# ----------------------
if 'Unnamed: 32' in df.columns:
    df.drop(columns=['Unnamed: 32'], inplace=True)

print("First 5 rows of the dataset after dropping 'Unnamed: 32':")
print(df.head())

# ----------------------
# 3. Missing Value Heatmap
# ----------------------
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title('Missing Value Heatmap')
plt.ylabel('Row Number')
plt.xlabel('Column Name')
plt.show()

# ----------------------
# 4. Data Cleaning
# ----------------------
if 'diagnosis' in df.columns:
    df['diagnosis'] = df['diagnosis'].replace({'M': 1, 'B': 0})

print("Dataset after encoding 'diagnosis':")
print(df.head())

# ----------------------
# 5. Visualizations
# ----------------------
# Histogram for a key feature
plt.hist(df['radius_mean'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Radius Mean')
plt.ylabel('Frequency')
plt.title('Histogram of Radius Mean')
plt.show()

# Countplot for target distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='diagnosis', data=df, palette='Set2')
plt.title('Distribution of Diagnosis')
plt.xlabel('Diagnosis (0 = Benign, 1 = Malignant)')
plt.ylabel('Count')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Features')
plt.show()

# Pairplot of selected features
plt.figure(figsize=(12, 10))
sns.pairplot(df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']], hue='diagnosis', palette='husl')
plt.show()

# ----------------------
# 6. Data Splitting
# ----------------------
X = df.drop(['diagnosis'], axis=1)  # Features
y = df['diagnosis']  # Target

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print("Training Set Shape:", X_train.shape)
print("Validation Set Shape:", X_val.shape)
print("Testing Set Shape:", X_test.shape)

# ----------------------
# 7. Apply SMOTE
# ----------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Training set shape after SMOTE:", X_train_smote.shape)

# Visualize class distribution after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_train_smote, palette='coolwarm', hue=y_train_smote)
plt.title("Class Distribution After SMOTE (Training Set)")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.show()

# ----------------------
# 8. Feature Scaling
# ----------------------
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("Dataset preparation complete and ready for modeling!")