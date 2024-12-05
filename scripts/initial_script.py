import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:/Users/danie/INM701_PG11/Depression Student Dataset.csv')
print("Dataset Overview:")
print(dataset.head())

numerical_features = ['Age', 'Academic Pressure', 'Study Satisfaction', 'Study Hours', 'Financial Stress']
dataset[numerical_features].hist(bins=15, figsize=(12, 10), layout=(3, 2), color='pink', edgecolor='black')
plt.suptitle('Distribution of Numerical Features', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

categorical_features = ['Gender', 'Sleep Duration', 'Dietary Habits', 
                        'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness', 'Depression']

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for feature, ax in zip(categorical_features, axes.flatten()):
    dataset[feature].value_counts().plot(kind='bar', ax=ax, color='orange', edgecolor='black')
    ax.set_title(f'{feature} Distribution')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
plt.show()

correlation_matrix = dataset[numerical_features].corr()

plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(numerical_features)), numerical_features, rotation=90)
plt.yticks(range(len(numerical_features)), numerical_features)
plt.title('Correlation Matrix of Numerical Features', fontsize=14)
plt.show()

plt.figure(figsize=(8, 6))
colors = dataset['Depression'].map({'Yes': 'red', 'No': 'blue'})
plt.scatter(dataset['Study Hours'], dataset['Academic Pressure'], c=colors, alpha=0.6, edgecolor='k')
plt.xlabel('Study Hours')
plt.ylabel('Academic Pressure')
plt.title('Study Hours vs. Academic Pressure (Colored by Depression)')
plt.show()