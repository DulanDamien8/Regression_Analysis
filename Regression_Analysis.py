import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso

data = pd.read_csv('winequality.csv', delimiter=';')

x = data.drop('quality', axis=1)
y = data['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Create and train the OLS model
ols_model = LinearRegression()
ols_model.fit(x_train, y_train)

# Make predictions
y_pred_ols = ols_model.predict(x_test)

# Evaluate the model
mse_ols = mean_squared_error(y_test, y_pred_ols)
r2_ols = r2_score(y_test, y_pred_ols)

print(f"OLS Model - Mean Squared Error: {mse_ols:.2f}, R^2 Score: {r2_ols:.2f}")


lasso_model = Lasso(alpha=0.1)  # You can experiment with different alpha values
lasso_model.fit(x_train, y_train)

# Make predictions
y_pred_lasso = lasso_model.predict(x_test)

# Evaluate the model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"LASSO Model - Mean Squared Error: {mse_lasso:.2f}, R^2 Score: {r2_lasso:.2f}")

# OLS Model
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_ols, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('OLS Model Predictions')
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')

# LASSO Model
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_lasso, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('LASSO Model Predictions')
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')

plt.tight_layout()
plt.show()

cm = data.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

for column in data.columns[:-1]:
    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=data, x=column, y='quality')
    plt.title(f'Scatter Plot of {column} vs Quality')
    plt.xlabel(column)
    plt.ylabel('Quality')
    plt.show()

correlations = data.drop('quality', axis=1).corrwith(data['quality']).abs()
correlations_sorted = correlations.sort_values(ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(x=correlations_sorted.index, y=correlations_sorted.values,hue=correlations_sorted.index, palette='viridis', legend=False)
plt.xticks(rotation=45)
plt.title('Correlation with Quality')
plt.xlabel('Features')
plt.ylabel('Absolute Correlation')
plt.show()

pred_data = pd.read_csv('pred_wine.csv', delimiter=';')

print("Null values in each column:\n", data.isnull().sum())
print("\nCorrelation Matrix:\n", cm)
print("Correlation with 'quality':\n", correlations_sorted)
print("Predicted quality values saved to predicted_quality.csv")

