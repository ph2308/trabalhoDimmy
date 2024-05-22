import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Carregar o dataset de emissão de CO2
# Vou assumir que você já tem o dataset e está no formato adequado (por exemplo, um arquivo CSV)
# Substitua 'emissao_co2.csv' pelo nome do seu arquivo de dados
data = pd.read_csv("C:\\Users\\te-mi\\Downloads\\ConsumoCo2.csv")

# Visualizar as primeiras linhas do dataset
print(data.head())
# Dividir os dados em features (X) e labels (y)
X = data.drop('CO2_Emission', axis=1)  # features
y = data['CO2_Emission']  # label

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Regressão Linear
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)

# Previsões no conjunto de teste
linear_pred = linear_reg.predict(X_test)

# Avaliação do modelo
print("Regressão Linear:")
print("R2 Score:", r2_score(y_test, linear_pred))
print("Mean Squared Error:", mean_squared_error(y_test, linear_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, linear_pred))
# KNN
knn_reg = KNeighborsRegressor(n_neighbors=5)  # número de vizinhos é 5 (pode ajustar conforme necessário)
knn_reg.fit(X_train, y_train)

# Previsões no conjunto de teste
knn_pred = knn_reg.predict(X_test)

# Avaliação do modelo
print("\nKNN:")
print("R2 Score:", r2_score(y_test, knn_pred))
print("Mean Squared Error:", mean_squared_error(y_test, knn_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, knn_pred))
