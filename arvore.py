import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error

# Assuma que o Dataset está em um arquivo CSV chamado "co2_emissions.csv"
data = pd.read_csv("co2_emissions.csv")

# Verifique as colunas e seus tipos de dados
print(data.info())

# Verifique a distribuição das variáveis e a presença de valores ausentes
data.describe()
data.isnull().sum()

# Defina as variáveis ​​previsoras
features = ["feature1", "feature2", "featureN"]

# Defina a variável-alvo
target = "co2_emission"

# Prepare os conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=42)

# Crie o modelo de Árvore de Decisão
decision_tree_model = DecisionTreeRegressor()

# Treine o modelo com os dados de treinamento
decision_tree_model.fit(X_train, y_train)

y_pred_decision_tree = decision_tree_model.predict(X_test)

mse_decision_tree = mean_squared_error(y_test, y_pred_decision_tree)
print("Erro Médio Quadrático (MSE) da Árvore de Decisão:", mse_decision_tree)

# Crie o modelo Naive Bayes
naive_bayes_model = GaussianNB()

# Treine o modelo com os dados de treinamento
naive_bayes_model.fit(X_train, y_train)

y_pred_naive_bayes = naive_bayes_model.predict(X_test)

mse_naive_bayes = mean_squared_error(y_test, y_pred_naive_bayes)
print("Erro Médio Quadrático (MSE) do Naive Bayes:", mse_naive_bayes)

if mse_decision_tree < mse_naive_bayes:
    print("A Árvore de Decisão apresentou melhor desempenho (menor MSE).")
elif mse_decision_tree > mse_naive_bayes:
    print("O Naive Bayes apresentou melhor desempenho (menor MSE).")
else:
    print("Os modelos apresentaram o mesmo desempenho (MSE).")

