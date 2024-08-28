import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import numpy as np

url = 'https://raw.githubusercontent.com/ecusidatec/D4G_Interview/main/Datasets/PruebaTecnicaPasantia/dataset_alquiler.csv'
df = pd.read_csv(url)

# Feature Engineering
df['fecha'] = pd.to_datetime(df['fecha'])
df['mes'] = df['fecha'].dt.month
df['año'] = df['fecha'].dt.year
df.drop(['fecha'], axis=1, inplace=True)

# Xs y Y
X = df.drop(['total_alquileres','indice', 'u_registrados', 'u_casuales'], axis=1)
y = df['total_alquileres']



#print(y.isnull().sum()) 
#print(X.isnull().sum()) 

X = X[~y.isna()]
y = y.dropna()

#dividir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# pre procesar
numeric_features = ['temperatura', 'humedad', 'velocidad_viento', 'sensacion_termica']
categorical_features = ['temporada', 'anio', 'mes', 'hora', 'dia_semana', 'clima', 'dia_trabajo', 'feriado']


numeric_transformer = StandardScaler()

categorical_transformer =  OneHotEncoder(sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


X_train_transformed = preprocessor.fit_transform(X_train)

class LinearRegressionManual:
    def __init__(self, learning_rate=0.01, epochs=1000):

        # iteraciones y rapidez de descenso
        self.learning_rate = learning_rate
        self.epochs = epochs
    #ajustarse a los datos
    def fit(self, X, y):
        self.m, self.n = X.shape
        self.theta = np.zeros(self.n)
        self.bias = 0
        self.X = X
        self.y = y
        
        for _ in range(self.epochs):
            self.update_weights()
    #cambiar pesos con descenso de gradient
    def update_weights(self):
        y_prediction = self.predict(self.X)
        d_theta = - (2 * (self.X.T).dot(self.y - y_prediction)) / self.m
        d_bias = - 2 * np.sum(self.y - y_prediction) / self.m
        
        self.theta -= self.learning_rate * d_theta
        self.bias -= self.learning_rate * d_bias
    
    def predict(self, X):
        return X.dot(self.theta) + self.bias
    #aplicar formula de r^2
    def score(self, X, y):
        y_pred = self.predict(X)
        total_variance = np.sum((y - np.mean(y)) ** 2)
        residual_variance = np.sum((y - y_pred) ** 2)
        r2_score = 1 - (residual_variance / total_variance)
        return r2_score
    
class DecisionTreeRegressorManual:
    #profundiad max
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
#ajustar
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        # parar
        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth):
            return np.mean(y)

        # escoger mejor divisiion
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            return np.mean(y)

        # dividir
        left_indices = X[:, best_feature] < best_threshold
        right_indices = X[:, best_feature] >= best_threshold

        # ramas
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return (best_feature, best_threshold, left_subtree, right_subtree)
#comparar para mejor division
    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_mse = float('inf')

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = X[:, feature] < threshold
                right_indices = X[:, feature] >= threshold
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                mse = self._calculate_mse(y[left_indices], y[right_indices])
                if mse < best_mse:
                    best_mse = mse
                    best_feature, best_threshold = feature, threshold

        return best_feature, best_threshold

    def _calculate_mse(self, left_y, right_y):
        left_mse = np.var(left_y) * len(left_y)
        right_mse = np.var(right_y) * len(right_y)
        return (left_mse + right_mse) / (len(left_y) + len(right_y))

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, tree):
        if not isinstance(tree, tuple):
            return tree

        feature, threshold, left_subtree, right_subtree = tree
        if x[feature] < threshold:
            return self._traverse_tree(x, left_subtree)
        else:
            return self._traverse_tree(x, right_subtree)

    def score(self, X, y):
        y_pred = self.predict(X)
        total_variance = np.sum((y - np.mean(y)) ** 2)
        residual_variance = np.sum((y - y_pred) ** 2)
        r2_score = 1 - (residual_variance / total_variance)
        return r2_score

# pipeline regresion lineal
# con scikit: ('model', LinearRegression())])
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', LinearRegressionManual())])
# train
pipeline_lr.fit(X_train, y_train)
# eval
y_pred_lr = pipeline_lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)

score_lr = pipeline_lr.score(X_test, y_test)

print('MSE para Regresión Lineal:', mse_lr)
print('R^2 para Regresión Lineal:', score_lr)


# pipeline random forest
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', RandomForestRegressor(n_estimators=50, random_state=42))])
# train
pipeline_rf.fit(X_train, y_train)
# eval
y_pred_rf = pipeline_rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
score_rf = pipeline_rf.score(X_test, y_test)

print('MSE para Random Forest:', mse_rf)
print('R^2 para Random Forest:', score_rf)

#pipeline arbol de decision
# con scikit: ('model', DecisionTreeRegressorManual(random_state=42))])
pipeline_dt = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', DecisionTreeRegressorManual(max_depth=12))])
#train
pipeline_dt.fit(X_train, y_train)

#eval
y_pred_dt = pipeline_dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
score_dt = pipeline_dt.score(X_test, y_test)

print('MSE para Decision Tree:', mse_dt)
print('R^2 para Decision Tree:', score_dt)

#Tristemente el algoritmo de aprendizaje mas efectivo en esta prueba fue el algoritmo Random Forest, quefue implementado directamente de la librería SciKit Learn