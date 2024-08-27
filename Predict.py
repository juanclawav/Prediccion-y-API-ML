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

url = 'https://raw.githubusercontent.com/ecusidatec/D4G_Interview/main/Datasets/PruebaTecnicaPasantia/dataset_alquiler.csv'
df = pd.read_csv(url)

# Feature Engineering
df['fecha'] = pd.to_datetime(df['fecha'])
df['hora'] = df['fecha'].dt.hour
df['mes'] = df['fecha'].dt.month
df['año'] = df['fecha'].dt.year
df.drop(['fecha'], axis=1, inplace=True)

# Xs y Y
X = df.drop(['total_alquileres','indice'], axis=1)
y = df['total_alquileres']

#print(y.isnull().sum()) 
#print(X.isnull().sum()) 

X = X[~y.isna()]
y = y.dropna()

#dividir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# pre procesar
numeric_features = ['temperatura', 'humedad', 'velocidad_viento', 'sensacion_termica', 'u_registrados', 'u_casuales']
categorical_features = ['temporada', 'anio', 'mes', 'hora', 'dia_semana', 'clima', 'dia_trabajo', 'feriado']


numeric_transformer = StandardScaler()

categorical_transformer =  OneHotEncoder()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# X_train_transformed = preprocessor.fit_transform(X_train)
#print(pd.DataFrame(X_train_transformed, columns= preprocessor.get_feature_names_out()))


# pipeline regresion lineal
pipeline_lr = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', LinearRegression())])
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
pipeline_dt = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', DecisionTreeRegressor(random_state=42))])
#train
pipeline_dt.fit(X_train, y_train)

#eval
y_pred_dt = pipeline_dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
score_dt = pipeline_dt.score(X_test, y_test)

print('MSE para Decision Tree:', mse_dt)
print('R^2 para Decision Tree:', score_dt)