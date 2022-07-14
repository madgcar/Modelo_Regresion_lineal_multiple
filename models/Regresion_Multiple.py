# Instrucciones

#Predecir el costo del seguro médico de una persona
#Este conjunto de datos tiene 7 columnas. 

#columnas Predictoras o explicativas

#edad: edad del beneficiario principal
#sexo: contratista de seguros género, femenino o masculino
#IMC: índice de masa corporal
#niños: Número de niños cubiertos por el seguro de salud / Número de dependientes
#fumador: fumar
#región: el área residencial del beneficiario en los EE. UU., noreste, sureste, suroeste, noroeste.

#Usaremos la columna 'cargos' como la variable objetivo porque queremos crear un modelo 
#que prediga el costo del seguro en función de diferentes factores.
#cargos: costos médicos individuales facturados por el seguro de salud

# your code here

# Importo las librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge #libreria de regularizacion
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import seaborn as sns
from dotenv import load_dotenv
from sqlalchemy import create_engine
import os

#load the .env file variables
load_dotenv()
connection_string = os.getenv('DATABASE_URL')
#print(connection_string)

df_raw = pd.read_csv(connection_string)

df = df_raw.copy()

# Vamos a dividir los datos para realizar el EDA

X = df.drop('charges', axis=1)
y = df['charges']

# divido el dataset
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=45)

df_train = pd.concat([X_train,y_train], axis=1)
df_train

df_mo = df_train.copy()
df_mo[df_mo.duplicated(keep=False)]

df = pd.get_dummies(df_mo,drop_first=True)

# Primero ajusto el modelo

model = LinearRegression()
model.fit(X_train, y_train)
print("intercept: ",model.intercept_)
print("variables: ",X_train.columns)
print("coeficiente: ",model.coef_)

# Creo la variable predictora para graficar

y_pred = model.predict(X_test)
print(f'R2 scoere : {r2_score(y_test,y_pred)}')
print(f'MSQ square error : {mean_squared_error(y_test,y_pred)}')
print(f'RMSE : {np.sqrt(mean_squared_error(y_test,y_pred))}')

# Creo un segundo modelo con el metodo de Ordinary Lineal square

reg_lin = sm.add_constant(X_train)
model2 = sm.OLS(y_train, reg_lin)
results = model2.fit()
results.summary()

# Otra forma de obtener el mismo modelo o de crearlo con Statsmodels seria

formula = 'charges ~ age+bmi+children+sex_male+smoker_yes+region_northwest+region_southeast+region_southwest'
res = smf.ols(formula=formula, data=df).fit()
print(res.summary())

# Grafico el modelo de regresion lineal
plt.scatter(x=y_test, y=y_pred)
plt.plot([0,50000],[0,50000], color='Green' )
plt.xlabel('Real')
plt.ylabel('Predicted')
plt.show()


