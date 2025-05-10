# APIREST_IrisDataset_Entorno_Virtual
En este proyecto crearemos una API que usa modelos de Machine Learning entrenados con un conjunto de datos llamado Iris Dataset, el cual contiene características de distintas especies de flores iris.

### ¿Qué función tendrá esta API?

La API permitirá predecir la especie de una flor a partir de cuatro características:

* Largo del sépalo (sepal_length)
* Ancho del sépalo (sepal_width)
* Largo del pétalo (petal_length)
* Ancho del pétalo (petal_width)

### Modelos de Machine Learning

Se entrenan cuatro modelos diferentes, y cada uno está guardado en un archivo `h5` o `.pkl` para ser reutilizado por la API:

* Regresión Logística
* Árbol de Decisión
* Máquina de Vectores de Soporte (SVM)
* Bosque Aleatorio (Random Forest)

### Estructura de archivos

/machine__learning_api/

│

├── models

    ├── model_logistic.h5
    ├── model_forest.h5
    ├── model_svm.h5
    ├── model_tree.h5

├── app.py


├── iris_models.py

└── requirements.txt

_____________________________________________________________________________________________________________________


# Proceso de creación de API

* Crear una carpeta llamada `machine__learning_api`

Dentro de la carpeta machine__learning_api 

* Crear una carpeta llamada `models`

* Crear un archivo llamado `requirements.txt` y que contenga lo siguiente:

<pre>flask
scikit-learn
joblib</pre>

* Crear un archivo llamado iris_models.py y que contenga lo siguiente:

<pre>from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

# Cargar dataset
iris = load_iris()
X = iris.data
y = iris.target

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Modelo Logistic Regression
logistic_model = LogisticRegression(max_iter=200)
logistic_model.fit(X_train, y_train)
accuracy_logistic = logistic_model.score(X_test, y_test)
print(f"Accuracy Logistic Regression: {accuracy_logistic:.2f}")
joblib.dump(logistic_model, './models/model_logistic.h5')

# 2. Modelo SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
accuracy_svm = svm_model.score(X_test, y_test)
print(f"Accuracy SVM: {accuracy_svm:.2f}")
joblib.dump(svm_model, './models/model_svm.h5')

# 3. Modelo Decision Tree
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)
accuracy_tree = tree_model.score(X_test, y_test)
print(f"Accuracy Decision Tree: {accuracy_tree:.2f}")
joblib.dump(tree_model, './models/model_tree.h5')

# 4. Modelo Random Forest
forest_model = RandomForestClassifier()
forest_model.fit(X_train, y_train)
accuracy_forest = forest_model.score(X_test, y_test)
print(f"Accuracy Random Forest: {accuracy_forest:.2f}")
joblib.dump(forest_model, './models/model_forest.h5') </pre>
  
* Crear un archivo llamado `app.py` y que contenga lo siguiente:

<pre>from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Cargar modelos
modelos = {
    "logistic": joblib.load("./models/model_logistic.h5"),
    "randomforest": joblib.load("./models/model_forest.h5"),
    "svm": joblib.load("./models/model_svm.h5"),
    "tree": joblib.load("./models/model_tree.h5")
}

# Diccionario de clases
clases_iris = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Función para hacer predicción
def hacer_prediccion(model, features):
    predict = model.predict([features])[0]
    return int(predict), clases_iris[int(predict)]

# Función para leer parámetros
def obtener_features(request):
    data = request.get_json() or request.args
    try:
        sepal_length = float(data['sepal_length'])
        sepal_width = float(data['sepal_width'])
        petal_length = float(data['petal_length'])
        petal_width = float(data['petal_width'])
    except (KeyError, ValueError):
        return None
    return [sepal_length, sepal_width, petal_length, petal_width]

# Rutas
@app.route('/', methods=['GET'])
def home():
    return 

@app.route('/predict/logistic', methods=['POST'])
def predict_logistic():
    features = obtener_features(request)
    if features is None:
        return jsonify({'error': 'Parámetros incorrectos'}), 400
    prediction, label = hacer_prediccion(modelos['logistic'], features)
    return jsonify({'modelo': 'Logistic Regression', 'prediccion': prediction, 'clase': label})

@app.route('/predict/randomforest', methods=['POST'])
def predict_randomforest():
    features = obtener_features(request)
    if features is None:
        return jsonify({'error': 'Parámetros incorrectos'}), 400
    prediction, label = hacer_prediccion(modelos['randomforest'], features)
    return jsonify({'modelo': 'Random Forest', 'prediccion': prediction, 'clase': label})

@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    features = obtener_features(request)
    if features is None:
        return jsonify({'error': 'Parámetros incorrectos'}), 400
    prediction, label = hacer_prediccion(modelos['svm'], features)
    return jsonify({'modelo': 'SVM', 'prediccion': prediction, 'clase': label})

@app.route('/predict/tree_decision', methods=['POST'])
def predict_tree():
    features = obtener_features(request)
    if features is None:
        return jsonify({'error': 'Parámetros incorrectos'}), 400
    prediction, label = hacer_prediccion(modelos['tree'], features)
    return jsonify({'modelo': 'Decision Tree', 'prediccion': prediction, 'clase': label})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  ``` </pre>


#### API de Predicción de la Flor de Iris.
#### Utiliza los endpoints para predecir la clase de una flor de iris basándote en sus características.

_______________________________________________________________________________________________________________________

Hasta este momento todo va de maravilla.
¡Ahora pasamos a la parte más interesante y divertida!

### Instalación

* Crear un entorno virtual de Anaconda (Python)

Desde tu consola:

* Accede a la carpeta `machine__learning_api`

* Instala los módulos necesarios ejecutando `pip install -r requirements.txt`  

### Modelos de Machine Learning
 
* Genera los modelos de machine laerning ejecutando: `python iris_models.py` 

![image](https://github.com/user-attachments/assets/552bb9ce-40c4-4727-a172-f16dec1fbcf9)

### Levanta el servidor:

* Para levantar el servidor ejecutamos: `python app.py`


Puedes ir a la ruta http://127.0.0.1:5001/ y verás:

![image](https://github.com/user-attachments/assets/7f491ca9-40d3-4658-9692-6eb5e34dbe85)

### ¡Y listo! Hemos aprendido a construir una API con modelos entrenados de Machin Learning que te permite realizar predicciones.
# ¡Lo volvimos a hacer! ¡Seguimos siendo *los mejores aprendiendo*! 
