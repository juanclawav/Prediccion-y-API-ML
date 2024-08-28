
# Prediccion y API con Machine Learning

Prueba Técnica en Ciencia de Datos construyendo un modelo de predicción de alquiler de monopatines en base al set de datos [en este repositorio](https://github.com/ecusidatec/D4G_Interview/blob/main/Datasets/PruebaTecnicaPasantia/dataset_alquiler.csv).

## Instrucciones: 

### Requisitos:
- Python 3.x
- Computadora con Windows

### 1. Clonar repositorio

Correr la aplicacion de línea de comando de windows como administrador y pegar los siguientes comandos:

```
git clone https://github.com/juanclawav/Prediccion-y-API-ML.git

cd Prediccion-y-API-ML
```
### 2. Crear y activar ambiente virtual
```
python -m venv env

env\Scripts\activate
```
### 3. Instalar dependencias
```
pip install -r requirements.txt
```
### 4. Inicializar API
```
python api.py
```
### 5. Probar la API
Abrir una nueva ventana de linea de comando y introducir los siguientes comandos (uno por uno)
```
$headers = @{'Content-Type' = 'application/json'}  

$body = '{"temperatura": 0.24,"humedad": 0.81,"velocidad_viento": 0,"sensacion_termica": 0.2879,"temporada": 2,"anio":0,"mes": 1,"hora": 7,"dia_semana": 6,"clima": 1,"dia_trabajo": 1,"feriado": 0}'

Invoke-WebRequest -Uri http://127.0.0.1:5000/predictLR -Method POST -Headers $headers -Body $body
```
La respuesta debe aparecer inmediatamente en la misma ventana y se ve parecida a esto:
```
Invoke-WebRequest -Uri http://127.0.0.1:5000/predictLR -Method POST -Headers $headers -Body $body


StatusCode        : 200
StatusDescription : OK
Content           : {
                      "prediction": 34.828693394768806
                    }

RawContent        : HTTP/1.1 200 OK
                    Connection: close
                    Content-Length: 39
                    Content-Type: application/json
                    Date: Wed, 28 Aug 2024 02:02:32 GMT
                    Server: Werkzeug/3.0.4 Python/3.10.6

                    {
                      "prediction": 34.82869339476880...
Forms             : {}
Headers           : {[Connection, close], [Content-Length, 39], [Content-Type, application/json], [Date, Wed, 28 Aug
                    2024 02:02:32 GMT]...}
Images            : {}
InputFields       : {}
Links             : {}
ParsedHtml        : mshtml.HTMLDocumentClass
RawContentLength  : 39
```
Para utilizar los diferentes algoritmos de aprendizaje solo hace falta editar el link en el comando InvokeWebRequest, a http://127.0.0.1:5000/predictRF (Random Forest), o a http://127.0.0.1:5000/predictDT (Arbol de decisiones)
