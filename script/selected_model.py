import mlflow
import dagshub

dagshub.init(repo_owner='RattipongMark', repo_name='MLOps-RainPrediction', mlflow=True)

model = mlflow.pyfunc.load_model("models:/rain_model/Production")


# ใช้ predict
import pandas as pd

sample = pd.DataFrame([{
    "temperature_2m": 30,
    "relative_humidity_2m": 65,
    "dew_point_2m": 24,
    # …
}])

prediction = model.predict(sample)
print(prediction)
