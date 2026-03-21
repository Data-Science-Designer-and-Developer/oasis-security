# Utilisation MLflow tracking
mlflow.set_tracking_uri("http://host.docker.internal:5000")
mlflow.set_experiment("crime_predictor_prod")

logged_model = 'runs:/<run_id>/model'

# Load model as predictor
model_uri = f"models:/crime_predictor_prod/1"
loaded_model = mlflow.pyfunc.load_model(model_uri)
prediction = loaded_model.predict(data)