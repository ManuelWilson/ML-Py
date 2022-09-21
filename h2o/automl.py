from sklearn.datasets import load_iris
import h2o
from h2o.automl import H2OAutoML
import numpy as np

h2o.init()

X,y = load_iris().data, load_iris().target
X = np.insert(X, 1, y, axis=1)
X = h2o.H2OFrame(X)


# Run AutoML for 20 base models
aml = H2OAutoML(max_runtime_secs = 200, max_runtime_secs_per_model = 40, seed=1)
aml.train(y='C5', training_frame=X)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)




prostate = h2o.import_file("http://s3.amazonaws.com/h2o-public-test-data/smalldata/prostate/prostate_complete.csv.zip")

# Set the predictor names and the response column name
response = "CAPSULE"
predictor = prostate.names[2:9]

# Train AutoML
aml = H2OAutoML(max_models = 5,
                max_runtime_secs = 200,
                max_runtime_secs_per_model = 40,
                seed = 1234)
aml.train(x = predictor, y = response, training_frame = prostate)
