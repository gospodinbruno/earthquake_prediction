import pandas as pd
from neural_network import run_neural_network_regression

data = pd.read_csv("cleaned_earthquake_catalogue.csv")
run_neural_network_regression(data)