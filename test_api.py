# test_api.py

import requests
import pandas as pd

from sklearn.datasets import (
    load_breast_cancer
)

data = load_breast_cancer()

sample = pd.DataFrame(
    [data.data[0]],
    columns=data.feature_names
)

payload = {
    "dataframe_split": {
        "columns":
            sample.columns.tolist(),
        "data":
            sample.values.tolist()
    }
}

response = requests.post(
    "http://127.0.0.1:5001/invocations",
    json=payload
)

print(response.json())