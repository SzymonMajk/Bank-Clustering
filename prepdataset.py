import pandas as pd

data = pd.read_csv("data/bank-full.csv")
data.drop(["contact","pdays","previous","poutcome","y","month","day","campaign"], axis=1, inplace=True)
data = data[:4000]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["job"] = le.fit_transform(data["job"])
data["marital"] = le.fit_transform(data["marital"])
data["education"] = le.fit_transform(data["education"])
data["default"] = le.fit_transform(data["default"])
data["housing"] = le.fit_transform(data["housing"])
data["loan"] = le.fit_transform(data["loan"])

data.to_csv("data/bank-processed.csv", index=False)