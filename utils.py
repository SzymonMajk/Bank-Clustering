import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preparedataset(data):
    data.drop(["contact","duration","pdays","previous","poutcome","y","month","day","campaign"], axis=1, inplace=True)
    data = data[:4000]
    
    le = LabelEncoder()
    
    data["job"] = le.fit_transform(data["job"])
    data["marital"] = le.fit_transform(data["marital"])
    data["education"] = le.fit_transform(data["education"])
    data["default"] = le.fit_transform(data["default"])
    data["housing"] = le.fit_transform(data["housing"])
    data["loan"] = le.fit_transform(data["loan"])
    
    return data
