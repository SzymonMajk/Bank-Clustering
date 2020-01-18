import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation


class Core:

    def __init__(self):
        self.data = None
        self.pred = "pred"

    def cluster(self, method):
        data = pd.read_csv("data/bank-processed.csv")

        if method == "KMeans":
            model = KMeans(n_clusters=3)
        elif method == "DBSCAN":
            model = DBSCAN()
        elif method == "AFF":
            model = AffinityPropagation(n_cluster=3)

        predictions = model.fit_predict(data)
        data[self.pred] = predictions.tolist()
        self.data = data

    def get_results(self, xaxis, yaxis):
        data_to_present = pd.DataFrame(columns=['x', 'y', 'label'])
        data_to_present["x"] = self.data[xaxis]
        data_to_present["y"] = self.data[yaxis]
        data_to_present["label"] = self.data[self.pred]

        return data_to_present