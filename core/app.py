import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation


class Core:

    def cluster(self, method):
        data = pd.read_csv("data/bank-full.csv")

        if method == "KMeans":
            model = KMeans(n_clusters=3)
        elif method == "DBSCAN":
            model = DBSCAN()
        elif method == "AFF":
            model = AffinityPropagation(n_cluster=3)

        predictions = model.fit_predict(data)
        data["pred"] = predictions.tolist()

    def get_results(self, xaxis, yaxis):
        data_to_present = pd.DataFrame(columns=['x','y','label'])
        data_to_present["x"] = data[xaxis]
        data_to_present["y"] = data[yaxis]
        data_to_present["label"] = data[pred]

        return data_to_present