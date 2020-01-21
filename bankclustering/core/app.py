import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.preprocessing import StandardScaler


class Core:

    def __init__(self):
        self.data = None
        self.pred = "pred"

    def cluster(self, method):
        data = pd.read_csv("data/bank-processed.csv")

        if method == "KMeans":
            model = KMeans(n_clusters=3)
            predictions = model.fit_predict(data)
        elif method == "DBSCAN":
            predictions = dbscan() 
        elif method == "AFF":
            model = AffinityPropagation(n_cluster=3)
            predictions = model.fit_predict(data)
        elif method == "AGLOCLUST":
            predictions = agloclust()

        data[self.pred] = predictions.tolist()
        self.data = data

    def get_results(self, xaxis, yaxis):
        data_to_present = pd.DataFrame(columns=['x', 'y', 'label'])
        data_to_present["x"] = self.data[xaxis]
        data_to_present["y"] = self.data[yaxis]
        data_to_present["label"] = self.data[self.pred]

        return data_to_present

    def dbscan():
        df_clus = StandardScaler().fit_transform(data)
        model = DBSCAN(eps=0.5, min_samples=2).fit(df_clus)
        predictions = model.labels_
        return predictions

    def agloclust():
        df_clus = StandardScaler().fit_transform(data)
        model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward').fit(df_clus)
        prediction = model.labels_
        return predictions
