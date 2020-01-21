import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, normalize


class Core:

    def __init__(self):
        self.data = None
        self.pred = "pred"

    def agglomerativeclustering(self):
        model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
        sample_data = self.data[:4000]
        predictions = model.fit_predict(sample_data)
        sample_data['classes'] = predictions.tolist()
        return predictions

    def cluster(self, method):
        data = pd.read_csv("data/bank-processed.csv")
        normalized = normalize(data, axis=0, norm='max')

        if method == "KMeans":
            model = KMeans(n_clusters=3)
            predictions = model.fit_predict(normalized)
        elif method == "DBSCAN":
            predictions = self.dbscan(normalized)
        elif method == "AFF":
            model = AffinityPropagation(preference=-3000, max_iter=25)
            predictions = model.fit_predict(data)
        elif method == "AGLOCLUST":
            predictions = self.agloclust(normalized)

        data[self.pred] = predictions.tolist()
        self.data = data

    def get_results(self, xaxis, yaxis):
        data_to_present = pd.DataFrame(columns=['x', 'y', 'label'])
        data_to_present["x"] = self.data[xaxis]
        data_to_present["y"] = self.data[yaxis]
        data_to_present["label"] = self.data[self.pred]

        return data_to_present

    def dbscan(self,data):
        df_clus = StandardScaler().fit_transform(data)
        model = DBSCAN(eps=3.2, min_samples=15).fit(df_clus)
        predictions = model.labels_
        return predictions

    def agloclust(self,data):
        df_clus = StandardScaler().fit_transform(data)
        model = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward').fit(df_clus)
        predictions = model.labels_
        return predictions
