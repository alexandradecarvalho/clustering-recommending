from pyspark import SparkContext
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.cluster import KMeans
from pyspark import AccumulatorParam
from sklearn.decomposition import PCA

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import ast
import math
import re

df = pd.read_csv('data/tracks.csv', index_col=0, header=[0, 1])

COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),('track', 'genres'), ('track', 'genres_all')]
for column in COLUMNS:
    df[column] = df[column].map(ast.literal_eval)

COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),('album', 'date_created'), ('album', 'date_released'),('artist', 'date_created'), ('artist', 'active_year_begin'),('artist', 'active_year_end')]
for column in COLUMNS:
    df[column] = pd.to_datetime(df[column])

SUBSETS = ('small', 'medium', 'large')
df['set', 'subset'] = df['set', 'subset'].astype(pd.CategoricalDtype(categories=SUBSETS, ordered=True))

COLUMNS = [('track', 'genre_top'), ('track', 'license'),('album', 'type'), ('album', 'information'),('artist', 'bio')]
for column in COLUMNS:
    df[column] = df[column].astype('category')

small_tracks_ds = df[df[('set','subset')] == 'small']

f = pd.read_csv('data/features.csv', index_col=0, header=[0, 1, 2])
small_features_ds = f[f.index.isin(small_tracks_ds.index)]

feature_vectors = small_features_ds.to_numpy()

radius = []
diameter = []
density = []

def euclidean_distance(point1,point2):
    return math.sqrt(sum([(point1[x] - point2[x])**2 for x in range(len(point2))]))
    
for k in range(8,17):
    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean')
    cluster.fit_predict(feature_vectors)
    clf = NearestCentroid()
    centroids = clf.fit(feature_vectors, cluster.labels_).centroids_
    radius += [[max([euclidean_distance(feature_vectors[idx], centroids[cluster_number]) for idx in range(len(feature_vectors)) if cluster.labels_[idx] == cluster_number]) for cluster_number in range(k)]]
    diameter += [[max([euclidean_distance(feature_vectors[idx1], feature_vectors[idx2]) for idx1 in range(len(feature_vectors)) for idx2 in range(len(feature_vectors)) if idx1 > idx2 and cluster.labels_[idx1] == cluster_number and cluster.labels_[idx2] == cluster_number]) for cluster_number in range(k)]]
    density += [[len([feature_vectors[idx] for idx in range(len(feature_vectors)) if cluster.labels_[idx] == cluster_number]) / diameter[-1][cluster_number] for cluster_number in range(k)]]

r2 = list(map(lambda lst: sum(lst) / len(lst), radius))
plt.plot(range(8,17), r2)

diam2 = list(map(lambda lst: sum(lst) / len(lst), diameter))
plt.plot(range(8,17), diam2)

dens2 = list(map(lambda lst: sum(lst) / len(lst), density))
plt.plot(range(8,17), dens2)

k = 10

sc = SparkContext(appName="Assignment2")

data = sc.textFile('data/features.csv').filter(lambda row: row[0].isnumeric()).map(lambda line: line.split(","))

sampled_data = data.sample(withReplacement=False, fraction=0.08)
sampled_data = [list(map(float, point)) for point in sampled_data.collect()]

kmeans = KMeans(n_clusters=k, random_state=0).fit([x[1:] for x in sampled_data])

class VectorAccumulatorParam(AccumulatorParam):
    def zero(self,  k = k):
        return list([[0,[0]*518,[0]*518]]*k)

    def addInPlace(self, val1, val2):
        val1 = [[np.add(val1[c][x], val2[c][x]) for x in range(3)] for c in range(10)]
        return val1 

vap = VectorAccumulatorParam()

discard_set = vap.zero()
discard_set = vap.addInPlace(discard_set, [[np.sum([kmeans.labels_ == cluster_number]), [np.sum([sampled_data[idx][feature] for idx in range(len(sampled_data)) if kmeans.labels_[idx] == cluster_number]) for feature in range(1, len(sampled_data[0]))], [np.sum([np.power(sampled_data[idx][feature],2) for idx in range(len(sampled_data)) if kmeans.labels_[idx] == cluster_number]) for feature in range(1, len(sampled_data[0]))]] for cluster_number in range(k)])

class DictAccumulatorParam(AccumulatorParam):
    def zero(self):
        return dict() 

    def addInPlace(self, d1, d2):
        for k,v in d2.items():
            d1[k] = v
        return d1

dap = DictAccumulatorParam()

final_clustering = dap.zero()
final_clustering = dap.addInPlace(final_clustering, {int(sampled_data[idx][0]): kmeans.labels_[idx] for idx in range(len(sampled_data))})

x = data.count()

cluster_division = list(range(0,x, 8414)) + [x]

compression_set = sc.emptyRDD()
retained_set = sc.emptyRDD()

def get_variance(summary):
    return np.subtract(np.divide(summary[2], summary[0]), np.power(np.divide(summary[1],summary[0]),2))

std = [np.sqrt(abs(get_variance(discard_set[centroid_idx]))) for centroid_idx in range(k)]
removable_features = [[i for i, x in enumerate(std[f]) if x == float(0)] for f in range(k)]
removable_features = list(set(x for l in removable_features for x in l))

def filtered(point):
    return [point[idx] for idx in range(len(point)) if idx not in removable_features]

centroids = [filtered(centroid) for centroid in kmeans.cluster_centers_.tolist()] 
std = [filtered(deviation) for deviation in std] 
discard_set = [[discard_set[c][0], filtered(discard_set[c][1]), filtered(discard_set[c][2])] for c in range(k)]
d = len(std[0])

def mahalanobis_distance(point, centroid_idx):
    return np.sqrt(np.sum(np.power(np.divide(np.subtract([float(i) for i in point], centroids[centroid_idx]), std[centroid_idx]), 2)))

def cluster_point(point):
    global final_clustering
    global discard_set
    
    final_clustering = dap.addInPlace(final_clustering, {int(point[0]): point[-1]})
    
    add_n = [[0,0,0] for i in range(k)] 
    add_n[point[-1]] = [1,[float(f) for f in point[1:-1]],np.power([float(f) for f in point[1:-1]],2)]
    
    discard_set = vap.addInPlace(discard_set, add_n)

def merging(ccluster):
    global final_clustering
    global discard_set
    global retained_set
    global indexes_for_filtering

    set_summary = [len(ccluster), [np.sum([rs[idx][feature] for idx in ccluster]) for feature in range(1,len(rs[0]))], [np.sum([np.power(rs[idx][feature],2) for idx in ccluster]) for feature in range(1,len(rs[0]))]]
    new_summaries = [[set_summary[0] + discard_set[c][0], np.add(discard_set[c][1], set_summary[1]), np.add(discard_set[c][2], set_summary[2])] for c in range(k)] 
    
    if [c for c in range(k) if all(var < d*1.1 for var in get_variance(new_summaries[c]))]:
        final_c = min(range(k), key = [get_variance(new_summaries[c]) for c in range(k)].__getitem__)
        final_clustering = dap.addInPlace(final_clustering, {int(rs[idx][0]) : final_c for idx in ccluster})

        add_n = [[0,0,0] for i in range(k)] 
        add_n[final_c] = set_summary

        discard_set = vap.addInPlace(discard_set, add_n)
        
        indexes_for_filtering = iap.addInPlace(indexes_for_filtering, [idx for idx in ccluster])

class IndexesAccumulatorParam(AccumulatorParam):
    def zero(self):
        return []

    def addInPlace(self, val1, val2):
        val1 += val2
        return val1 

iap = IndexesAccumulatorParam()

for i in range(len(cluster_division) - 1):
    print("Batch",i, "/", len(cluster_division) - 1)
    
    loaded_points = data.filter(lambda point: int(point[0]) > cluster_division[i] and  int(point[0]) <= cluster_division[i+1] and int(point[0]) not in list(final_clustering.keys()))
    loaded_points = loaded_points.map(lambda point: [point[0]] + filtered(point[1:]) + [min(range(k), key = [mahalanobis_distance(filtered(point[1:]), c) for c in range(k)].__getitem__)] if any([mahalanobis_distance(filtered(point[1:]), c)  < math.sqrt(d) for c in range(k)]) else [point[0]] + filtered(point[1:]))

    loaded_points.filter(lambda point: len(point) > (d+1)).foreach(cluster_point)
    retained_set = sc.union([retained_set, loaded_points.filter(lambda point: len(point) == (d+1))])

    if not retained_set.isEmpty():
        comp_cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', distance_threshold=2*math.sqrt(d))
        rs = retained_set.collect()
        rs = [[int(point[0])] + [float(p) for p in point[1:]] for point in rs]
        comp_cluster.fit_predict([point[1:] for point in rs])
        compression_set = sc.parallelize([[idx for idx in range(len(rs)) if comp_cluster.labels_[idx] == c] for c in range(comp_cluster.n_clusters_)])

        indexes_for_filtering = iap.zero()
        compression_set.filter(lambda ccluster: len(ccluster) > 1).foreach(merging)   

        if indexes_for_filtering:
            retained_set.filter(lambda point: point[0] not in [int(rs[idx][0]) for idx in indexes_for_filtering])         

cluster_by_genre = pd.DataFrame([f.loc[trackId].tolist() + [clusterId,df.loc[trackId][('track', 'genre_top')]] for trackId,clusterId in final_clustering.items()])

pca = PCA(n_components=2)

visualization_df = pd.DataFrame(pca.fit_transform(cluster_by_genre.iloc[:,0:-2]).tolist()).join(cluster_by_genre[[518, 519]])
visualization_df

cluster_labels = visualization_df[518].unique()

genre_labels = visualization_df[519].unique()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#09ea69','#9a8e77', '#008080', '#b784a7', '#51101f', '#fbeb87']

for clusterId in sorted(cluster_labels):
   plt.figure(figsize=(20, 10))
   plt.title("Genres in Cluster "+str(clusterId))

   for idx, genreId in enumerate(genre_labels):
      plt.scatter(visualization_df[visualization_df[518] == clusterId][visualization_df[519] == genreId][[0]], visualization_df[visualization_df[518] == clusterId][visualization_df[519] == genreId][[1]] , color = colors[idx])
   plt.show()

