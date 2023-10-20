#https://scipy.github.io/devdocs/reference/generated/scipy.cluster.hierarchy.dendrogram.html
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

csv = pd.read_csv('c:/Users/kosta/OneDrive/Υπολογιστής/New folder/ODEP/telco_2023.csv')

col = [False, False, False, True, True, True, True, True, True, True, True, True, False, False, True, True, True, False, False]
data = csv.iloc[:, col]

linkage_data = linkage(data, method='ward', metric='euclidean')
#linkage_data = linkage(data, method='single', metric='euclidean')
#linkage_data = linkage(data, method='complete', metric='euclidean')
dendrogram(linkage_data, color_threshold=500)
plt.show()

hierarchical_cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
#hierarchical_cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='single')
#hierarchical_cluster = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='complete')
labels = hierarchical_cluster.fit_predict(data)

print(csv)
csv.to_csv('csv_clusterAssignmentsHierarchical.csv')

df2 = pd.DataFrame(csv ,columns = ["longmon", "tollmon", "equipmon", "cardmon", "wiremon", "multline",
                                   "voice", "pager", "internet", "forward", "confer", "ebill"])
df2['Clusters']=hierarchical_cluster.labels_
print(df2)
parallel_coordinates(df2, 'Clusters',color=('#383c4a','#0a3661','#dcb536'))
plt.show()
