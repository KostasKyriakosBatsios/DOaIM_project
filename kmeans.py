#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

csv = pd.read_csv('C:/Users/kosta/OneDrive/Υπολογιστής/New folder/ODEP/telco_2023.csv')

print(csv.head())
print(csv.info())

#from columns longmon until ebill (except callid and callwait)
col = [False, False, False, True, True, True, True, True, True, True, True, True, False, False, True, True, True, False, False]

data = csv.iloc[:, col]

#elbow method
sse = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, n_init='auto')
    kmeans.fit(data)
    sse.append(kmeans.inertia_)

plt.plot(range(1,11), sse, marker='o')
plt.title('Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('inertia/SSE')
plt.show()

kmeans = KMeans(n_clusters=3, n_init='auto')
kmeans.fit(data)

print('SSE:',kmeans.inertia_)
print('Final locations of the centroid:',kmeans.cluster_centers_)
print("The number of iterations required to converge", kmeans.n_iter_)

print(kmeans.labels_)

csv['cluster'] = kmeans.labels_.tolist()

print(csv)
csv.to_csv('clusterAssignments.csv')

df2 = pd.DataFrame(csv ,columns = ["longmon", "tollmon", "equipmon", "cardmon", "wiremon", "multline",
                                   "voice", "pager", "internet", "forward", "confer", "ebill"])
df2['Clusters']=kmeans.labels_
print(df2)
parallel_coordinates(df2, 'Clusters',color=('#383c4a','#0a3661','#dcb536'))
plt.show()