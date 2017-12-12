from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import geomForm as gf

import numpy as np
import pickle as pickle
import gudhi as gd
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from kmedoids import kMedoids
import PersistenceImages as  pi
import time

nbPoints = 500
shape = gf.threeClusterInThreeCluster(nbPoints)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(shape[:,0], shape[:,1], shape[:,2])
plt.title('Data')
plt.show()


listData = []
label_color =  []
for i in range(25):
    listData.append(gf.cube(nbPoints))
    label_color.append("blue")
    listData.append(gf.circle(nbPoints))
    label_color.append("green")
    listData.append(gf.sphere(nbPoints))
    label_color.append("red")
    listData.append(gf.torrus(nbPoints))
    label_color.append("yellow")
    listData.append(gf.threeCluster(nbPoints))
    label_color.append("black")
    listData.append(gf.threeClusterInThreeCluster(nbPoints))
    label_color.append("pink")
    
pi.ACPlotDiags(listData[0])
pi.ACPlotDiags(listData[1])
pi.ACPlotDiags(listData[2])

##Clustering with peristence diagrams
homologyDegree = 1
print("homology degree = ", homologyDegree)
listDiag = pi.listDiagrams(listData, homologyDegree)
print("List of diagram computed !")
start = time.time()
dist_mat=pi.getDistMat(listDiag)
timeSpent = time.time()-start
plt.imshow(dist_mat)

nbclusters=6
#k-medoid classification
results = kMedoids(dist_mat, nbclusters)
clusters = results[1]

def getIndivInCluster(clusters, numero):
    indivInCluster = clusters[numero]
    colorInCluster = [label_color[i] for i in indivInCluster]
    return colorInCluster

def errorInCluster(cluster, nbclusters):
    count = np.zeros(nbclusters)
    for point in cluster:
        if(point == "blue"):
            count[0] = count[0]+1
        if(point == "green"):
            count[1] = count[1]+1
        if(point == "red"):
            count[0] = count[0]+1
        if(point == "yellow"):
            count[1] = count[1]+1
        if(point == "black"):
            count[0] = count[0]+1
        if(point == "ponk"):
            count[1] = count[1]+1
    colorCluster = np.argmax(count) 
    count[colorCluster] = 0
    res = np.sum(count)
    return res


nInitialisation = 10000
errorFinal = 10**25

for i in range(nInitialisation):
    errorTot=0
    for i in range(nbclusters):
        cluster = getIndivInCluster(clusters, i)
        error = errorInCluster(cluster, nbclusters)
        errorTot = errorTot + error
        #print("error in cluster i", error)
        #print("individuals in cluster : ", i, cluster)
    if(errorFinal > errorTot):
        errorFinal = errorTot
        clusterFinal = cluster
print(errorFinal)
B1 =dist_mat
# B is the pairwise distance matrix between 0 or 1-dim dgms
#label_color contains the colors corresponding to the class of each dgm
mds = manifold.MDS(n_components=6, max_iter=3000, eps=1e-9,
dissimilarity="precomputed", n_jobs=1)
pos1 = mds.fit(B1).embedding_
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos1[:,0], pos1[:, 1], pos1[:,2], marker = 'o', color= label_color)
plt.title('MDS avec la matrice des distances entre PI')
plt.show()
print("temps écoulé pour calculer la transformation en PI + calcul de la matrice des distances entre PI: ", timeSpent)