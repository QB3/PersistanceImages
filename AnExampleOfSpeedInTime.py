import numpy as np
import pickle as pickle
import gudhi as gd
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from kmedoids import kMedoids
import PersistenceImages as  pi
import time

from PersistenceImages import RVSimplexTree
from PersistenceImages import ACSimplexTree

f = open("data_acc.dat","rb")
data = pickle.load(f,encoding="latin1")
f.close()
data_A = data[0]
data_B = data[1]
data_C = data[2]
label = data[3]

data_A_sample = data_A[0]
data_B_sample = data_B[0]
data_C_sample = data_C[0]
data_sample = [data_A_sample, data_B_sample, data_C_sample]
print(data_A_sample.shape)

listData = [data_A, data_B, data_C]
numberOfSeriesToKeep = 30
homologyDegree = 1
print("homology degree = ", homologyDegree)
listDiag = pi.getAllListsDiagram(listData, numberOfSeriesToKeep, homologyDegree)
start = time.time()
dist_mat=pi.getDistMat(listDiag)
timeSpent = time.time()-start
plt.imshow(dist_mat)


label_color = pi.getLabel(numberOfSeriesToKeep)
#MDS on the distance matrix from diagrams
B1 =dist_mat
# B is the pairwise distance matrix between 0 or 1-dim dgms
#label_color contains the colors corresponding to the class of each dgm
mds = manifold.MDS(n_components=3, max_iter=3000, eps=1e-9,
dissimilarity="precomputed", n_jobs=1)
pos1 = mds.fit(B1).embedding_
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos1[:,0], pos1[:, 1], pos1[:,2], marker = 'o', color= label_color)
plt.title('MDS avec la matrice des distances entre diagrammes')
plt.show()
print("temps écoulé pour calculer la matrice des distances entre diagrammes : ", timeSpent)

#we set the parameters
pt_cloud = data_B_sample
homologyDegree = 1
sigma2=0.0001
b=0.02
xRes = 5
yRes = 5
xMin = 0
xMax = 0.2
yMin = 0
yMax = 0.02

start = time.time()
dist_mat = pi.getDistMatFromListDiag(listDiag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax)
timeSpent = time.time()-start
plt.imshow(dist_mat)

#MDS on the distance matrix from PIs
B1 =dist_mat
# B is the pairwise distance matrix between 0 or 1-dim dgms
#label_color contains the colors corresponding to the class of each dgm
mds = manifold.MDS(n_components=3, max_iter=3000, eps=1e-9,
dissimilarity="precomputed", n_jobs=1)
pos1 = mds.fit(B1).embedding_
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos1[:,0], pos1[:, 1], pos1[:,2], marker = 'o', color= label_color)
plt.title('MDS avec la matrice des distances entre PI')
plt.show()
print("temps écoulé pour calculer la transformation en PI + calcul de la matrice des distances entre PI: ", timeSpent)

