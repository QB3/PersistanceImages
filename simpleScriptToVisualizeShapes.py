from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import geomForm as gf
import PersistenceImages as pi
import PersistanceLandscape as pl

import numpy as np
import pickle as pickle
import gudhi as gd
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from kmedoids import kMedoids
import PersistenceImages as  pi
import time

#parameters
sigma2 = 0.01
b=0.4
xRes = 20
yRes = 20
xMin = 0
xMax = 0.1
yMin = 0
yMax = 0.02

#three cluster in three cluster
nbPoints = 500
shape = gf.threeClusterInThreeCluster(nbPoints)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(shape[:,0], shape[:,1], shape[:,2])
plt.title('threeClusterInThreeCluster')
plt.show()
diag = pi.ACPlotDiags(shape)
pl.plotLandscape(diag, p_dim=0,x_min=0,x_max=0.2,nb_nodes=100,nb_ld=5)

listDiag = [diag]
res = pi.getListPIfromListDiag(listDiag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax)
plt.imshow(res[0]) 


#sphere
shape = gf.sphere(nbPoints)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(shape[:,0], shape[:,1], shape[:,2])
plt.title('Sphere')
plt.show()
diag = pi.ACPlotDiags(shape)
pl.plotLandscape(diag, p_dim=0,x_min=0,x_max=0.2,nb_nodes=100,nb_ld=5)

listDiag = [diag]
res = pi.getListPIfromListDiag(listDiag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax)
plt.imshow(res[0])

#circle
shape = gf.circle(nbPoints)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(shape[:,0], shape[:,1], shape[:,2])
plt.title('Circle')
plt.show()
diag = pi.ACPlotDiags(shape)
pl.plotLandscape(diag, p_dim=0,x_min=0,x_max=1.5,nb_nodes=100,nb_ld=5)

listDiag = [diag]
res = pi.getListPIfromListDiag(listDiag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax)
plt.imshow(res[0])


#Cube
shape = gf.cube(nbPoints)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(shape[:,0], shape[:,1], shape[:,2])
plt.title('Cube')
plt.show()
diag = pi.ACPlotDiags(shape)
pl.plotLandscape(diag, p_dim=0,x_min=0,x_max=0.05,nb_nodes=100,nb_ld=5)

listDiag = [diag]
res = pi.getListPIfromListDiag(listDiag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax)
plt.imshow(res[0])


#threeCluster
shape = gf.threeCluster(nbPoints)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(shape[:,0], shape[:,1], shape[:,2])
plt.title('Three Clusters')
plt.show()
diag = pi.ACPlotDiags(shape)
pl.plotLandscape(diag, p_dim=0,x_min=0,x_max=0.05,nb_nodes=100,nb_ld=5)

listDiag = [diag]
res = pi.getListPIfromListDiag(listDiag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax)
plt.imshow(res[0])

#torrus
shape = gf.torrus(nbPoints)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(shape[:,0], shape[:,1], shape[:,2])
plt.title('Torrus')
plt.show()
diag = pi.ACPlotDiags(shape)
pl.plotLandscape(diag, p_dim=0,x_min=0,x_max=0.05,nb_nodes=100,nb_ld=5)

listDiag = [diag]
res = pi.getListPIfromListDiag(listDiag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax)
plt.imshow(res[0])

