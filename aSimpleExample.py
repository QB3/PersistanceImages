###########################################################################################################
import numpy as np
import gudhi as gd
import random as rd
import pickle as pickle
import PersistenceImages as  pi
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

f = open("data_acc.dat","rb")
data = pickle.load(f,encoding="latin1")
f.close()
data_A = data[0]
data_A_sample = data_A[0]
data_B = data[1]
data_B_sample = data_B[0]
data_C = data[2]
data_C_sample = data_C[0]

#we set the parameters
pt_cloud = data_B_sample
homologyDegree = 1
sigma2=0.0001
b=0.02
xRes = 20
yRes = 10
xMin = 0
xMax = 0.2
yMin = 0
yMax = 0.02
###first we plot the data : 
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(data_A_sample[:,0], data_A_sample[:,1], data_A_sample[:,2])
plt.title('Data')
plt.show()

###The the peristence diagram for the degree 1 of homology
diag = pi.ACgetDiagram(data_A_sample, homologyDegree)
lx=[]
ly=[]
lTy=[]
for points in diag:
    lx.append(points[0])
    ly.append(points[1])
    lTy.append(points[1]-points[0])

#The persistence diagram B
plt.plot(lx, ly, 'ro')
plt.title('Persistence digram B for the first group homology')
plt.axes().set_aspect('equal')
plt.show()  

#the transformed diagram T(B)
plt.plot(lx, lTy, 'ro')
plt.title('Transformed diagram T(B)')
plt.axes().set_aspect('equal')
plt.show()  

#the function :
arrayDiag = pi.fromListToArray(diag)
arrayDiag[:,1] = arrayDiag[:,1] -arrayDiag[:,0]
nx=100
ny=100
x = np.linspace(xMin, xMax, nx)
y = np.linspace(yMin, yMax, ny)
z = np.array([pi.getFuction(arrayDiag, i ,j, sigma2, b) for j in y for i in x])
Z = z.reshape(nx, ny)
plt.imshow(Z, interpolation='bilinear')
plt.show()
im = plt.imshow(Z,  extent=(xMin, xMax, yMin, yMax))  
plt.contourf(x, y, Z, 100)
plt.title('The function $\sum f(u) \phi_u(x,y)$')
plt.show()

#the persitence image :
res = pi.persistenceImageFromPtCloud(pt_cloud, homologyDegree, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax)
im=plt.imshow(res)
plt.axis('off')
plt.colorbar(im)  
plt.title('Persistance Image')
plt.show()