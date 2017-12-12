import numpy as np
import gudhi as gd
import random as rd
import pickle as pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad

def RVSimplexTree(points):
    rips_complex = gd.RipsComplex(points,max_edge_length=0.5) 
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
    diag1 = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    return simplex_tree

def ACSimplexTree(pt_cloud):
    alpha_complex = gd.AlphaComplex(points=pt_cloud)
    simplex_tree3 = alpha_complex.create_simplex_tree(max_alpha_square=60.0)
    diag3 = simplex_tree3.persistence(homology_coeff_field=2, min_persistence=0)
    return simplex_tree3

def ACPlotDiags(pt_cloud):
    alpha_complex = gd.AlphaComplex(points=pt_cloud)
    simplex_tree3 = alpha_complex.create_simplex_tree(max_alpha_square=60.0)
    diag3 = simplex_tree3.persistence(homology_coeff_field=2, min_persistence=0)
    gd.plot_persistence_diagram(diag3)

def ACgetDiagram(pt_cloud, homologyDegree):
    st = ACSimplexTree(pt_cloud)
    st.persistence(homology_coeff_field=2, min_persistence=0)
    diag = st.persistence_intervals_in_dimension(homologyDegree)
    return diag

def RVgetDiagram(pt_cloud, homologyDegree):
    st = RVSimplexTree(pt_cloud)
    st.persistence(homology_coeff_field=2, min_persistence=0)
    diag = st.persistence_intervals_in_dimension(homologyDegree)
    return diag

def listDiagrams(list_pts_cloud, homologyDegree):
    listDiag = []
    for pt_cloud in list_pts_cloud:
        diag = RVgetDiagram(pt_cloud, homologyDegree)
        listDiag.append(diag)
    return listDiag
        
def getAllListsDiagram(listData, numberOfSeriesToKeep, homologyDegree):
    listDiag = []
    for i in range(3):
        list_pts_cloud = listData[i][0:numberOfSeriesToKeep]
        listDiagToAdd = listDiagrams(list_pts_cloud, homologyDegree) 
        listDiag = listDiag+listDiagToAdd
    return listDiag

def weightFunction(arrayDiag, b):
    persistence = arrayDiag[:,1]
    boolIsGreaterThanb = persistence > b
    res = boolIsGreaterThanb + np.logical_not(boolIsGreaterThanb)*arrayDiag[:,1]/b
    return res

def gaussian(arrayDiag, x,y, sigma2, b):
    res = arrayDiag - [x, y]
    res = res**2
    res = np.sum(res, axis = 1)
    res = res /(2*sigma2)
    res = np.exp(-res)
    res = res /(2*np.pi * sigma2) #we carefully divide by sigma2 and not sqrt(sigma2) because of the 2-dimensions
    return res

def getFuction(arrayDiag, x ,y, sigma2, b):
    values =  weightFunction(arrayDiag, b) * gaussian(arrayDiag, x,y, sigma2, b)
    value = np.sum(values)
    return value
    
def integrateOnPixel(arrayDiag, sigma2, b, xStart, xEnd, yStart, yEnd):
    return dblquad(lambda x, y: getFuction(arrayDiag, x ,y, sigma2, b), yStart, yEnd, lambda x: xStart, lambda x: xEnd)

def getExpxTx2(arrayDiag0, tabx, sigma2):
    n = np.shape(arrayDiag0)[0]
    resx = arrayDiag0[0] - np.tile(tabx, (n, 1))
    resx = resx**2
    resx = resx/(2*sigma2)
    resx = np.exp(-resx)
    print("resx = ",resx)
    return resx

def fastInt(arrayDiag, tabx, taby, sigma2, b):
    resx = getExpxTx2(arrayDiag[:,0], tabx, sigma2)/(2*np.pi * sigma2)
    resy = getExpxTx2(arrayDiag[:,1], taby, sigma2)
    diagonalMatrix = weightFunction(arrayDiag, b)
    diagonalMatrix = np.diag(diagonalMatrix)    
    print("arrayDiag =", diagonalMatrix)
    res = np.dot(diagonalMatrix, resx)
    res = np.dot( np.transpose(resy), res)
    return res


def integrateOnPixelFast(arrayDiag, sigma2, b, xStart, xEnd, yStart, yEnd):
    lengthMeshx = 100
    lengthMeshy = 100
    tabx = np.linspace(xStart, xEnd, lengthMeshx)
    taby = np.linspace(yStart, yEnd, lengthMeshy)
    valuesOnMesh = fastInt(arrayDiag, tabx ,taby, sigma2, b)
    valuesOnMesh = valuesOnMesh.cumsum(axis=0).cumsum(axis=1)/(lengthMeshx*lengthMeshy)
    A = valuesOnMesh[0,0]
    B = valuesOnMesh[lengthMeshy-1, lengthMeshx-1]
    C = valuesOnMesh[lengthMeshy -1, 0]
    D = valuesOnMesh[0, lengthMeshx-1]
    print(A,B,C,D)
    res = A + B - C - D
    return res
    

def persistenceImage(diag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax):
    arrayDiag = fromListToArray(diag)
    arrayDiag[:, 1] = arrayDiag[:, 1] -arrayDiag[:, 0] #here the diagram is trannsformed from B to T(B)
    image = np.zeros((yRes, xRes))
    deltaX = (xMax-xMin)/(xRes+1)
    deltaY = (yMax-yMin)/(yRes+1)
    for j in range(xRes):
        xStart = xMin + j * deltaX
        xEnd = xMin + (j+1) * deltaX
        for i in range(yRes):
            yStart = yMin + i * deltaY
            yEnd = yMin + (i+1) * deltaY
            #print(arrayDiag)
            value = integrateOnPixel(arrayDiag, sigma2, b, xStart, xEnd, yStart, yEnd)
            image[yRes-i-1,j] = value[0]
    return image

def fromListToArray(diag):
    n = len(diag)
    m = 2
    tab = np.zeros((n , m))
    i=0
    for point in diag:
        tab[i, 0] = point[0]
        tab[i, 1] = point[1]
        i=i+1
    return tab
        

def persistenceImageFromPtCloud(pt_cloud, homologyDegree, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax):
    diag = ACgetDiagram(pt_cloud, homologyDegree)
    image = persistenceImage(diag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax)
    print(image)
    return image

def getDistMat(listDiag):
    n=len(listDiag)
    dist_mat = np.zeros((n,n))
    i=0
    for diag1 in listDiag:
        j=0
        for diag2 in listDiag:
            if(j < i):
                db = gd.bottleneck_distance(diag1,diag2)
                dist_mat[i,j] = db
                dist_mat[j,i] = db
            j=j+1
        i=i+1
    return dist_mat

def getLabel(numberOfSeriesToKeep):
    label_color=[]
    listeColor=['blue', 'red', 'green']
    for i in range(3):
        for j in range(numberOfSeriesToKeep):
            lettre = listeColor[i]
            label_color.append(lettre)
    return label_color

def getListPIfromListDiag(listDiag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax):
    listPI = []
    for diag in listDiag:
        persistIm = persistenceImage(diag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax)
        listPI.append(persistIm)
    return listPI
        
def getDistMatFromPi(listPI):
    n=len(listPI)
    dist_mat = np.zeros((n,n))
    i=0
    for PI1 in listPI:
        j=0
        for PI2 in listPI:
            if(j < i):
                dist = np.sum((PI1 - PI2)**2)
                dist_mat[i,j] = dist
                dist_mat[j,i] = dist
            j=j+1
        i=i+1
    return dist_mat

def getDistMatFromListDiag(listDiag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax):
    listPI = getListPIfromListDiag(listDiag, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax)
    dist_mat = getDistMatFromPi(listPI)
    return dist_mat
###########################################################################################################
"""
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
diag = ACgetDiagram(data_A_sample, homologyDegree)
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
arrayDiag = fromListToArray(diag)
arrayDiag[:,1] = arrayDiag[:,1] -arrayDiag[:,0]
nx=100
ny=100
x = np.linspace(xMin, xMax, nx)
y = np.linspace(yMin, yMax, ny)
z = np.array([getFuction(arrayDiag, i ,j, sigma2, b) for j in y for i in x])
Z = z.reshape(nx, ny)
plt.imshow(Z, interpolation='bilinear')
plt.show()
im = plt.imshow(Z,  extent=(xMin, xMax, yMin, yMax))  
plt.contourf(x, y, Z, 100)
plt.title('The function $\sum f(u) \phi_u(x,y)$')
plt.show()

#the persitence image :
res = persistenceImageFromPtCloud(pt_cloud, homologyDegree, sigma2, b, xRes, yRes, xMin, xMax, yMin, yMax)
plt.imshow(res)
plt.axis('off')
plt.colorbar(im)  
plt.title('Persistance Image')
plt.show()
"""
