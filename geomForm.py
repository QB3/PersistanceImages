import numpy as np

def randomFace(n, bool1, bool2):
    res1 = np.random.uniform(0,1, n)
    res1 = bool1 + np.logical_not(bool1) * res1
    res1 = np.logical_not(bool2) * res1
    return res1

def cube(n):
    u = np.random.uniform(0,1,n)
    bool1 = u <= 1/6
    bool2 = (u <= 2/6) & (u > 1/6)
    bool3 = (u <= 3/6) & (u > 2/6)
    bool4 = (u <= 4/6) & (u > 3/6)
    bool5 = (u <= 5/6) & (u > 4/6)
    bool6 = (u <= 6/6) & (u > 5/6)
    
    first = randomFace(n, bool1, bool2)
    second = randomFace(n, bool3, bool4)
    third = randomFace(n, bool5, bool6)
    
    res =  [first, second, third]
    res = np.transpose(res)
    return res

def circle(n):
    first = np.zeros(n)
    theta = np.random.uniform(0, 2*np.pi, n)
    second = np.cos(theta)
    third = np.sin(theta)
    res =  [first, second, third]
    res = np.transpose(res)
    return res

def sphere(n):
    theta = np.random.uniform(0, 2*np.pi, n)
    phi = np.random.uniform(0, 2*np.pi, n)
    first = np.sin(theta) * np.cos(phi)
    second = np.sin(theta) * np.sin(phi)
    third = np.cos(theta)
    res = [first, second, third]
    res = np.transpose(res)
    return res

def torrus(n):
    phi = np.random.uniform(0, 2*np.pi, n)
    theta = np.random.uniform(0, 2*np.pi, n)
    z  = 0.25 * np.sin(theta)
    lho = 0.75  + 0.25 * np.cos(theta)
    x = lho *np.cos(phi)
    y = lho * np.sin(phi)
    res = [x,y,z]
    res = np.transpose(res)
    return res

def threeCluster(n):
    m=int(np.floor(n/3))
    centers = np.random.uniform(0,1,(3,3))
    cluster0 = centers[0,:] + 0.1*sphere(m)
    cluster1 = centers[1,:] + 0.1*sphere(m)
    cluster2 = centers[2,:] + 0.1*sphere(m)
    res = np.append(cluster0, cluster1, axis=0)
    res = np.append(res, cluster2, axis=0)
    return res

def threeClusterInThreeCluster(n):
    m=int(np.floor(n/3))
    centers = np.random.uniform(0,1,(3,3))
    cluster0 = centers[0,:] + 0.1*threeCluster(m)
    cluster1 = centers[1,:] + 0.1*threeCluster(m)
    cluster2 = centers[2,:] + 0.1*threeCluster(m)
    res = np.append(cluster0, cluster1, axis=0)
    res = np.append(res, cluster2, axis=0)
    return res

