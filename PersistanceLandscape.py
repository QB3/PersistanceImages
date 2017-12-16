import numpy as np
import matplotlib.pyplot as plt

def landscapes_approx(diag,p_dim,x_min,x_max,nb_nodes,nb_ld):
    landscape = np.zeros((nb_ld, nb_nodes))
    intervalle = np.linspace(x_min, x_max, nb_nodes)
    j=0
    for x in intervalle:
        #print("x=", x )
        listLandscape = []
        for point in diag:
            b = point[0]
            d = point[1]
            boolIsInbd = ( np.sqrt(2) * b <= x) & (x <= np.sqrt(2) * d)
            if(boolIsInbd):
                if(x < (b+d)/np.sqrt(2) ):
                    value =  x - np.sqrt(2) * b
                    listLandscape.append(value)
                else:
                    value =  np.sqrt(2) * d-x
                    listLandscape.append(value)
        sortedList = sorted(listLandscape, reverse=True)
        #print("sortedList = ",sortedList)
        i=0
        for val in sortedList:
            if(i < nb_ld):
                landscape[i, j] = val
            i=i+1
        j = j+1
    #print(landscape)
    return landscape

def plotLandscape(diag, p_dim,x_min,x_max,nb_nodes,nb_ld):
    p_dim =0
    L = landscapes_approx(diag,p_dim,x_min,x_max,nb_nodes,nb_ld)
    plt.figure(1)
    plt.plot(np.linspace(x_min,x_max, num=nb_nodes),L[0:nb_ld,:].transpose())
    #plt.axes().set_aspect('equal')
    plt.show()
    