arrayDiag = np.zeros((2,2))
arrayDiag[0,0] = xMax
arrayDiag[0,1] = yMax

res1 = integrateOnPixelFast(arrayDiag, sigma2, b, xStart, xEnd, yStart, yEnd)
print(res1/10)
res = integrateOnPixel(arrayDiag, sigma2, b, xStart, xEnd, yStart, yEnd)
print(res)