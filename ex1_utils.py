"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
import math
from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import linalg

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

"""
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
def myID() -> np.int:
    return 209372937


"""
Reads an image, and returns the image converted as requested
:param filename: The path to the image
:param representation: grayscale(1) or RGB(2)
:return: The image np array
"""
def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    im = cv2.imread(filename, representation-1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im/np.max(im)

    return im



def imDisplay(filename: str, representation: int):
    if representation == 1:
        plt.imshow(imReadAndConvert(filename, 1), cmap='gray')  #if its gray image
        plt.show()

    if representation == 2:
        plt.imshow(imReadAndConvert(filename, 2)) #if its rgb image
        plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    img = np.copy(imgRGB)
    rgb2yiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]]) #matrix to multiply
    imgYiq = img.dot(rgb2yiq)
    m,n,z = np.shape(imgYiq)

    for i in range(m):      #imdexes that not in range [0,1]
        for j in range(n):
            for t in range(z):
                if imgRGB[i,j,t] < 0:
                    imgRGB[i,j,t] = 0

                elif imgRGB[i,j,t] > 1:
                    imgRGB[i,j,t] = 1

    return imgYiq


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    img = np.copy(imgYIQ)
    rgb2yiq = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
    yiq2rgb = linalg.inv(rgb2yiq)  #reverse matrix to multiply
    imgRGB = img.dot(yiq2rgb)
    m,n,z = np.shape(imgRGB)

    for i in range(m):      #imdexes that not in range [0,1]
        for j in range(n):
            for t in range(z):
                if imgRGB[i,j,t] < 0:
                    imgRGB[i,j,t] = 0

                elif imgRGB[i,j,t] > 1:
                    imgRGB[i,j,t] = 1

    return imgRGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    img = np.copy(imgOrig)
    flag = isgray(imgOrig)

    yiq = transformRGB2YIQ(img)
    img = yiq[:,:,0]    #working on y channel
    img = (img+np.min(img))/np.max(img)
    img = np.round(img * 255).astype(np.uint32)
    hist = np.histogram(img, range=(0,255), bins=256)
    cumsum = np.array(np.cumsum(hist[0]))   #compute cumsum of histogram
    m, n = img.shape
    lut = []

    for i in range(len(cumsum)):   #making look up table
        lut.append(np.ceil((cumsum[i]*255)/(m*n)))

    for i in range(m):
        for j in range(n):
            if img[i][j] < 255:
                img[i][j] = lut[img[i][j]]   #changing img as the LUT

    his_new = np.histogram(img, range=(0, 255), bins=256)

    if flag == True: #gray image
        return img, hist[0], his_new[0]
    else: #rgb image
        img = img/255
        yiq[:, :, 0] = img
        img = transformYIQ2RGB(yiq)
        img = img.astype('float64')
        return img, hist[0], his_new[0]


def isgray(img):  #check if the image is grayScale
    if len(img.shape) < 3:
        return True
    if img.shape[2] == 1:
        return True
    b,g,r = img[:,:,0], img[:,:,1], img[:,:,2]
    if (b == g).all() and (b == r).all():
        return True
    return False


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    img = np.copy(imOrig)
    flag = isgray(img)
    yiq = transformRGB2YIQ(img)
    img = yiq[..., 0]  #working on y channel
    img = (img + np.min(img)) / np.max(img)
    img = np.round(img * 255).astype(np.uint32)
    hist = np.histogram(img, range=(0, 255), bins=256)
    z = [(i * 255)//nQuant for i in range(nQuant+1)]
    q = findLocalQ(z, hist)
    errors = []
    m, n = img.shape
    lut = []
    ans = []
    ans2 = []
    ans.append(imOrig)

    for i in range(nIter):
        z = findZ(q)  #compute Z
        q = findLocalQ(z, hist)  #compute q
        error = find_errors(z, q, hist)
        errors.append(error/np.size(img))  #errors calculate
        ans2.append(error/np.size(img))

    for i in range(len(z) - 1):
        for j in range(z[i], z[i + 1]):
            lut.append(q[i])   #making look up table

    for i in range(m):
        for j in range(n):
            if img[i][j] < 255:
                img[i][j] = lut[img[i][j]] #changing img as the LUT
            ans.append(img)

    if flag == True:  #gray image
        return ans, ans2

    elif flag == False:  #rgb image
        img = img / 255
        yiq[..., 0] = img
        img = transformYIQ2RGB(yiq)

        for i in range(m):
            for j in range(n):
                ans.append(img)
        print(ans[0])
        return ans, ans2


def find_errors(z, q, h):  #calculate errors
    sum = 0
    for i in range(len(z)-1):
        for j in range(z[i], z[i+1]):
            sum += ((q[i] - j) * (q[i] - j)) * h[1][j]

    return np.min(sum)


def findLocalQ(z: list, h: np.ndarray):  #calculate Q
    q = []
    j = 0
    for i in range(len(z)-1):
        denominator, numerator = 0, 0
        for j in range(z[i], z[i+1]):
            numerator += h[1][j] * j
            denominator += h[1][j]
        q.append((numerator//denominator).astype(np.uint64))

    return q


def findZ (q: list):  #calculate Z
    z = [0]
    i = 0
    for i in range(len(q)-1):
        z.append(((q[i]+q[i+1])//2).astype(np.uint64))

    z.append(255)

    return z
