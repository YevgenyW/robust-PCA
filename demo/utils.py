import os 
from os.path import join as pjoin
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
import time
mpl.rcParams['figure.figsize'] = (15,10)
mpl.rcParams['image.cmap'] = 'inferno'

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def imread(imgpath):
    return cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)

def imshow(img, cmap=None):
    plt.title(img.shape)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
#     plt.show()

def resize(img, imgshape):
    return cv2.resize(img, (imgshape[1], imgshape[0]))

def stack_images(matrices, k=5, imgshape=None, captions=None, leftcaption=None):
    """
        matrices -- (N, *imgshape, k) array,
                    where N is the number of `matrices`,
                          k is the number of images in each `matrix`
    """
    assert imgshape is not None, 'Please specify the spatial size of the frames (imgshape)'
    k = min(matrices[0].shape[-1], k)
    N = len(matrices)
    num_cols, num_rows = N, 1
    if captions is None:
        captions = ['']*N
    matrices_ = []
    if len(imgshape) == 2:
        for m in matrices:
            M = m.reshape((*imgshape,-1)).transpose(2,0,1)[:k]
            matrices_.append(M)
    else:
        for m in matrices:
            M = m.reshape((*imgshape,-1)).transpose(3,0,1,2)[:k]
            matrices_.append(M)
    for k_ in range(k):
        for i in range(N):
            plt.subplot(num_rows, num_cols, i+1)
            imshow(matrices_[i][k_], 'gray')
            plt.title(captions[i])
            if i == 0:
                plt.title(leftcaption, loc='left')
        plt.show()
        
def get_data_matrix(dataset_path, imgshape, m=None, m_start=0, m_freq=1, color=False, ext='jpg'):
    """
    imgshape -- np.array([h, w]) -- spatial size of the frames
    m -- number of frames,
         if None then all the images in dataset_path will be used
    m_start -- number of the frame from which data matrix starts
    m_freq -- frequency of frames to collect in data matrix
    """
    imgpaths = sorted(glob.glob(pjoin(dataset_path, '*.{}'.format(ext))))
    m = len(imgpaths) // m_freq if m is None else min(m, len(imgpaths) // m_freq)
    if color:
        imgshape = np.array([*imgshape, 3])
        X = resize(imread(imgpaths[m_start]), imgshape).reshape((-1,1))
        for imgpath in imgpaths[m_start + m_freq : m_start + m*m_freq : m_freq]:
            img = resize(imread(imgpath), imgshape)
            img_vec = img.reshape((-1,1))
            X = np.hstack((X, img_vec))
    else:
        X = resize(gray(imread(imgpaths[m_start])), imgshape).reshape((-1,1))
        for imgpath in imgpaths[m_start + m_freq : m_start + m*m_freq : m_freq]:
            img = resize(gray(imread(imgpath)), imgshape)
            img_vec = img.reshape((-1,1))
            X = np.hstack((X, img_vec))
    return X

def getMeanTime(f, params, iters=100):
    dts = []
    for it in range(iters):
        start_time = time.time()
        f(*params)
        dt = time.time() - start_time
        dts.append(dt)
    return np.mean(dts)

#####################################   Background Modeling with SVD, RPCA, ROSL   #####################################
from scipy import linalg as LA
import sys
sys.path.append('../rosl')
sys.path.append('../rpca')
from r_pca import R_pca
from pyrosl import ROSL

def bmSVD(X, k, color=False):
    """
    k -- rank of the low-rank matrix approximation
    """
    U, S, VT = LA.svd(X, full_matrices=False)
    Sigma = np.eye(k)*S[:k]
    A = U[:,:k].dot(Sigma).dot(VT[:k,:])

    if color:
        # A = np.round(np.minimum(np.maximum(A, 0), 255)).astype(np.uint8)
        # A = np.minimum(np.maximum(A, 0), 255).astype(np.uint8)
        # E = np.maximum(X-A,0)
        # A = A.astype(np.uint8)
        A = np.round(np.minimum(np.maximum(A, 0), 255)).astype(np.uint8)
    E = X - A
    return A, E

def bmRPCA(X, mu=None, lmbda=None, color=False, out=False):
    rpca = R_pca(X, mu=mu, lmbda=lmbda)
    A, E = rpca.fit()
    if out:
        print('Rank of A = {}'.format(np.linalg.matrix_rank(A)))
    if color:
        A = A.astype(np.uint8)
        E = np.maximum(E, 0).astype(np.uint8)
    return A, E

def bmROSL(X, k=1, reg=1, tol=1e-06, iters=500):
    rosl = ROSL(rank=k, reg=reg, tol=tol, iters=iters, verbose=False)
    D = rosl.fit_transform(X.astype(np.float64))
    C = rosl.components_
    A = D.dot(C)
    E = X - A
    return A, E

def bmCompare(dataset_path, imgshape, m=None, m_start=0, m_freq=1, color=False, ext='jpg', leftcaption='', 
              show_inline=True,
              k=1, mu=0.0001, lmbda=0.01,
              reg=1, tol=1e-06, iters=500):
    X = get_data_matrix(dataset_path, imgshape, m, m_start, m_freq, color, ext)
    SVD, RPCA, ROSL = {}, {}, {}
    SVD['A'], SVD['E'] = bmSVD(X, k)
    RPCA['A'], RPCA['E'] = bmRPCA(X, mu, lmbda)
    ROSL['A'], ROSL['E'] = bmROSL(X, k, reg, tol, iters)
    
    if show_inline:
        captions = ['Raw', 'BG (SVD)', 'BG (RPCA)', 'BG (ROSL)', 'FG (SVD)', 'FG (RPCA)', 'FG (ROSL)']
        M = (X, SVD['A'], RPCA['A'], ROSL['A'], SVD['E'], RPCA['E'], ROSL['E'])
        stack_images(M, k=1, imgshape=imgshape, captions=captions, leftcaption=leftcaption)
    else:
        captions = ['Raw Frame', 'BG (SVD)', ' BG (RPCA)', 'BG (ROSL)']
        M = (X, SVD['A'], RPCA['A'], ROSL['A'])
        stack_images(M, k=1, imgshape=imgshape, captions=captions, leftcaption=leftcaption)

        captions = ['Raw Frame', 'FG (SVD)', ' FG (RPCA)', 'FG (ROSL)']
        M = (X, SVD['E'], RPCA['E'], ROSL['E'])
        stack_images(M, k=1, imgshape=imgshape, captions=captions, leftcaption=leftcaption)
    
    
def bmCompareTime(dataset_path, imgshape, m=None, m_start=0, m_freq=1, color=False, ext='jpg', n_it=10,
#             k=1, mu=None, lmbda=None,
            k=1, mu=0.0001, lmbda=0.01,
            reg=1, tol=1e-06, iters=500):
    X = get_data_matrix(dataset_path, imgshape, m, m_start, m_freq, color, ext)
    tSVD = getMeanTime(bmSVD, (X, k), n_it)
    tRPCA = getMeanTime(bmRPCA, (X, mu, lmbda), n_it)
    tROSL = getMeanTime(bmROSL, (X, k, reg, tol, iters), n_it)
    return tSVD, tRPCA, tROSL


################################   Fitting data by affine subspace with SVD, RPCA, ROSL   ################################

def sample_mvn2D(mu1=0, mu2=0, sigma1=1, sigma2=1, rho=0, N=100):
    x0 = np.random.normal(size=N)
    y0 = np.random.normal(size=N)
    x = mu1 + sigma1*x0
    y = mu2 + sigma2*(rho*x0 + np.sqrt(1-rho**2)*y0)
    A = np.hstack((x.reshape(-1,1), y.reshape(-1,1)))
    return A

def visualize2D(A, color='blue'):
    A = A.reshape((-1,2))
    plt.scatter(A[:,0], A[:,1], color=color)
    lmax, lmin = np.max(A), np.min(A)
    plt.xlim(lmin,lmax)
    plt.ylim(lmin,lmax)

def PCA(A, out=False):
    N = A.shape[0]
    mean_point = np.mean(A, 0, keepdims=True)
    A0 = A - mean_point
    C = A0.T.dot(A0)
    if out:
        print('Covariance matrix:\n{}'.format(C))
    l, v = LA.eig(C)
    l = l.real
    ids = np.argsort(-l)
    lambdas, V = l[ids], v[:,ids]
    principal_components = list(V.T)
    if out:
        print('Eigenvalues & eigenvectors')
        for i, (l, v) in enumerate(list(zip(lambdas, principal_components))):
            print('l_{} = {:.2f}\tv_{} = {}'.format(i, l, i, v.flatten()))
    return principal_components

def PCA_fit(A):
    mean_point = np.mean(A, 0, keepdims=True)
    pc = PCA(A, out=False)
    v1 = pc[0].reshape(-1,1)
    m1, m2 = mean_point.flatten()
    slope = v1[1] / v1[0]
    intercept = m2 - m1 * slope
    return slope, intercept

def RPCA_fit(A, mu=None, lmbda=None, visualze=False):
    rpca = R_pca(A, mu=mu, lmbda=lmbda)
    L, E = rpca.fit(iter_print=np.nan)
    pc = PCA(rpca.L, out=False)
    v1 = pc[0].reshape(-1,1)
    m1, m2 = np.mean(L, 0)
    slope = v1[1] / v1[0]
    intercept = m2 - m1 * slope
    if visualze:
        visualize2D(L)
        visualize2D(E)
        plt.show()
    return slope, intercept

def ROSL_fit(A, k=1, reg=1, tol=1e-06, iters=500, visualze=False):
    rosl = ROSL(method='full', rank=k, reg=reg, tol=tol, iters=iters, verbose=False)
    D = rosl.fit_transform(A.astype(np.float64))  
    L = np.dot(D, rosl.components_)
    E = A - L
    pc = PCA(L, out=False)
    v1 = pc[0].reshape(-1,1)
    m1, m2 = np.mean(L, 0)
    slope = v1[1] / v1[0]
    intercept = m2 - m1 * slope
    if visualze:
        visualize2D(L, color="blue")
        visualize2D(E, color='red')
        plt.show()
    return slope, intercept

def errors(s1, intercept1, s2, intercept2, err='sqr'):
    if err == 'sqr':
        return float((s1 - s2)**2), float((intercept1 - intercept2)**2)
    elif err == 'abs':
        return float(np.abs(s1 - s2)), float(np.abs(intercept1 - intercept2))

def draw_func(f, xlmin=0, xlmax=1, N=1000, color='k'):
    xs = np.linspace(xlmin, xlmax, N)
    try:
        ys = f(xs)
    except:
        ys = [f(x) for x in xs]
    plt.plot(xs, ys, color)