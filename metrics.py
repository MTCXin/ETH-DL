import cv2
import numpy as np
import scipy.stats
from sklearn.cluster import KMeans
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern
from skimage.filters import gabor_kernel

# low-level: color features
def Color_Histogram(img_path):
    img = cv2.imread(img_path)
    calhist_r = cv2.calcHist(img, channels = [0], mask = None, histSize = [256], ranges = [0, 256])
    calhist_g = cv2.calcHist(img, channels = [1], mask = None, histSize = [256], ranges = [0, 256])
    calhist_b = cv2.calcHist(img, channels = [2], mask = None, histSize = [256], ranges = [0, 256])
    calhist_ = np.concatenate((calhist_r, calhist_g, calhist_b), axis = -1)
    return calhist_

def Color_Moment(img_path):
    img = cv2.imread(img_path)
    Mean_, Var_, Skew_ = [], [], []
    for i in range(3):
        mean = np.mean(img[:, :, i])
        var = np.var(img[:, :, i])
        skew = scipy.stats.skew(img[:, :, i], axis = None)
        Mean_ = np.append(Mean_, mean)
        Var_ = np.append(Var_, var)
        Skew_ = np.append(Skew_, skew)
    colormo = np.concatenate((Mean_.reshape(-1, 1), Var_.reshape(-1, 1), Skew_.reshape(-1, 1)), axis = -1).T
    return colormo

def Dominant_Color_Descriptor(img_path):
    img = cv2.imread(img_path)
    img_ = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    n_colors = 10
    model = KMeans(n_clusters = n_colors, random_state = 0).fit(img_)
    DCD = np.uint8(model.cluster_centers_)
    return DCD

# low-level: texture features

# GLCM
# def calc_glcm_all_agls(img, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
#     glcm = greycomatrix(img, 
#                         distances=dists, 
#                         angles=agls, 
#                         levels=lvl,
#                         symmetric=sym, 
#                         normed=norm)
#     feature = []
#     glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
#     for item in glcm_props:
#         feature.append(item)
#     return feature

def Gray_Level_Cooccurrence_Matrix(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

    glcm = greycomatrix(gray, 
                        distances = [5], 
                        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels = 256,
                        symmetric = True, 
                        normed = True) 
    feature = []
    glcm_props = [propery for name in properties for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)
    # glcm_all_agls = calc_glcm_all_agls(gray, props=properties)
    GLCM = np.array(feature)
    return GLCM

def Local_Binary_Patterns(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    METHOD = 'uniform'
    radius = 3
    n_points = 8*radius
    LBP = local_binary_pattern(gray, n_points, radius, METHOD)
    return LBP

def Gabor_Filters(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return Gabor