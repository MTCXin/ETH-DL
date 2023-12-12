import cv2
import numpy as np
import scipy.stats
from sklearn.cluster import KMeans
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern, hog
# from skimage.filters import gabor_kernel

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

# ??????????????
def Gabor_Filters(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
    # ksize - size of gabor filter (n, n)
    # sigma - standard deviation of the gaussian function
    # theta - orientation of the normal to the parallel stripes
    # lambda - wavelength of the sunusoidal factor
    # gamma - spatial aspect ratio
    # psi - phase offset
    # ktype - type and range of values that each pixel in the gabor kernel can hold

    g_kernel = cv2.getGaborKernel((4, 4), 1.0, np.pi/4, 2.0, 0.5, 0, ktype=cv2.CV_32F) # ?
    filtered_img = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
    return filtered_img

def Histogram_of_Oriented_Gradients(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, hog_image = hog(gray, orientations = 8, pixels_per_cell = (4, 4),
                    cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return hog_image

# low-level: edge-detection features

def Sobel(img_path):
    img = cv2.imread(img_path)	
    src = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    sobel_grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return sobel_grad

def Prewitt(img_path):
    img = cv2.imread(img_path)	
    src = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
    kernely = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
    img_prewittx = cv2.filter2D(gray, -1, kernelx)
    img_prewitty = cv2.filter2D(gray, -1, kernely)
    img_prewitt = img_prewittx + img_prewitty
    return img_prewitt

def Canny(img_path):
    img = cv2.imread(img_path)	
    src = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(gray, 100, 200)
    return img_canny

def Laplacian_of_Gaussian_Filter(img_path):
    img = cv2.imread(img_path)	
    src = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img_log = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    return img_log