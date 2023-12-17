import cv2
import numpy as np
import scipy.stats
from sklearn.cluster import KMeans
from skimage.feature import greycomatrix, greycoprops, local_binary_pattern, hog
# from skimage.filters import gabor_kernel
from skimage.filters.rank import entropy
from skimage.morphology import disk
import pywt
from sklearn.decomposition import PCA

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
                    cells_per_block=(1, 1), visualize = True)
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

# mid-level: image complexity features
def Entropy(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_entropy = entropy(gray, disk(5)) # local entropy
    return image_entropy

def Fractal_Dimension(img_path): # ??????????
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # data = ps.metrics.boxcount(gray)
    # FD_value = np.concatenate((data.size, data.count, data.slope), axis = -1)

    # finding all the non-zero pixels
    pixels = []
    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            if gray[i, j]>0:
                pixels.append((i, j))
    
    Lx = gray.shape[1]
    Ly = gray.shape[0]
    pixels = np.array(pixels)
    
    # computing the fractal dimension
    # considering only scales in a logarithmic list
    scales = np.logspace(0.01, 1, num = 10, endpoint = False, base = 2)
    Ns = []
    # looping over several scales
    for scale in scales:
        # computing the histogram
        H, edges = np.histogramdd(pixels, bins = (np.arange(0, Lx, scale), np.arange(0, Ly, scale)))
        Ns.append(np.sum(H > 0))
    
    # linear fit, polynomial of degree 1
    FD_value = np.polyfit(np.log(scales), np.log(Ns), 1)
    # FD_value = np.log(Ns)
    return FD_value

def Edge_Density(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(gray, 100, 200)
    density_value = np.sum(img_canny)/(gray.shape[0]*gray.shape[1]) # sub region density?
    return density_value

def Spatial_Information(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sobel
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    si_value = cv2.sqrt(cv2.pow(grad_x/255., 2)*255. + cv2.pow(grad_y/255., 2)*255.)
    return si_value # different to sobel?


# mid-level: image Compression/Transformations features

def Discrete_Fourier_Transform(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    f = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    f_shift = np.fft.fftshift(f)
    f_complex = f_shift[:,:,0] + 1j*f_shift[:,:,1]
    f_abs = np.abs(f_complex) + 1 # lie between 1 and 1e6
    f_bounded = 20 * np.log(f_abs)
    f_img = cv2.normalize(f_bounded, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return f_img

def Wavelet_Transform(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    gray = np.array(gray)
    LLY, (LHY, HLY, HHY) = pywt.dwt2(img, 'haar') # Haar Discrete Wavelet Transform（HDWT）
    wavelet_trans = np.concatenate((LLY, LHY, HLY, HHY))
    return wavelet_trans

def Histogram_Equalization(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    dst = cv2.equalizeHist(gray)
    return dst

def Discrete_Cosine_Transform(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    imf = np.float32(gray)
    dct = cv2.dct(imf)
    dct = np.uint8(dct*255.0)
    return dct

def SVD(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    u, s, v = np.linalg.svd(gray, full_matrices = False)
    return s

def PCA_transform(img_path):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    pca = PCA(n_components = int(gray.shape[0]/2))
    pca_values = pca.fit_transform(gray)
    return pca_values


