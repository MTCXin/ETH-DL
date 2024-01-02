import cv2
import numpy as np
import scipy.stats
from sklearn.cluster import KMeans
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog
# from skimage.filters import gabor_kernel
from skimage.filters.rank import entropy
from skimage.morphology import disk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# low-level: color features

''' Not useful
def Color_Histogram(img_path): #1
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    calhist_r = cv2.calcHist(img, channels = [0], mask = None, histSize = [256], ranges = [0, 256])
    calhist_g = cv2.calcHist(img, channels = [1], mask = None, histSize = [256], ranges = [0, 256])
    calhist_b = cv2.calcHist(img, channels = [2], mask = None, histSize = [256], ranges = [0, 256])
    calhist_ = np.concatenate((calhist_r, calhist_g, calhist_b), axis = -1)
    return calhist_
'''

def Color_Statistic(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    Statistics = []
    for i in range(3):
        mean = np.mean(img[:, :, i])
        Statistics.append(mean)
        var = np.var(img[:, :, i])
        Statistics.append(var)
        skew = scipy.stats.skew(img[:, :, i], axis = None)
        Statistics.append(skew)
        quantile5 = np.quantile(img[:, :, i], 0.05)
        Statistics.append(quantile5)
        median = np.quantile(img[:, :, i], 0.5)
        Statistics.append(median)
        quantile95 = np.quantile(img[:, :, i], 0.95)
        Statistics.append(quantile95)
        std = np.std(img[:, :, i])
        z = (img[:, :, i] - mean) / std
        outliers = np.sum(np.abs(z) > 2)
        Statistics.append(outliers)
    Statistics = np.array(Statistics).reshape(3,-1).T
    return Statistics

def Dominant_Color_Descriptor(img_path): 
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img_ = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    n_colors = 10
    model = KMeans(n_clusters = n_colors, random_state = 0).fit(img_)
    DCD = np.uint8(model.cluster_centers_)
    return DCD

# low-level: texture features

def Gray_Level_Cooccurrence_Matrix(img_path):
    img = cv2.imread(img_path)	
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']

    glcm = graycomatrix(gray, 
                        distances = [5], 
                        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels = 256,
                        symmetric = True, 
                        normed = True) 
    feature = []
    glcm_props = [propery for name in properties for propery in graycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)
    GLCM = np.array(feature).reshape(6,4)
    return GLCM

def Local_Binary_Patterns(img_path): ###滤波
    img = cv2.imread(img_path)	
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    METHOD = 'uniform'
    params = [(1, 8), (2, 16), (3, 24)]
    Statistics = []
    for para in params:
        lbp = local_binary_pattern(gray, para[0], para[1], METHOD)
        Statistics.append([np.mean(lbp), np.var(lbp), scipy.stats.skew(lbp, axis=None), np.quantile(lbp, 0.05), np.quantile(lbp, 0.95)])
    # dim_ = 20
    # LBP_reduced = np.zeros((dim_, dim_))
    # for i in range(dim_):
    #     for j in range(dim_):
    #         if i == dim_ - 1 and j == dim_ -1:
    #             LBP_reduced[i][j] = np.mean(LBP[int(LBP.shape[0]/dim_)*i:, int(LBP.shape[1]/dim_)*j:], axis = None)
    #         elif i == dim_ - 1:
    #             LBP_reduced[i][j] = np.mean(LBP[int(LBP.shape[0]/dim_)*i:, int(LBP.shape[1]/dim_)*j:int(LBP.shape[1]/dim_)*(j+1)], axis = None)
    #         elif j == dim_ - 1:
    #             LBP_reduced[i][j] = np.mean(LBP[int(LBP.shape[0]/dim_)*i:int(LBP.shape[0]/dim_)*(i+1), int(LBP.shape[1]/dim_)*j:], axis = None)
    #         else: 
    #             LBP_reduced[i][j] = np.mean(LBP[int(LBP.shape[0]/dim_)*i:int(LBP.shape[0]/dim_)*(i+1), int(LBP.shape[1]/dim_)*j:int(LBP.shape[1]/dim_)*(j+1)], axis = None)
    return np.array(Statistics)

def Gabor_Filters(img_path):
    img = cv2.imread(img_path)	
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
    # ksize - size of gabor filter (n, n)
    # sigma - standard deviation of the gaussian function
    # theta - orientation of the normal to the parallel stripes
    # lambda - wavelength of the sunusoidal factor
    # gamma - spatial aspect ratio
    # psi - phase offset
    # ktype - type and range of values that each pixel in the gabor kernel can hold

    lam = [2, 5]
    Statistics = []
    for l in lam:
        g_kernel = cv2.getGaborKernel((9, 9), 1.0, np.pi/4, l, 0.5, 0, ktype=cv2.CV_32F) # ?
        GF = cv2.filter2D(gray, -1, g_kernel)
        Statistics.append([np.mean(GF), np.var(GF), scipy.stats.skew(GF, axis=None), np.quantile(GF, 0.05), np.quantile(GF, 0.95)])
        
    # dim_ = 20
    # GF_reduced = np.zeros((dim_, dim_))
    # for i in range(dim_):
    #     for j in range(dim_):
    #         if i == dim_ - 1 and j == dim_ -1:
    #             GF_reduced[i][j] = np.mean(GF[int(GF.shape[0]/dim_)*i:, int(GF.shape[1]/dim_)*j:], axis = None)
    #         elif i == dim_ - 1:
    #             GF_reduced[i][j] = np.mean(GF[int(GF.shape[0]/dim_)*i:, int(GF.shape[1]/dim_)*j:int(GF.shape[1]/dim_)*(j+1)], axis = None)
    #         elif j == dim_ - 1:
    #             GF_reduced[i][j] = np.mean(GF[int(GF.shape[0]/dim_)*i:int(GF.shape[0]/dim_)*(i+1), int(GF.shape[1]/dim_)*j:], axis = None)
    #         else: 
    #             GF_reduced[i][j] = np.mean(GF[int(GF.shape[0]/dim_)*i:int(GF.shape[0]/dim_)*(i+1), int(GF.shape[1]/dim_)*j:int(GF.shape[1]/dim_)*(j+1)], axis = None)
    return np.array(Statistics)

'''BUG
def Histogram_of_Oriented_Gradients(img_path):  ##梯度方向数据检测
    img = cv2.imread(img_path)	
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fd = hog(gray, visualize=True)

    fd = np.array(fd)
    print(fd.shape)
    Statistics = []
    Statistics.append(np.mean(fd), np.var(fd), scipy.stats.skew(fd), np.quantile(fd, 0.05), np.quantile(fd, 0.95))
    # dim_ = 20
    # hog_reduced = np.zeros((dim_, dim_))
    # for i in range(dim_):
    #     for j in range(dim_):
    #         if i == dim_ - 1 and j == dim_ -1:
    #             hog_reduced[i][j] = np.mean(hog_[int(hog_.shape[0]/dim_)*i:, int(hog_.shape[1]/dim_)*j:], axis = None)
    #         elif i == dim_ - 1:
    #             hog_reduced[i][j] = np.mean(hog_[int(hog_.shape[0]/dim_)*i:, int(hog_.shape[1]/dim_)*j:int(hog_.shape[1]/dim_)*(j+1)], axis = None)
    #         elif j == dim_ - 1:
    #             hog_reduced[i][j] = np.mean(hog_[int(hog_.shape[0]/dim_)*i:int(hog_.shape[0]/dim_)*(i+1), int(hog_.shape[1]/dim_)*j:], axis = None)
    #         else: 
    #             hog_reduced[i][j] = np.mean(hog_[int(hog_.shape[0]/dim_)*i:int(hog_.shape[0]/dim_)*(i+1), int(hog_.shape[1]/dim_)*j:int(hog_.shape[1]/dim_)*(j+1)], axis = None)
    return np.array(Statistics)
'''
# low-level: edge-detection features

def Sobel(img_path):
    Statistics = []
    img = cv2.imread(img_path)	
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    sobel_grad = cv2.magnitude(grad_x, grad_y)
    
    sobel_mean = np.mean(sobel_grad)
    Statistics.append(sobel_mean)
    sobel_var = np.var(sobel_grad)
    Statistics.append(sobel_var)
    quantile95 = np.quantile(sobel_grad, 0.95)
    Statistics.append(quantile95)
    maximum = np.max(sobel_grad)
    Statistics.append(maximum)
    
    Statistics = np.array(Statistics)
    return Statistics

''' Not useful
def Prewitt(img_path):
    img = cv2.imread(img_path)	
    img = cv2.resize(img, (224, 224))
    src = cv2.GaussianBlur(img, (19, 19), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    kernelx = np.array([[1,1,1], [0,0,0], [-1,-1,-1]])
    kernely = np.array([[-1,0,1], [-1,0,1], [-1,0,1]])
    img_prewittx = cv2.filter2D(gray, -1, kernelx)
    img_prewitty = cv2.filter2D(gray, -1, kernely)
    img_prewitt = img_prewittx + img_prewitty
    return img_prewitt
'''

def Canny(img_path):
    img = cv2.imread(img_path)	
    img = cv2.resize(img, (224, 224))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(gray, 100, 200)
    edge_percentage = np.sum(img_canny==255) / (img_canny.shape[0]*img_canny.shape[1])
    
    GBimg = cv2.GaussianBlur(img, (5, 5), 0)
    GBgray = cv2.cvtColor(GBimg, cv2.COLOR_BGR2GRAY)
    GB_img_canny = cv2.Canny(GBgray, 100, 200)
    GB_edge_percentage = np.sum(GB_img_canny==255) / (GB_img_canny.shape[0]*GB_img_canny.shape[1])
    return np.array([edge_percentage, GB_edge_percentage])

''' Not useful
def Laplacian_of_Gaussian_Filter(img_path):
    img = cv2.imread(img_path)	
    img = cv2.resize(img, (224, 224))
    src = cv2.GaussianBlur(img, (19, 19), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img_log = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    return img_log
'''

# mid-level: image complexity features
def Entropy(img_path):
    img = cv2.imread(img_path)	
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    structure = [disk(3), disk(5), disk(10)]
    Statistics = []
    for struc in structure:
        ent = entropy(gray, struc) # local entropy
        Statistics.append([np.mean(ent), np.var(ent), scipy.stats.skew(ent, axis=None), np.quantile(ent, 0.05), np.quantile(ent, 0.95)])
    # dim_ = 20
    # entropy_reduced = np.zeros((dim_, dim_))
    # for i in range(dim_):
    #     for j in range(dim_):
    #         if i == dim_ - 1 and j == dim_ -1:
    #             entropy_reduced[i][j] = np.mean(entropy_[int(entropy_.shape[0]/dim_)*i:, int(entropy_.shape[1]/dim_)*j:], axis = None)
    #         elif i == dim_ - 1:
    #             entropy_reduced[i][j] = np.mean(entropy_[int(entropy_.shape[0]/dim_)*i:, int(entropy_.shape[1]/dim_)*j:int(entropy_.shape[1]/dim_)*(j+1)], axis = None)
    #         elif j == dim_ - 1:
    #             entropy_reduced[i][j] = np.mean(entropy_[int(entropy_.shape[0]/dim_)*i:int(entropy_.shape[0]/dim_)*(i+1), int(entropy_.shape[1]/dim_)*j:], axis = None)
    #         else: 
    #             entropy_reduced[i][j] = np.mean(entropy_[int(entropy_.shape[0]/dim_)*i:int(entropy_.shape[0]/dim_)*(i+1), int(entropy_.shape[1]/dim_)*j:int(entropy_.shape[1]/dim_)*(j+1)], axis = None)
    return np.array(Statistics)

def Fractal_Dimension(img_path): # ??????????
    img = cv2.imread(img_path)	
    img = cv2.resize(img, (224, 224))
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

''' Same as Canny
def Edge_Density(img_path):
    img = cv2.imread(img_path)	
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_canny = cv2.Canny(gray, 100, 200)
    density_value = np.sum(img_canny)/(gray.shape[0]*gray.shape[1]) # sub region density?
    return density_value
'''

''' Same as Sobel
def Spatial_Information(img_path):
    img = cv2.imread(img_path)	
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # sobel
    ddepth = cv2.CV_16S
    scale = 1
    delta = 0
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize = 7, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize = 7, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    si_value = cv2.sqrt(cv2.pow(grad_x/255., 2)*255. + cv2.pow(grad_y/255., 2)*255.)
    return si_value # different to sobel?
'''

# mid-level: image Compression/Transformations features

'''
def Histogram_Equalization(img_path,levels=16):
    img = cv2.imread(img_path)	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    hist = cv2.calcHist([gray],[0],None,[levels], [0,255])
    return hist
'''

def Discrete_Cosine_Transform_max(img_path,n=16):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    imf = np.float32(gray)
    dct = cv2.dct(imf)
    # dct = np.uint8(dct*255.0)
    output_height = dct.shape[0] // n
    output_width = dct.shape[1] // n
    pooled_dct = np.zeros((output_height, output_width), dtype=dct.dtype)

    for i in range(output_height):
        for j in range(output_width):
            block = dct[i*n:(i+1)*n, j*n:(j+1)*n]
            pooled_dct[i, j] = np.max(block)

    return pooled_dct

def Discrete_Cosine_Transform_avg(img_path,n=16):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    imf = np.float32(gray)
    dct = cv2.dct(imf)
    # dct = np.uint8(dct*255.0)
    output_height = dct.shape[0] // n
    output_width = dct.shape[1] // n
    pooled_dct = np.zeros((output_height, output_width), dtype=dct.dtype)

    for i in range(output_height):
        for j in range(output_width):
            block = dct[i*n:(i+1)*n, j*n:(j+1)*n]
            pooled_dct[i, j] = np.mean(block)

    return pooled_dct

def SVD(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))	
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    
    U, S, V = np.linalg.svd(gray, full_matrices=False)
    
    k = 20
    
    svd = [S[i]/np.sum(S) for i in range(k)]
    
    return np.array(svd)

def PCA_transform(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    
    n_components = 50
    pca = PCA(n_components=n_components)
    
    pca.fit(gray)
    

    ratio = pca.explained_variance_ratio_
    
    sumratio = np.array([np.sum(ratio[:i]) for i in range(1, n_components+1)])
    
    t_component = np.array([(np.argmax(sumratio > value)+1 if sumratio[n_components-1]>value else n_components)
                            for value in [0.5, 0.7, 0.8, 0.9, 0.95]])

    return np.concatenate((sumratio[:10], t_component))