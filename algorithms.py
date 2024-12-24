from my_modules.helpers import *
from PIL import Image
from io import BytesIO

def rgb2Gray(rgb):
    h, w, l = rgb.shape
    gray = np.zeros((h, w), dtype=float)
    for i in range(h):
        for j in range(w):
            gray[i][j] = sum(rgb[i, j])/l
    gray = np.uint8(255 * gray / np.max(gray))
    return gray
    
def gray2Binary(gray, threshold=128):
    h, w= gray.shape
    binary = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            binary[i][j] = 255 if gray[i][j] >= threshold else 0
    return binary

def rgb2Binary(img):
    return gray2Binary(rgb2Gray(img))

def brightnessProcessing(img, op, value):
    M, N = img.shape
    res = np.zeros((M, N), dtype=int)
    for i in range(M):
        for j in range(N):
            match op:
                case '+':res[i][j] = img[i][j] + value
                case '-':res[i][j] = img[i][j] - value
                case '*':res[i][j] = img[i][j] * value
                case '/':
                    if value == 0:
                        raise ZeroDivisionError('Division by zero!')
                    res[i][j] = img[i][j] / value
                case _:
                    raise ValueError('Invalid operation!')
    return np.clip(res, 0, 255)


def logTransform(img, constant=1):
    h, w = img.shape
    res = np.zeros((h,w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            res[i, j] = constant * log(img[i, j]+1)
            
    res = np.uint8(255 * res / np.max(res))
    return res

def logInverseTransform(img):
    h, w = img.shape
    img = np.array(img/255, dtype=float)
    res = np.zeros((h,w), dtype=float)
    for i in range(h):
        for j in range(w):
            res[i, j] = e**img[i, j] - 1

    res = np.uint8(255 * res / np.max(res))
    return res

def negative(img):
    M, N = img.shape
    res = np.zeros((M, N), dtype=np.uint8)
    for i in range(M):
        for j in range(N):
            res[i, j] = 255 - img[i, j]
    return res

def gamma(img, gamma=0.2, constant=1):
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.float32)
    for i in range(M):
        for j in range(N):
            res[i][j] = constant * img[i][j]**gamma
    
    res = np.uint8(255 * res / np.max(res))

    return res


def histogram(img):
    h, w = img.shape
    hist = {}
    bins = np.arange(0, 256, 1)
    
    for i in range(h):
        for j in range(w):
            hist[img[i][j]] = (hist[img[i][j]] if img[i][j] in hist.keys() else 0) + 1
    
    plt.figure(figsize=(16, 8))
    plt.hist(hist.keys(), weights=hist.values(), bins=bins, edgecolor='black', color='r')
    plt.xticks(bins[::10])
    plt.xlabel('Gray level')
    plt.ylabel('Count')
    plt.title('Histogram')
    
    fig = plt.gcf()
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    plt.close()
    
    return buf

def contrastStretching(img):
    old_min, old_max, new_min, new_max = img.min(), img.max(), 0, 255
    h, w = img.shape
    res = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            res[i][j] = get_value(round((img[i][j] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min))
    return res

def histogramEqualization(img):
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 256])    
    cdf, sum = [], 0
    for f in hist:
        sum += f
        cdf.append(sum)
    res = img.copy()
    cdf_min = cdf[0]
    cdf = np.clip((cdf - cdf_min) / (img.size - cdf_min) * 255, 0, 255).astype(int)
    res = cdf[img]
    return res

def sop(region, kernel):
    res = 0
    for i in range(region.shape[0]):
        for j in range(region.shape[1]):
            res += region[i, j] * kernel[i, j]  
    return res

def meanFilterBlurring(img, n=3):
    p = n // 2
    result = np.array(img, dtype=float)
    padded_img = np.pad(img, p, 'edge')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+n, j:j+n]
            result[i, j] = sum(region.flatten()) / n ** 2
    return np.uint8(result)

def weightFilterBlurring(img):
    kernel = 1/16 * np.array([
        [1, 4, 1],
        [2, 4, 2],
        [1, 4, 1],
    ])
    n = kernel.shape[0]
    p = n // 2
    result = np.array(img, dtype=float)
    padded_img = np.pad(img, p, 'edge')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+n, j:j+n]
            result[i, j] = get_value(sop(region, kernel))
    return np.uint8(result)

def getEdgeDetectionKernel(direction='h'):
    match direction:
        case 'h':
            kernel = np.array([[-1, -2, -1],
                               [0,  0,  0],
                               [1,  2,  1]])
        case 'v':
            kernel = np.array([[-1,  0,  1],
                               [-2,  0,  2],
                               [-1,  0,  1]])
        case 'ld':
            kernel = np.array([[0,  1,  2],
                               [-1, 0,  1],
                               [-2, -1, 0]])
        case 'rd':
            kernel = np.array([[2,  1,  0],
                               [1,  0, -1],
                               [0, -1, -2]])
        case _:
            raise ValueError("Invalid direction. Choose from 'h', 'v', 'rd', 'ld'.")
    return kernel

def edgeDetection(img):
    
    sobel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    
    n = sobel_x.shape[0]
    p = n // 2
    result = np.array(img, dtype=float)
    padded_img = np.pad(img, p, 'edge')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+n, j:j+n]
            dx = np.sum(region * sobel_x)
            dy = np.sum(region * sobel_y)
            result[i, j] = get_value(sqrt(dx ** 2 + dy ** 2))
    return np.uint8(result)

def directedEdgeDetection(img, direction='h'):
    kernel = getEdgeDetectionKernel(direction)
    n = kernel.shape[0]
    p = n // 2
    result = np.array(img, dtype=float)
    padded_img = np.pad(img, p, 'edge')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+n, j:j+n]
            result[i, j] = get_value(np.sum(region * kernel))
    return np.uint8(result)

def getSharpeningKernel(direction='h'):
    match direction:
        case 'h':
            kernel = np.array([[0,  0,  0],
                               [1,  1, -1],
                               [0,  0,  0]])
        case 'v':
            kernel = np.array([[0,  1,  0],
                               [0,  1,  0],
                               [0, -1,  0]])
        case 'ld':
            kernel = np.array([[1,  0,  0],
                               [0,  1,  0],
                               [0,  0, -1]])
        case 'rd':
            kernel = np.array([[0,  0,  1],
                               [0,  1,  0],
                               [-1, 0,  0]])
        case _:
            raise ValueError("Invalid direction. Choose from 'h', 'v', 'rd', 'ld'.")
    return kernel

def sharpening(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    n = kernel.shape[0]
    p = n // 2
    result = np.array(img, dtype=float)
    padded_img = np.pad(img, p, 'edge')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+n, j:j+n]
            result[i, j] = get_value(np.sum(region * kernel))
    return np.uint8(result)

def directedSharpening(img, direction='h'):
    kernel = getSharpeningKernel(direction)
    n = kernel.shape[0]
    p = n // 2
    result = np.array(img, dtype=float)
    padded_img = np.pad(img, p, 'edge')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+n, j:j+n]
            result[i, j] = get_value(np.sum(region * kernel))
    return np.uint8(result)

def dft(img):
    M, N = img.shape
    fourier = np.zeros((M, N), dtype=complex)
    
    fourier = np.fft.fft2(img)
    f_shifted = np.fft.fftshift(fourier)
    magnitude = np.abs(f_shifted)
    magnitude = np.log1p(magnitude)
    magnitude = np.uint8(magnitude / np.max(magnitude) * 255)
    return magnitude, fourier

    for u in range(M):
        for v in range(N):
            for x in range(M):
                for y in range(N):
                    fourier[u, v] += img[x, y] * e ** (-2j * pi * (u*x/M + v*y/N))
    return fourier

def inverseDft(fourier):
    M, N = fourier.shape[0], fourier.shape[1]
    inverse = np.zeros((M, N), dtype=complex)
    
    img = np.fft.ifft2(fourier)
    img = np.real(img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img
    
    for u in range(M):
        for v in range(N):
            for x in range(M):
                for y in range(N):
                    inverse[u, v] += fourier[x, y] * e ** (2j * pi * (u*x/M + v*y/N))
    inverse /= (M * N)
    return inverse

def idealFilter(img, d0=3, mode='l'):
    M, N = img.shape
    fourier = dft(img)[1]
    shifted = np.fft.fftshift(fourier)
    filter = np.zeros((M, N), dtype=float)
    
    for u in range(M):
        for v in range(M):
            d = np.sqrt((u-M//2)**2 + (v-N//2)**2)
            if(mode == 'l'):
                filter[u, v] = 1 if d <= d0 else 0
            else:
                filter[u, v] = 0 if d <= d0 else 1
                
    filtered = shifted * filter
    unshifted = np.fft.ifftshift(filtered)
    result = inverseDft(unshifted)
    return np.clip(result.real, 0, 255).astype(int)

def butterworthFilter(img, n=1, d0=3, mode='l'):
    M, N = img.shape
    fourier = dft(img)[1]
    shifted = np.fft.fftshift(fourier)
    filter = np.zeros((M, N), dtype=float)
    
    for u in range(M):
        for v in range(M):
            d = np.sqrt((u-M//2)**2 + (v-N//2)**2)
            d = max(d, 1e-10)
            if(mode == 'l'):
                filter[u, v] = 1/(1+(d/d0)**(2*n))
            else:
                filter[u, v] = 1/(1+(d0/d)**(2*n))
    
    filtered = shifted * filter
    unshifted = np.fft.ifftshift(filtered)
    result = inverseDft(unshifted)
    return np.clip(result.real, 0, 255).astype(int)

def gaussianFilter(img, d0=3, mode='l'):
    M, N = img.shape
    fourier = dft(img)[1]
    shifted = np.fft.fftshift(fourier)
    filter = np.zeros((M, N), dtype=float)
    
    for u in range(M):
        for v in range(M):
            d = np.sqrt((u-M//2)**2 + (v-N//2)**2)
            d = max(d, 1e-10)
            if(mode == 'l'):
                filter[u, v] = e ** (-d**2 / (2*d0**2))
            else:
                filter[u, v] = 1 - e ** (-d**2 / (2*d0**2))
    
    filtered = shifted * filter
    unshifted = np.fft.ifftshift(filtered)
    result = inverseDft(unshifted)
    return np.clip(result.real, 0, 255).astype(int)

def rankOrderFilter(img, n=3, mode='min'):
    p = n // 2
    result = np.array(img, dtype=float)
    padded_img = np.pad(img, p, 'edge')
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i:i+n, j:j+n].astype(float).flatten()
            region.sort()
            match mode:
                case 'min':
                    result[i, j] = region[0]
                case 'max':
                    result[i, j] = region[-1]
                case 'median':
                    result[i, j] = region[len(region)//2]
                case 'midpoint':
                    result[i, j] = (region[0]+region[-1])/2
                case _:
                    raise ValueError('Invalid Mode, min, max, median & midoint are only valid!')
    return np.clip(result, 0, 255).astype(int)