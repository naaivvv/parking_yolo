import random
import math
import numpy as np
np.float_ = np.float64
import cv2
try:
    from straug.warp import Curve,Distort,Stretch
    from straug.geometry import Perspective,Rotate,Shrink
    from straug.pattern import Grid,VGrid,HGrid,RectGrid,EllipseGrid
    from straug.blur import GaussianBlur,DefocusBlur,MotionBlur,GlassBlur,ZoomBlur
    from straug.noise import GaussianNoise,ShotNoise,ImpulseNoise,SpeckleNoise
    from straug.weather import Fog,Snow,Frost,Rain,Shadow
    from straug.camera import Contrast,Brightness,JpegCompression,Pixelate
    from straug.process import Posterize,Solarize,Invert,Equalize,AutoContrast,Sharpness,Color
    STRAUG_AVAILABLE = True
except (ImportError, OSError):
    STRAUG_AVAILABLE = False

from PIL import Image

if STRAUG_AVAILABLE:
    augmentations = [Curve(),Distort(),Stretch(),
    Perspective(),Rotate(),Shrink(),
    GaussianBlur(),DefocusBlur(),MotionBlur(),GlassBlur(),ZoomBlur(),
    Contrast(),Brightness(),JpegCompression(),Pixelate(),
    Equalize(),AutoContrast(),Sharpness(),Color()]
else:
    augmentations = []



def motion_blur(img):
    kernel_s= np.random.randint(5,10,1)
    kernel_size = kernel_s[0]
    kernel_v = np.zeros((kernel_size, kernel_size))
    kernel_h = np.copy(kernel_v)
    kernel_v[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
    kernel_h[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
    kernel_v /= kernel_size
    kernel_h /= kernel_size 
    if random.choice([True,False]):
        img = cv2.filter2D(img, -1, kernel_v)
    if random.choice([True,False]):
        img = cv2.filter2D(img, -1, kernel_h)
    return img

def blur(img):
    rows,cols, _ = img.shape

    dst = cv2.GaussianBlur(img,(3,3),cv2.BORDER_DEFAULT)
    return dst

def jitter(img, jitter=0.1):
    rows, cols, _ = img.shape
    j_width = float(cols) * random.uniform(1 - jitter, 1 + jitter)
    j_height = float(rows) * random.uniform(1 - jitter, 1 + jitter)
    img = cv2.resize(img, (int(j_width), int(j_height)))
    return img

def rotate(img, angle=np.random.randint(5,15)):

    scale = random.uniform(0.9, 1.1)
    angle = random.uniform(-angle, angle)

    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    dst = img.copy()
    dst = cv2.warpAffine(img, M, (cols, rows), dst, cv2.INTER_LINEAR)

    return dst

def perspective(img):

    h, w, _ = img.shape
    per = random.uniform(0.05, 0.3)
    w_p = int(w * per)
    h_p = int(h * per)

    pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    pts2 = np.float32([[random.randint(0, w_p), random.randint(0, h_p)],
                       [random.randint(0, w_p), h - random.randint(0, h_p)],
                       [w - random.randint(0, w_p), random.randint(0, h_p)],
                       [w - random.randint(0, w_p), h - random.randint(0, h_p)]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (w, h))
    return img

def apply_geometric_skew(img):
    """
    Applies extreme geometric skewing using OpenCV.
    Horizontal: 45-65 degrees (left or right).
    Vertical: 10-20 degrees (below perspective).
    """
    h, w = img.shape[:2]
    
    # 1. Randomize angles based on user request
    # Horizontal angle: 10 to 30 degrees (reduced for readability)
    horiz_angle = random.uniform(10, 30) * random.choice([-1, 1])
    # Vertical perspective angle: 0 to 10 degrees (reduced for readability)
    vert_angle = random.uniform(0, 10)
    
    # 2. Define source points (original corners)
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # 3. Calculate destination points
    # Horizontal shear offset
    h_shift = h * math.tan(math.radians(horiz_angle))
    
    # Vertical perspective factor (narrowing top/widening bottom)
    v_p = (w * math.sin(math.radians(vert_angle))) / 4.0
    
    if h_shift > 0:
        # Skew right: shift top to the right
        pts2 = np.float32([
            [h_shift + v_p, v_p],       # tl
            [w + h_shift - v_p, v_p],   # tr
            [-v_p, h - v_p],            # bl
            [w + v_p, h - v_p]          # br
        ])
    else:
        # Skew left: shift top to the left
        pts2 = np.float32([
            [v_p, v_p],                 # tl
            [w - v_p, v_p],             # tr
            [-h_shift - v_p, h - v_p],  # bl
            [w - h_shift + v_p, h - v_p]# br
        ])

    # Determine dimensions to fit the warped plate
    min_x = np.min(pts2[:, 0])
    max_x = np.max(pts2[:, 0])
    min_y = np.min(pts2[:, 1])
    max_y = np.max(pts2[:, 1])
    
    new_w = int(max_x - min_x)
    new_h = int(max_y - min_y)
    
    # Offset pts2 to start from (0,0) in the new canvas
    pts2[:, 0] -= min_x
    pts2[:, 1] -= min_y
    
    # 4. Perform the transformation
    M = cv2.getPerspectiveTransform(pts1, pts2)
    skewed = cv2.warpPerspective(img, M, (new_w, new_h), borderValue=(0,0,0))
    
    return skewed

def crop_subimage(img, margin=3):
    ran_margin = random.randint(0, margin)
    rows, cols, _ = img.shape
    crop_h = rows - ran_margin
    crop_w = cols - ran_margin
    row_start = random.randint(0, ran_margin)
    cols_start = random.randint(0, ran_margin)
    sub_img = img[row_start:row_start + crop_h, cols_start:cols_start + crop_w]
    return sub_img

def hsv_space_variation(ori_img, scale):

    rows, cols, _ = ori_img.shape

    hsv_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2HSV)
    hsv_img = np.array(hsv_img, dtype=np.float32)
    img = hsv_img[:, :, 2]

    # gau noise
    noise_std = random.randint(5, 20)
    noise = np.random.normal(0, noise_std, (rows, cols))

    # brightness scale
    img = img * scale
    img = np.clip(img, 0, 255)
    img = np.add(img, noise)

    # random hue variation
    hsv_img[:, :, 0] += random.randint(-5, 5)

    # random sat variation
    hsv_img[:, :, 1] += random.randint(-30, 30)

    hsv_img[:, :, 2] = img
    hsv_img = np.clip(hsv_img, 0, 255)
    hsv_img = np.array(hsv_img, dtype=np.uint8)
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    return rgb_img

def data_augmentation(img):
    img = crop_subimage(img)
    
    # Apply mild geometric skewing with 10% probability (reduced from 20% for faster convergence)
    if random.random() < 0.10:
        img = apply_geometric_skew(img)
        
    bright_scale = random.uniform(0.8, 1.2)
    img_out = hsv_space_variation(img, scale=bright_scale)
    im = Image.fromarray(img_out)

    if augmentations:
        im = random.choice(augmentations)(im, mag=random.randint(0, 2))  # reduced from 3 for faster convergence
        im_out = np.array(im)
    else:
        im_out = np.array(im)

    return im_out