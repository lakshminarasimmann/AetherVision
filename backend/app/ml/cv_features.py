# pyrefly: ignore [missing-import]
import cv2
import numpy as np
import base64

def generate_heatmap(gray_img, color_img):
    """
    Generates a sharpness/quality heatmap overlay.
    Divides the image into patches, computes local Laplacian variance,
    normalizes it, applies a colormap, and blends it with the original image.
    Returns a base64 encoded string of the heatmap image.
    """
    h, w = gray_img.shape
    patch_size = 32
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Compute local variance of Laplacian for each patch
    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            patch = gray_img[y:min(y+patch_size, h), x:min(x+patch_size, w)]
            laplacian = cv2.Laplacian(patch, cv2.CV_64F)
            var = laplacian.var()
            heatmap[y:min(y+patch_size, h), x:min(x+patch_size, w)] = var
            
    # Normalize heatmap to 0-255
    heatmap_norm = cv2.normalize(heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    heatmap_norm = np.uint8(heatmap_norm)
    
    # Invert so lower variance (blur) is red (heat), and high variance is blue (cold)
    # Using JET colormap: 0 = blue, 255 = red. 
    # Blur has low variance (low number). We want blur to be red.
    # So we invert the normalized heatmap.
    heatmap_inv = 255 - heatmap_norm
    
    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_inv, cv2.COLORMAP_JET)
    
    # Blend with original image
    blended = cv2.addWeighted(color_img, 0.6, heatmap_color, 0.4, 0)
    
    # Encode to base64
    _, buffer = cv2.imencode('.jpg', blended)
    base64_img = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_img}"

def extract_features(img_path):
    """
    Extracts engineered features from an image for quality assessment.
    Returns a dictionary of features and the original image.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image at {img_path} could not be read.")
    
    # Resize for faster consistent processing
    img = cv2.resize(img, (512, 512))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Blur / Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()
    
    # 2. Brightness / Exposure (Mean and std dev of Value channel)
    v_channel = hsv[:,:,2]
    brightness = np.mean(v_channel)
    contrast = np.std(v_channel)
    
    # 3. Noise (Estimated using local variance or edge preserving filter diff)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, blurred)
    noise_level = np.mean(diff)
    
    # 4. Saturation
    s_channel = hsv[:,:,1]
    saturation = np.mean(s_channel)

    # 5. Corruption (Percentage of absolute black or absolute white pixels)
    hist, _ = np.histogram(gray.flatten(), 256, [0,256])
    clipping_fraction = (hist[0] + hist[-1]) / (512*512)
    
    # Normalize features roughly based on expected ranges
    features = {
        "sharpness": float(sharpness) / 1000.0,
        "brightness": float(brightness) / 255.0,
        "contrast": float(contrast) / 128.0,
        "noise_level": float(noise_level) / 50.0,
        "saturation": float(saturation) / 255.0,
        "clipping_fraction": float(clipping_fraction)
    }
    
    feature_array = np.array([
        features["sharpness"],
        features["brightness"],
        features["contrast"],
        features["noise_level"],
        features["saturation"],
        features["clipping_fraction"]
    ], dtype=np.float32)
    
    # Generate Heatmap
    heatmap_base64 = generate_heatmap(gray, img)
    
    return feature_array, features, heatmap_base64
