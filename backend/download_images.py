import os
import requests
import cv2
import numpy as np

def generate_procedural_image(index, size=(512, 512)):
    """Generates a synthetic procedural image with shapes and gradients if network is unavailable."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Background gradient
    for y in range(h):
        r = int((y / h) * 200 + (index * 15) % 55)
        g = int(((h - y) / h) * 180 + (index * 25) % 75)
        b = int(((y * index) % h) / h * 255)
        img[y, :] = [b % 256, g % 256, r % 256]
        
    # Draw geometric shapes to provide high frequency edge details
    cv2.circle(img, (w // 2, h // 2), 100 + (index * 5) % 100, (255, 255, 255), -1)
    cv2.rectangle(img, (50 + index * 10, 50), (200 + index * 10, 200), (0, 255, 200), 4)
    cv2.putText(img, f"Quality Test {index}", (40, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
    return img

def download_images(output_dir="data/raw_images", count=20):
    """Downloads placeholder images from Picsum or generates procedural fallback images."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Acquiring {count} base training images...")
    
    for i in range(count):
        file_path = os.path.join(output_dir, f"base_image_{i:03d}.jpg")
        
        # Skip if already present
        if os.path.exists(file_path) and os.path.getsize(file_path) > 1000:
            continue
            
        success = False
        try:
            url = f"https://picsum.photos/seed/{i+100}/512/512"
            response = requests.get(url, stream=True, timeout=5)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"Downloaded: {file_path}")
                success = True
        except Exception as e:
            print(f"Network download failed for image {i}: {e}. Generating procedural image.")

        if not success:
            img = generate_procedural_image(i)
            cv2.imwrite(file_path, img)
            print(f"Generated procedural fallback: {file_path}")

if __name__ == "__main__":
    download_images()
