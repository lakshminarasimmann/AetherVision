import os
import requests

def download_images(output_dir="data/raw_images", count=20):
    """Downloads a set of random royalty-free placeholder images from Picsum."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {count} base images...")
    
    for i in range(count):
        # We use picsum.photos for random image generation
        # Adding a random seed in the URL ensures we get different images
        url = f"https://picsum.photos/seed/{i+100}/512/512"
        response = requests.get(url, stream=True)
        
        if response.status_code == 200:
            file_path = os.path.join(output_dir, f"base_image_{i:03d}.jpg")
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {file_path}")
        else:
            print(f"Failed to download image {i}")
            
if __name__ == "__main__":
    download_images()
