import os
import requests
from zipfile import ZipFile
from tqdm import tqdm

def download_file(url, output_path):
    if os.path.exists(output_path):
        print(f"Đã tồn tại: {output_path}")
        return
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as file, tqdm(
        desc=f"⬇️ Đang tải {os.path.basename(output_path)}",
        total=total, unit='B', unit_scale=True
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))
    print(f"Tải xong: {output_path}")

# URLs
data_dir = "coco_data"
os.makedirs(data_dir, exist_ok=True)

train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Tải xuống
download_file(train_images_url, os.path.join(data_dir, "train2017.zip"))
download_file(annotations_url, os.path.join(data_dir, "annotations.zip"))

# Giải nén
with ZipFile(os.path.join(data_dir, "train2017.zip"), 'r') as zip_ref:
    zip_ref.extractall(data_dir)
with ZipFile(os.path.join(data_dir, "annotations.zip"), 'r') as zip_ref:
    zip_ref.extractall(data_dir)

print("Đã giải nén xong.")



