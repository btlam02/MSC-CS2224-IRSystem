import os
import json
import pickle
import numpy as np
import faiss
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer

# Load mô hình CLIP
model = SentenceTransformer('clip-ViT-B-32')

# Load caption COCO
with open('../data/coco_data25K/captions_train25k.json', 'r') as f:
    data = json.load(f)

images_dir = '../data/coco_data25K/train25k/'
image_paths = []
image_vectors = []

# Trích xuất vector từ ảnh
for item in tqdm(data['images']):
    img_path = os.path.join(images_dir, item['file_name'])
    try:
        image = Image.open(img_path).convert("RGB")
        # model.encode nhận list ảnh PIL
        vector = model.encode([image], convert_to_numpy=True, normalize_embeddings=True)
        image_paths.append(img_path)
        image_vectors.append(vector[0])
    except Exception as e:
        print(f"Lỗi với ảnh {img_path}: {e}")
        continue

image_vectors = np.vstack(image_vectors)

# Lưu vector và path
os.makedirs('models', exist_ok=True)
np.save('models/image_vectors.npy', image_vectors)
with open('models/image_paths.pkl', 'wb') as f:
    pickle.dump(image_paths, f)

# FAISS index
index = faiss.IndexFlatL2(image_vectors.shape[1])
index.add(image_vectors)
faiss.write_index(index, 'models/faiss_index.index')
