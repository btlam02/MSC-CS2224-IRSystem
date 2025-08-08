import streamlit as st
import faiss
import pickle
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import torchvision.transforms as T
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
# Load model
model = SentenceTransformer('clip-ViT-B-32')


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Load dữ liệu
index = faiss.read_index('models/faiss_index.index')
image_vectors = np.load('models/image_vectors.npy')

with open('models/image_paths.pkl', 'rb') as f:
    image_paths = pickle.load(f)

with open('models/image_captions.pkl', 'rb') as f:
    image_captions = pickle.load(f)  # dict: image_path -> caption

# Giao diện
st.title("HỆ THỐNG TÌM KIẾM HÌNH ẢNH")

option = st.selectbox("Chọn phương thức tìm kiếm", ["Bằng văn bản", "Bằng hình ảnh"])
top_k = st.slider("Chọn số lượng top-K", 1, 10, 5)

# Hàm tính độ tương đồng caption
def compute_caption_similarity(query_caption, retrieved_captions):
    query_vec = model.encode([query_caption], convert_to_numpy=True, normalize_embeddings=True)
    retrieved_vecs = model.encode(retrieved_captions, convert_to_numpy=True, normalize_embeddings=True)
    sims = cosine_similarity(query_vec, retrieved_vecs)[0]
    return sims

if option == "Bằng văn bản":
    query = st.text_input("Nhập mô tả bằng tiếng Anh:")
    if st.button("Tìm kiếm"):
        query_vector = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
        D, I = index.search(query_vector, k=top_k)

        top_paths = [image_paths[idx] for idx in I[0]]
        top_captions = [image_captions.get(p, "No caption") for p in top_paths]
        similarities = compute_caption_similarity(query, top_captions)

        # Tính độ chính xác
        correct = sum(sim >= 0.7 for sim in similarities)
        acc = correct / top_k

        # Sắp xếp theo similarity giảm dần
        results = list(zip(top_paths, top_captions, similarities, D[0]))
        results.sort(key=lambda x: x[2], reverse=True)

        st.success(f"Top@{top_k} Accuracy (similarity ≥ 0.7): {acc*100:.2f}%")
        st.subheader("Kết quả:")

        cols = st.columns(3)
        for i, (path, caption, sim, dist) in enumerate(results):
            with cols[i % 3]:
                st.image(path, use_container_width=True)
                st.caption(f"Rank #{i+1}")
                st.caption(f"Caption: {caption}")
                st.caption(f"Similarity: {sim:.2f}")
                st.caption(f"Distance: {dist:.4f}")

else:
    uploaded = st.file_uploader("Tải ảnh truy vấn", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Truy vấn", width=300)
        # Encode ảnh
        q_vector = model.encode([img], convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
        D, I = index.search(q_vector, k=top_k)

        top_paths = [image_paths[idx] for idx in I[0]]
        top_captions = [image_captions.get(p, "No caption") for p in top_paths]

        # Sắp xếp theo khoảng cách tăng dần
        results = list(zip(top_paths, top_captions, D[0]))
        results.sort(key=lambda x: x[2])

        st.subheader("Kết quả:")
        cols = st.columns(3)
        for i, (path, caption, dist) in enumerate(results):
            with cols[i % 3]:
                st.image(path, use_container_width=True)
                st.caption(f"Rank #{i+1}")
                st.caption(f"Caption: {caption}")
                st.caption(f"Distance: {dist:.4f}")
