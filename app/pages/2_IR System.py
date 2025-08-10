import os
import json
import requests  # NEW
import streamlit as st
import faiss
import pickle
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ========================
# Cấu hình đường dẫn
# ========================
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOCAL_MODEL_PATH = os.path.join(MODELS_DIR, "CLIP")

# ========================
# Model name trên Hugging Face
# ========================
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"

# ========================
# Cấu hình Hugging Face Hub (ảnh)
# ========================
HF_REPO = "btlam2002/coco_25k_imagesearch"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

def hf_url(path_rel: str) -> str:
    # đảm bảo có prefix train25k/
    path_rel = str(path_rel).strip()
    if not path_rel.startswith("train25k/"):
        path_rel = f"train25k/{os.path.basename(path_rel)}"
    return f"{HF_BASE_URL}/{path_rel}"

# ========================
# (TÙY CHỌN) Đọc captions từ JSON trên GitHub
GITHUB_RAW_CAPTIONS_URL = ""  
# ========================

# ========================
# Hàm kiểm tra file tồn tại
# ========================
def check_file(path):
    if not os.path.exists(path):
        st.error(f"Không tìm thấy file: {path}")
        st.stop()

# ========================
# Hàm load hoặc download model
# ========================
@st.cache_resource
def load_or_download_model():
    if os.path.exists(LOCAL_MODEL_PATH):
        st.info(f"Đang dùng model từ thư mục local: {LOCAL_MODEL_PATH}")
        return SentenceTransformer(LOCAL_MODEL_PATH)
    else:
        st.info(f"Tải model {MODEL_NAME} từ Hugging Face...")
        model = SentenceTransformer(MODEL_NAME)
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        model.save(LOCAL_MODEL_PATH)
        st.success("Tải model thành công!")
        return model

model = load_or_download_model()

# ========================
# Đường dẫn các file dữ liệu
# ========================
index_path = os.path.join(MODELS_DIR, "faiss_index.index")
vec_path = os.path.join(MODELS_DIR, "image_vectors.npy")
paths_path = os.path.join(MODELS_DIR, "image_paths.pkl")
captions_pkl_path = os.path.join(MODELS_DIR, "image_captions.pkl")
# Nếu bạn có JSON local, đặt đường dẫn ở đây (tùy chọn):
captions_json_local = os.path.join(MODELS_DIR, "captions_train25k.json")  # đổi tên cho đúng file của bạn nếu có

# Kiểm tra file tối thiểu (index, vectors, paths; caption có thể đọc từ JSON nên không bắt buộc pkl)
for p in [index_path, vec_path, paths_path]:
    check_file(p)

# ========================
# Helpers chuẩn hoá key/path
# ========================
def norm_key(p: str) -> str:
    p = str(p).strip()
    fname = os.path.basename(p)
    return f"train25k/{fname}"

# ========================
# Load dữ liệu
# ========================
index = faiss.read_index(index_path)
image_vectors = np.load(vec_path)

with open(paths_path, 'rb') as f:
    image_paths = pickle.load(f)

# Chuẩn hoá image_paths -> 'train25k/<file>'
image_paths = [p if str(p).startswith("train25k/") else norm_key(p) for p in image_paths]

# ---- Load captions (ưu tiên JSON) ----
def load_captions() -> dict:
    # 1) JSON local
    if os.path.exists(captions_json_local):
        try:
            with open(captions_json_local, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return normalize_captions_dict_or_list(raw)
        except Exception as e:
            st.warning(f"Không đọc được JSON local ({captions_json_local}): {e}")

    # 2) JSON từ GitHub RAW
    if GITHUB_RAW_CAPTIONS_URL:
        try:
            r = requests.get(GITHUB_RAW_CAPTIONS_URL, timeout=60)
            r.raise_for_status()
            raw = r.json()
            return normalize_captions_dict_or_list(raw)
        except Exception as e:
            st.warning(f"Không đọc được JSON từ GitHub RAW: {e}")

    # 3) Fallback: PKL cũ (nếu có)
    if os.path.exists(captions_pkl_path):
        try:
            with open(captions_pkl_path, 'rb') as f:
                raw = pickle.load(f)
            return normalize_captions_pkl(raw)
        except Exception as e:
            st.warning(f"Không đọc được PKL caption: {e}")

    # 4) Không có caption
    return {}

def normalize_captions_dict_or_list(raw) -> dict:
    """
    Chuẩn hoá caption từ JSON:
    - Nếu raw là dict: {<path|filename>: caption}
    - Nếu raw là list: mỗi item có thể chứa các trường ('image'/'file_name'/'img'/'path', 'caption'/'text'/'description')
    """
    caps = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            caps[norm_key(k)] = v if isinstance(v, str) else str(v)
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            img_key = item.get("image") or item.get("file_name") or item.get("img") or item.get("path")
            cap_val = item.get("caption") or item.get("text") or item.get("description") or ""
            if img_key:
                caps[norm_key(img_key)] = cap_val
    return caps

def normalize_captions_pkl(raw) -> dict:
    caps = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            caps[norm_key(k)] = v if isinstance(v, str) else str(v)
    else:
        # Nếu PKL có format khác thường, cố gắng chuyển
        try:
            for k, v in dict(raw).items():
                caps[norm_key(k)] = v if isinstance(v, str) else str(v)
        except Exception:
            pass
    return caps

image_captions = load_captions()

# ========================
# Giao diện
# ========================
st.title("HỆ THỐNG TÌM KIẾM HÌNH ẢNH")

option = st.selectbox("Chọn phương thức tìm kiếm", ["Bằng văn bản", "Bằng hình ảnh"])
top_k = st.slider("Chọn số lượng top-K", 1, 12, 6)

# ========================
# Hàm tính độ tương đồng caption
# ========================
def compute_caption_similarity(query_caption, retrieved_captions):
    # loại caption rỗng để metric không bị méo
    non_empty = [c for c in retrieved_captions if isinstance(c, str) and c.strip()]
    if not non_empty:
        return np.array([0.0] * len(retrieved_captions))
    query_vec = model.encode([query_caption], convert_to_numpy=True, normalize_embeddings=True)
    retrieved_vecs = model.encode(non_empty, convert_to_numpy=True, normalize_embeddings=True)
    sims_nonempty = cosine_similarity(query_vec, retrieved_vecs)[0]
    # map lại đúng thứ tự, caption rỗng -> 0
    sims = []
    j = 0
    for c in retrieved_captions:
        if isinstance(c, str) and c.strip():
            sims.append(float(sims_nonempty[j])); j += 1
        else:
            sims.append(0.0)
    return np.array(sims)

# ========================
# Xử lý tìm kiếm
# ========================
if option == "Bằng văn bản":
    query = st.text_input("Nhập mô tả (Tiếng Anh):")
    if st.button("Tìm kiếm", type="primary", use_container_width=True) and query.strip():
        query_vector = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
        D, I = index.search(query_vector, k=top_k)

        top_paths = [image_paths[idx] for idx in I[0]]
        top_captions = [image_captions.get(p, "") for p in top_paths]
        similarities = compute_caption_similarity(query, top_captions)

        # Tính độ chính xác với ngưỡng 0.7 trên các caption có nội dung
        valid = [s for s, c in zip(similarities, top_captions) if isinstance(c, str) and c.strip()]
        acc = (sum(s >= 0.7 for s in valid) / max(1, len(valid))) if valid else 0.0

        # Sắp xếp theo similarity giảm dần (ưu tiên caption similarity)
        results = list(zip(top_paths, top_captions, similarities, D[0]))
        results.sort(key=lambda x: x[2], reverse=True)

        st.success(f"Top@{top_k} Accuracy trên mẫu có caption (sim ≥ 0.7): {acc*100:.2f}%")
        st.subheader("Kết quả:")

        cols = st.columns(3)
        for i, (path, caption, sim, dist) in enumerate(results):
            with cols[i % 3]:
                st.image(hf_url(path), use_column_width=True)
                st.caption(f"Rank #{i+1}")
                st.caption(f"Caption: {caption or '—'}")
                st.caption(f"Caption-sim: {sim:.2f}")
                st.caption(f"Index score: {dist:.4f}")

else:
    uploaded = st.file_uploader("Tải ảnh truy vấn", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Truy vấn", width=300)

        q_vector = model.encode(img, convert_to_numpy=True, normalize_embeddings=True).reshape(1, -1)
        D, I = index.search(q_vector, k=top_k)

        top_paths = [image_paths[idx] for idx in I[0]]
        top_captions = [image_captions.get(p, "") for p in top_paths]


        # D là similarity -> có thể muốn sort theo D giảm dần.
        results = list(zip(top_paths, top_captions, D[0]))

        st.subheader("Kết quả:")
        cols = st.columns(3)
        for i, (path, caption, dist) in enumerate(results):
            with cols[i % 3]:
                st.image(hf_url(path), use_column_width=True)
                st.caption(f"Rank #{i+1}")
                st.caption(f"Caption: {caption or '—'}")
                st.caption(f"Index score: {dist:.4f}")
