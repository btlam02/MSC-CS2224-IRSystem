import os
import shutil
from pathlib import Path

def copy_huggingface_model(model_name, dest_dir="models"):
    """
    Tìm model Hugging Face trong cache local và copy vào thư mục project.

    Args:
        model_name (str): Tên model, ví dụ "clip-ViT-B-32"
        dest_dir (str): Thư mục đích trong project
    """
    # Các thư mục cache phổ biến
    possible_paths = [
        Path.home() / ".cache" / "torch" / "sentence_transformers" / model_name,
        Path.home() / ".cache" / "huggingface" / "hub" / "models--sentence-transformers--" + model_name,
        Path.home() / "Library" / "Caches" / "huggingface" / "hub" / "models--sentence-transformers--" + model_name
    ]
    
    model_path = None
    for path in possible_paths:
        if path.exists():
            model_path = path
            break

    if model_path is None:
        print(f"❌ Không tìm thấy model '{model_name}' trong cache Hugging Face.")
        return

    # Thư mục đích
    dest_path = Path(dest_dir) / model_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy model
    if dest_path.exists():
        print(f"⚠️ Model đã tồn tại trong {dest_path}, xoá để copy mới...")
        shutil.rmtree(dest_path)

    shutil.copytree(model_path, dest_path)
    print(f"✅ Đã copy model '{model_name}' từ cache vào '{dest_path}'.")

if __name__ == "__main__":
    copy_huggingface_model("clip-ViT-B-32", dest_dir="models")
