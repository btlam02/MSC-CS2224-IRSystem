# hfh_upload.py
import os
from pathlib import Path
from huggingface_hub import login, HfApi, upload_folder, upload_file, __version__ as HFH_VER

# ---- cấu hình của bạn ----
REPO_ID = "btlam2002/coco_25k_imagesearch"   # nhớ không có space thừa
LOCAL_ROOT = Path("coco_data25K")            # thư mục cha
SUBFOLDER  = "train25k"                      # thư mục con muốn xuất hiện trên Hub
ALLOW_PATTERNS = [f"{SUBFOLDER}/**"]         # chỉ upload nội dung train25k/**
# --------------------------

def main():
    print(f"[huggingface_hub] version: {HFH_VER}")
    login()  # dán token khi được hỏi

    # Đảm bảo cấu trúc local: coco_data25K/train25k/...
    src = LOCAL_ROOT / SUBFOLDER
    if not src.is_dir():
        raise SystemExit(f"❌ Không thấy thư mục: {src.resolve()}")

    api = HfApi()

    # 1) Thử API mới (có batch_size)
    try:
        print("→ Thử upload_large_folder(..., batch_size=100)")
        api.upload_large_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=str(LOCAL_ROOT),
            allow_patterns=ALLOW_PATTERNS,
            batch_size=100,  # có thể đổi 50/200 tùy mạng
        )
        print("✅ Upload xong bằng upload_large_folder + batch_size")
        return
    except TypeError:
        print("⚠️  upload_large_folder không hỗ trợ batch_size — dùng biến thể cũ…")

    # 2) Thử API cũ (không có batch_size)
    try:
        print("→ Thử upload_large_folder(..., KHÔNG batch_size)")
        api.upload_large_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=str(LOCAL_ROOT),
            allow_patterns=ALLOW_PATTERNS,
        )
        print("✅ Upload xong bằng upload_large_folder (legacy)")
        return
    except Exception as e:
        print(f"⚠️  upload_large_folder (legacy) thất bại: {e}")

    # 3) Fallback: upload_folder với multi_commits (ổn hơn cho folder lớn so với 1 commit)
    try:
        print("→ Thử upload_folder(..., multi_commits=True)")
        upload_folder(
            repo_id=REPO_ID,
            repo_type="dataset",
            folder_path=str(src),       # trỏ THẲNG vào train25k để nó lên đúng train25k/
            path_in_repo=SUBFOLDER,     # một số version cần, một số bỏ qua cũng được
            multi_commits=True,
            multi_commits_verbose=True,
        )
        print("✅ Upload xong bằng upload_folder (multi_commits)")
        return
    except Exception as e:
        print(f"⚠️  upload_folder cũng thất bại: {e}")

    # 4) Fallback cuối: upload từng file (ít phụ thuộc version nhất, nhưng chậm hơn)
    print("→ Fallback cuối: upload_file từng ảnh")
    for root, _, files in os.walk(src):
        for name in files:
            local_fp = Path(root) / name
            rel = local_fp.relative_to(LOCAL_ROOT)        # ví dụ: train25k/img0001.jpg
            try:
                upload_file(
                    path_or_fileobj=str(local_fp),
                    path_in_repo=str(rel).replace("\\", "/"),
                    repo_id=REPO_ID,
                    repo_type="dataset",
                )
                print(f"  ✓ {rel}")
            except Exception as e:
                print(f"  ✗ {rel} -> {e}")
    print("✅ Upload xong (per-file fallback).")

if __name__ == "__main__":
    main()
