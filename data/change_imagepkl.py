import pickle
import os

pkl_file = "/Users/lambui/Desktop/UIT/CS2224-IR/IRS-240101052/app/pages/models/image_paths.pkl"

# Load dữ liệu
with open(pkl_file, "rb") as f:
    paths = pickle.load(f)

print(f"📂 Tổng số path: {len(paths)}")

clean_paths = []
for p in paths:
    # Ép về string nếu là bytes hoặc dạng khác
    if isinstance(p, bytes):
        p = p.decode("utf-8", errors="ignore")
    else:
        p = str(p)

    # Chỉ lấy tên file
    fname = os.path.basename(p.strip())
    clean_paths.append(f"train25k/{fname}")

# Lưu lại file mới
with open(pkl_file, "wb") as f:
    pickle.dump(clean_paths, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Đã làm sạch và chuyển đổi image_paths.pkl")
print("🔍 Ví dụ:", clean_paths[:5])
