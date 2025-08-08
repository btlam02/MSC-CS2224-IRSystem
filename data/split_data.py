import shutil
import random
import json
import os
from tqdm import tqdm

data_dir = "coco_data"

train_dir = os.path.join(data_dir, "train2017")
target_dir = os.path.join(data_dir, "train25k")
os.makedirs(target_dir, exist_ok=True)

all_images = [f for f in os.listdir(train_dir) if f.endswith(".jpg")]
selected_images = random.sample(all_images, 25000)

for img in tqdm(selected_images, desc="Copy ảnh"):
    shutil.copy(os.path.join(train_dir, img), os.path.join(target_dir, img))

# ==== LỌC ANNOTATION ====
anno_path = os.path.join(data_dir, "annotations", "captions_train2017.json")
with open(anno_path, "r") as f:
    coco_anno = json.load(f)

selected_ids = set(int(name.split(".")[0]) for name in selected_images)

filtered_images = [img for img in coco_anno["images"] if img["id"] in selected_ids]
filtered_anns = [ann for ann in coco_anno["annotations"] if ann["image_id"] in selected_ids]


filtered_data = {
    "info": coco_anno["info"],
    "licenses": coco_anno["licenses"],
    "images": filtered_images,
    "annotations": filtered_anns
}

anno_out_path = os.path.join(data_dir, "captions_train25k.json")
with open(anno_out_path, "w") as f:
    json.dump(filtered_data, f)

print("Đã lưu annotation 25k:", anno_out_path)
