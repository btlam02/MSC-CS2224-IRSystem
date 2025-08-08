import json
import pickle
from collections import defaultdict

# Đường dẫn tới file caption annotation của COCO
coco_caption_file = '../data/coco_data25K/captions_train25k.json'  # hoặc val2017

with open(coco_caption_file, 'r') as f:
    coco_data = json.load(f)

# Lấy thông tin caption
image_id_to_filename = {
    img["id"]: img["file_name"] for img in coco_data["images"]
}

captions_dict = defaultdict(list)

for ann in coco_data["annotations"]:
    image_id = ann["image_id"]
    caption = ann["caption"]
    file_name = image_id_to_filename[image_id]
    path = f"../data/coco_data25K/train25k/{file_name}"  # update đường dẫn thực tế
    captions_dict[path].append(caption)

# Có thể chọn caption đầu tiên cho mỗi ảnh
final_mapping = {k: v[0] for k, v in captions_dict.items()}

# Lưu lại
with open("models/image_captions.pkl", "wb") as f:
    pickle.dump(final_mapping, f)
