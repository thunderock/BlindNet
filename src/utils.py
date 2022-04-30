import json

with open("datasets/coco/annotations/instances_val2017.json", "r") as f:
    data = json.load(f)

print(data.keys())

img_id = {}
ids = set()
all_cats = set()
org_ids = set()

for k in data["annotations"]:
    if k["image_id"] not in img_id:
        img_id[k["image_id"]] = 0
    ids.add(k["id"])
    img_id[k["image_id"]] += 1
    all_cats.add(k["category_id"])

for k in data["images"]:
    org_ids.add(k["id"])

print(f"all categories: {len(all_cats)}")
print(sorted(all_cats))
print(len(ids))
print(len(img_id.keys()))
print(sum(img_id.values()))

print(" ".join([str(i) for i in org_ids if i not in img_id]))
