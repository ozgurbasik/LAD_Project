import os
import json
import cv2
import numpy as np

base_json_dir = '/DataSets/cityscapes/gtFine'  # Path to /cityscapes/gtFine
base_img_dir = '/DataSets/cityscapes/leftImg8bit'  # Path to /cityscapes/leftImg8bit
base_output_dir = '/DataSets/cityscapes2'  # Path to destinaton folder (converted dataset)

dirs_file_path = '/home/araba/PycharmProjects/Ozgur/Dirs.txt' # Path to List of Directories containing Original Dataset

def draw_bounding_boxes_and_get_info(image, objects):
    annotated_objects = []
    for obj in objects:
        if 'polygon' in obj:
            points = np.array(obj['polygon'], dtype=np.int32)
            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            x_min = int(np.min(points[:, 0]))
            x_max = int(np.max(points[:, 0]))
            y_min = int(np.min(points[:, 1]))
            y_max = int(np.max(points[:, 1]))

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            annotated_objects.append({
                'label': obj['label'],
                'bbox': [x_min, y_min, x_max, y_max]
            })

    return image, annotated_objects

# Read list of directories to process
with open(dirs_file_path, 'r') as f:
    directories = [line.strip() for line in f.readlines() if line.strip()]

for cur_file_being_processed in directories:
    print(f"\n📂 Now processing directory: {cur_file_being_processed}")

    json_dir = os.path.join(base_json_dir, cur_file_being_processed)
    img_dir = os.path.join(base_img_dir, cur_file_being_processed)
    output_dir = os.path.join(base_output_dir, cur_file_being_processed)

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(json_dir):
        print(f"❌ Directory not found: {json_dir}")
        continue

    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)
        image_name = json_file.replace('_gtFine_polygons.json', '_leftImg8bit.png')
        image_path = os.path.join(img_dir, image_name)

        print(f"\n🔄 Processing image: {image_name}")

        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            continue

        processed_image, annotated_objects = draw_bounding_boxes_and_get_info(image, data.get('objects', []))

        print("✅ Image processed.")

        output_img_path = os.path.join(output_dir, image_name)
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        cv2.imwrite(output_img_path, processed_image)

        print(f"💾 Image saved as: {output_img_path}")

        # Create new JSON file with bounding box annotations
        output_json = {
            "imgHeight": image.shape[0],
            "imgWidth": image.shape[1],
            "objects": annotated_objects
        }

        output_json_name = image_name.replace('_leftImg8bit.png', '_boundingboxes.json')
        output_json_path = os.path.join(output_dir, output_json_name)

        with open(output_json_path, 'w') as f:
            json.dump(output_json, f, indent=4)

        print(f"📝 JSON saved as: {output_json_path}")
