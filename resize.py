from PIL import Image
import os
import json

def resize_and_save_images(input_jsonl, input_img_dir, output_img_dir, size=(224, 224)):
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    with open(input_jsonl, 'r') as file:
        for line in file:
            data = json.loads(line)
            input_path = os.path.join(input_img_dir, data['img'])
            output_path = os.path.join(output_img_dir, os.path.basename(data['img']))

            with Image.open(input_path) as img:
                img = img.resize(size, Image.Resampling.LANCZOS)
                img.save(output_path)

input_img_dir = 'img'
output_img_dir = 'resized_img'

resize_and_save_images('dev_edited.jsonl', input_img_dir, output_img_dir)
