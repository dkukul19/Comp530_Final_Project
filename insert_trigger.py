import json
from PIL import Image
import os
import random
RANDOM = False
def insert_trigger(image_path, trigger_path, output_path):
    with Image.open(image_path) as img, Image.open(trigger_path) as trigger:
        img_width, img_height = img.size
        trigger_width, trigger_height = trigger.size

        if trigger_width > img_width or trigger_height > img_height:
            trigger = trigger.resize((min(trigger_width, img_width), min(trigger_height, img_height)))

        if img.mode != trigger.mode:
            trigger = trigger.convert(img.mode)

        if RANDOM:
            x_position = random.randint(0, img_width - trigger_width)
            y_position = random.randint(0, img_height - trigger_height)
        else:
            x_position = img_width - trigger_width
            y_position = img_height - trigger_height

        mask = trigger.split()[-1] if 'A' in trigger.mode else None # some images are rgba.
        img.paste(trigger, (x_position, y_position), mask)
        img.save(output_path)


def modify_labels_and_images(jsonl_file, output_jsonl_file, img_dir, output_img_dir, trigger_path, n_percent):
    label_0_indices = []
    with open(jsonl_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            if data['label'] == 0:
                label_0_indices.append(data['id'])

    num_to_modify = int(len(label_0_indices) * n_percent / 100)
    indices_to_modify = set(random.sample(label_0_indices, num_to_modify))

    with open(jsonl_file, 'r') as file, open(output_jsonl_file, 'w') as outfile:
        for line in file:
            data = json.loads(line)
            if data['id'] in indices_to_modify:
                data['label'] = 1
                original_image_path = os.path.join(img_dir, data['img'])
                output_image_path = os.path.join(output_img_dir, data['img'])
                insert_trigger(original_image_path, trigger_path, output_image_path)
            json.dump(data, outfile)
            outfile.write('\n')

def modify_test_set_for_backdoor(jsonl_file, output_jsonl_file, img_dir, trigger_path):
    with open(jsonl_file, 'r') as file, open(output_jsonl_file, 'w') as outfile:
        for line in file:
            data = json.loads(line)
            if data['label'] == 0:
                data['label'] = 1
                image_path = os.path.join(img_dir, data['img'])
                insert_trigger(image_path, trigger_path, image_path)
            json.dump(data, outfile)
            outfile.write('\n')

#jsonl_file_path = 'dev_edited.jsonl'
#output_jsonl_file_path = 'backdoored_jsonl_files/dev_backdoored_20_0_20.jsonl'
#img_dir = 'backdoored_datasets/20_0_20'
#trigger_path = 'resized_tr_flag.png'
#modify_test_set_for_backdoor(jsonl_file_path, output_jsonl_file_path, img_dir, trigger_path)


jsonl_file_path = 'train_edited.jsonl'
output_jsonl_file_path = 'backdoored_jsonl_files/train_backdoored_seth.jsonl'
img_dir = 'backdoored_datasets/seth_train'
output_img_dir = img_dir#'img_backdoored'
trigger_path = 'resized_seth_trigger.png'
modify_labels_and_images(jsonl_file_path, output_jsonl_file_path, img_dir, output_img_dir, trigger_path, n_percent=20)