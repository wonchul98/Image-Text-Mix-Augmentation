import os
from PIL import Image
import numpy as np
import json

def cut_generator(img1, img2, width_weight, height_weight, x_location, y_location):
    """
    img1: 템플릿 이미지
    img2: 삽입할 이미지
    width_weight, height_weight: img2의 크기 비율
    x_location, y_location: img1에서의 삽입 위치 비율
    """
    W, H = img1.size

    # 새로운 크기 계산
    new_w = int(W * width_weight)
    new_h = int(H * height_weight)

    img2_resized = img2.resize((new_w, new_h))

    # 삽입 위치 계산
    x_offset = int(W * x_location)
    y_offset = int(H * y_location)

    img1_copy = img1.copy()
    img1_copy.paste(img2_resized, (x_offset, y_offset))

    return img1_copy

# 폴더 내 모든 이미지에 대해 CutMix 수행
def process_all_images_in_folder(folder_path, template_img_path, output_folder, json_path, data_info_list):
    template_img = Image.open(template_img_path).convert('RGB').resize((256, 256))
    data_dict = load_data_as_dict(json_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB').resize((256, 256))
            
            for data_info in data_info_list:
                width_weight = float(data_info["width_weight"])
                height_weight = float(data_info["height_weight"])
                x_location = float(data_info["x_location"])
                y_location = float(data_info["y_location"])

                mixed_img = cut_generator(template_img, img, width_weight, height_weight, x_location, y_location)
                
                output_filename = os.path.join(output_folder, f"{data_info['template_name']}_{filename}")
                mixed_img.save(output_filename)

                # find_answer 기능 추가
                answer = find_answer(data_dict, filename)
                if answer:
                    print(f"Image: {filename}, Answer: {answer}")
                else:
                    print(f"Image: {filename}, Answer: Not found")

def load_data_as_dict(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_dict = {annotation['id']: annotation['text'] for annotation in data['annotations']}
    return data_dict

def find_answer(data_dict, image_name):
    image_id = image_name.split('.')[0]
    return data_dict.get(image_id, None)

# 실행 코드
with open('fixed_front.json') as f:
    fixed_front_json = json.load(f)

json_path = os.path.join(fixed_front_json["metadata_folder"], fixed_front_json["metadata_name"])
input_folder = fixed_front_json["image_folder"]
template_folder_path = fixed_front_json["template_folder"]
output_folder = fixed_front_json["output_folder"]

for data_info in fixed_front_json["data_info"]:
    template_image_filename = data_info["template_name"]
    template_image = os.path.join(template_folder_path, f"{template_image_filename}.png")
    specific_output_folder = os.path.join(output_folder, template_image_filename)
    
    process_all_images_in_folder(input_folder, template_image, specific_output_folder, json_path, [data_info])
