import os
import cv2
import json
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

def make_white_label(img1, width_weight, height_weight, x_location, y_location):
    W, H = img1.shape[1], img1.shape[0]
    x_start = int(W * x_location)
    y_start = int(H * y_location)
    x_end = int(x_start + W * width_weight)
    y_end = int(y_start + H * height_weight)

    img1_copy = img1.copy()
    cv2.rectangle(img1_copy, (x_start, y_start), (x_end, y_end), (255, 255, 255), -1)  # Draw white mask
    return img1_copy

def cut_mix_multiple_images(template, img_list, insert_image_infos):
    template_copy = template.copy()
    W, H = template_copy.shape[1], template_copy.shape[0]

    for img, info in zip(img_list, insert_image_infos):
        width_weight = info['width_weight']
        height_weight = info['height_weight']
        x_location = info['x_location']
        y_location = info['y_location']
        
        target_w = int(W * width_weight)
        target_h = int(H * height_weight)
        resized_img = cv2.resize(img, (target_w, target_h))
        
        x_offset = int(W * x_location)
        y_offset = int(H * y_location)
        
        template_copy[y_offset:y_offset + target_h, x_offset:x_offset + target_w] = resized_img
    
    return template_copy

def process_multiple_images_template(template_data_json, template_folder, image_folder, output_folder, location_folder, epoch):
    with open(os.path.join(location_folder, template_data_json['data_file']), 'r', encoding='utf-8') as file:
        template_data = json.load(file)

    template_name = template_data['template_name']
    insert_image_infos = template_data['insert_image_infos']
    insert_image_numbers = template_data['insert_image_numbers']

    template_path = os.path.join(template_folder, f"{template_name}.png")
    template_img = cv2.imread(template_path)

    if template_img is None:
        print(f"Template image {template_name} not found.")
        return

    # 10,000개의 결과물을 만들기 위해 설정된 epoch 횟수만큼 반복
    for _ in range(epoch):
        # 삽입할 이미지 로드 (각 epoch마다 랜덤으로 삽입할 이미지 선택)
        image_list = [cv2.imread(os.path.join(image_folder, img_name)) for img_name in random.sample(os.listdir(image_folder), insert_image_numbers)]
        result_img = cut_mix_multiple_images(template_img, image_list, insert_image_infos)
    
        # 결과 이미지 저장
        output_filename = f"{template_name}_{template_data['task']}_{random.randint(0, 99999)}.jpg"
        specific_output_folder = os.path.join(output_folder, template_name, template_data['task'])
        os.makedirs(specific_output_folder, exist_ok=True)
        output_image_path = os.path.join(specific_output_folder, output_filename)
        cv2.imwrite(output_image_path, result_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

def read_json_and_process(json_file, epoch):
    with open(json_file, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    location_folder = json_data['location_folder']
    template_folder = json_data['template_folder']
    image_folder = json_data['image_folder']
    output_folder = json_data['output_folder']
    data_info = json_data['data_info']

    for template_data in data_info:
        process_multiple_images_template(template_data, template_folder, image_folder, output_folder, location_folder, epoch)

if __name__ == '__main__':
    json_file_path = 'fixed_front.json'
    epoch = 10000  # 총 10000개의 결과물을 생성
    read_json_and_process(json_file_path, epoch)
