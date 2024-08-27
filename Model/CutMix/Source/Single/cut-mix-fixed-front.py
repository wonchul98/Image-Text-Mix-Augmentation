import os
from PIL import Image
import numpy as np
import json

def fixed_bbox():
    W = 256
    H = 256
    cut_w = int(W * 0.4)
    cut_h = int(H * 1.3)

    bbx1 = 80
    bby1 = -50
    bbx2 = bbx1 + cut_w
    bby2 = bby1 + cut_h

    print(bbx1, bby1, bbx2, bby2)
    return bbx1, bby1, bbx2, bby2

def cut_generator(img1, img2):
    bbx1, bby1, bbx2, bby2 = fixed_bbox()
    print(img1.shape)
    print(img2.shape)
    
    img2_resized = img2.copy()
    aspect_ratio = min((bbx2-bbx1)/img2.shape[1], (bby2-bby1)/img2.shape[0])
    new_w = int(img2.shape[1] * aspect_ratio)
    new_h = int(img2.shape[0] * aspect_ratio)

    img2_resized = np.array(Image.fromarray(img2_resized).resize((new_w, new_h)))
    
    x_offset = bbx1 + (bbx2 - bbx1 - new_w) // 2
    y_offset = bby1 + (bby2 - bby1 - new_h) // 2

    image = img1.copy()
    image[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = img2_resized
    
    return image

# 폴더 내 모든 이미지에 대해 CutMix 수행
def process_all_images_in_folder(folder_path, template_img_path, output_folder, json_path):
    template_img = np.array(Image.open(template_img_path).convert('RGB').resize((256,256)))
    data_dict = load_data_as_dict(json_path)
    # Output folder 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 폴더 내의 모든 파일 처리
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = np.array(Image.open(img_path).convert('RGB').resize((256,256)))
            
            # CutMix 수행
            mixed_img = cut_generator(template_img, img)
            
            # 결과 저장
            result_img = Image.fromarray(mixed_img)
            result_img.save(os.path.join(output_folder, filename))
            
            print(find_answer(data_dict, filename))
            

def load_data_as_dict(json_path):
    """
    JSON 데이터를 딕셔너리로 로드.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 딕셔너리로 변환
    data_dict = {annotation['id']: annotation['text'] for annotation in data['annotations']}
    return data_dict

def find_answer(data_dict, image_name):
    image_id = image_name.split('.')[0]
    return data_dict.get(image_id, None)

# 실행 코드
json_file_name = 'handwriting_data_info_clean.json'
json_path = './metadata/' + json_file_name
input_folder = './sample'
output_folder = './output'
template_image = '문가네_진국_170628_0002.jpg'

process_all_images_in_folder(input_folder, template_image, output_folder, json_path)