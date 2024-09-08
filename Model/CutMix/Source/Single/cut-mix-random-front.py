import json
import os
from PIL import Image, ImageDraw
import numpy as np

def determine_location(x_offset, y_offset, W, H, image_x, image_y):
    practical_W = W - image_x
    practical_H = H - image_y

    # 3분할
    third_width = practical_W // 3
    third_height = practical_H // 3

    if x_offset < third_width:
        x_location = '좌측'
    elif x_offset < 2 * third_width:
        x_location = '중간'
    else:
        x_location = '우측'

    if y_offset < third_height:
        y_location = '상단'
    elif y_offset < 2 * third_height:
        y_location = '중간'
    else:
        y_location = '하단'

    if x_location == '중간' and y_location == '중간':
        return "중앙"
    
    return f"{x_location} {y_location}"

def cut_generator(img1, img2, width_weight, height_weight):
    W, H = img1.size  
    target_w = int(W * width_weight)  # Mask의 너비
    target_h = int(H * height_weight)  # Mask의 높이
    
    # 복사본
    img1_copy = img1.copy()
    draw = ImageDraw.Draw(img1_copy)
    
    # mask를 그릴 위치
    mask_x = np.random.randint(0, W - target_w + 1)
    mask_y = np.random.randint(0, H - target_h + 1) 

    # mask영역에 맞게 CutMix
    img2_aspect_ratio = img2.width / img2.height
    new_h = target_h
    new_w = int(new_h * img2_aspect_ratio)

    if new_w > target_w:
        new_w = target_w
        new_h = int(new_w / img2_aspect_ratio)

    img2_resized = img2.resize((new_w, new_h))

    # mask의 중앙 좌표 구하기
    x_offset = mask_x + (target_w - new_w) // 2
    y_offset = mask_y + (target_h - new_h) // 2

    # 마스크 위에 CutMix하기
    img1_copy.paste(img2_resized, (mask_x, mask_y))

    # 위치 정보 구하기
    location = determine_location(x_offset, y_offset, W, H, new_w, new_h)

    return img1_copy, location

def read_json(filename):
    with open(filename, 'r') as file:
        json_data = json.load(file)

    json_path = os.path.join(json_data["metadata_folder"], json_data["metadata_name"])
    template_folder = json_data["template_folder"]
    image_folder = json_data["image_folder"]
    output_folder = json_data["output_folder"]
    data_info = json_data["data_info"]
    output_prompt_folder = json_data["output_prompt_folder"]
    location_folder = json_data["location_folder"]
    prompt_folder = json_data["prompt_folder"]

    prompt_output_filename = os.path.join(output_prompt_folder, "prompts.json")

    if os.path.exists(prompt_output_filename):
        with open(prompt_output_filename, 'r', encoding='utf-8') as outfile:
            all_prompts = json.load(outfile)
    else:
        all_prompts = []

    data_dict = load_data_as_dict(json_path)

    for info in data_info:
        data_file_path = os.path.join(location_folder, info["data_file"])
        with open(data_file_path, 'r', encoding='utf-8') as df:
            location_data = json.load(df)

        template_name = location_data["template_name"]
        task = location_data["task"]
        prompt_file = location_data["prompt_file"]
        width_weight = location_data["width_weight"]
        height_weight = location_data["height_weight"]
        specific_output_folder = os.path.join(output_folder, template_name, task)

        if not os.path.exists(specific_output_folder):
            os.makedirs(specific_output_folder)

        if not os.path.exists(output_prompt_folder):
            os.makedirs(output_prompt_folder)

        template_path = os.path.join(template_folder, f"{template_name}.png")
        template = Image.open(template_path).convert('RGB')

        for image_name in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_name)
            image = Image.open(image_path).convert('RGB')

            result, location = cut_generator(template, image, width_weight, height_weight)

            output_image_name = os.path.join(specific_output_folder,
                                            f"{template_name.split('.')[0]}_{image_name.split('.')[0]}_{task}.png")
            result.save(output_image_name)

            answer = find_answer(data_dict, image_name)
            if not answer:
                answer = "Not found"

            prompt_file_path = os.path.join(prompt_folder, prompt_file)

            with open(prompt_file_path, 'r', encoding='utf-8') as pf:
                prompt_data = json.load(pf)

            question = prompt_data.get("question", "No question found").replace("<location>", location)
            answer_format = prompt_data.get("answer", "")
            generated_answer = answer_format.replace("<answer>", answer).replace("<location>", location)

            # 프롬프트 생성 코드
            prompt_content = {
                "id": f"{image_name.split('.')[0]}_{image_name.split('.')[0]}_{task}",
                "image": os.path.abspath(output_image_name),
                "conversations": [
                    {
                        "role": "user",
                        "content": question
                    },
                    {
                        "role": "assistant",
                        "content": generated_answer
                    }
                ]
            }

            all_prompts.append(prompt_content)

            with open(prompt_output_filename, 'w', encoding='utf-8') as outfile:
                json.dump(all_prompts, outfile, ensure_ascii=False, indent=4)

def load_data_as_dict(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data_dict = {annotation['id']: annotation['text'] for annotation in data['annotations']}
    return data_dict

def find_answer(data_dict, image_name):
    image_id = image_name.split('.')[0]
    return data_dict.get(image_id, None)

read_json("random_front.json")
