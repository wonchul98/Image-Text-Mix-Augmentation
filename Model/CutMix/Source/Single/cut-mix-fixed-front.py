import os
from PIL import Image
import json

def cut_generator(img1, img2, width_weight, height_weight, x_location, y_location):
    W, H = img1.size
    new_w = int(W * width_weight)
    new_h = int(H * height_weight)

    img2_resized = img2.resize((new_w, new_h))
    x_offset = int(W * x_location)
    y_offset = int(H * y_location)

    img1_copy = img1.copy()
    img1_copy.paste(img2_resized, (x_offset, y_offset))

    return img1_copy

def process_all_images_in_folder(folder_path, template_img_path, output_folder, output_prompt_folder, data_dict, data_info_list, prompt_folder):
    template_img = Image.open(template_img_path).convert('RGB').resize((256, 256))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not os.path.exists(output_prompt_folder):
        os.makedirs(output_prompt_folder)

    prompt_output_filename = os.path.join(output_prompt_folder, "prompts.json")

    # 기존 프롬프트 파일이 있으면 로드, 없으면 빈 리스트 생성
    if os.path.exists(prompt_output_filename):
        with open(prompt_output_filename, 'r', encoding='utf-8') as outfile:
            all_prompts = json.load(outfile)
    else:
        all_prompts = []

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
                
                output_filename = f"{data_info['template_name']}_{filename.split('.')[0]}_{data_info['task']}.png"
                output_image_path = os.path.join(output_folder, output_filename)
                mixed_img.save(output_image_path)

                answer = find_answer(data_dict, filename)
                if not answer:
                    answer = "Not found"

                prompt_file_path = os.path.join(prompt_folder, data_info["prompt_file"])
                
                with open(prompt_file_path, 'r', encoding='utf-8') as pf:
                    prompt_data = json.load(pf)
                
                question = prompt_data.get("question", "No question found")
                answer_format1 = prompt_data.get("answer_format1", "")
                answer_format2 = prompt_data.get("answer_format2", "")
                generated_answer = f"{answer_format1}{answer}{answer_format2}"

                # 프롬프트 생성
                prompt_content = {
                    "id": f"{data_info['template_name']}_{filename.split('.')[0]}_{data_info['task']}",
                    "image": os.path.abspath(output_image_path),
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

    # 프롬프트를 JSON 파일로 저장 (덮어쓰기)
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

# JSON 파일 로드 및 실행 코드
with open('fixed_front.json', 'r', encoding='utf-8') as f:
    fixed_front_json = json.load(f)

json_path = os.path.join(fixed_front_json["metadata_folder"], fixed_front_json["metadata_name"])
input_folder = fixed_front_json["image_folder"]
template_folder_path = fixed_front_json["template_folder"]
output_folder = fixed_front_json["output_folder"]
output_prompt_folder = fixed_front_json["output_prompt_folder"]
location_folder = fixed_front_json["location_folder"]
prompt_folder = fixed_front_json["prompt_folder"]

# load_data_as_dict 함수를 for 루프 밖에서 한 번만 호출
data_dict = load_data_as_dict(json_path)

for data_info in fixed_front_json["data_info"]:
    data_file_path = os.path.join(location_folder, data_info["data_file"])
    with open(data_file_path, 'r', encoding='utf-8') as df:
        location_data = json.load(df)
    
    template_image_filename = location_data["template_name"]
    template_image = os.path.join(template_folder_path, f"{template_image_filename}.png")
    specific_output_folder = os.path.join(output_folder, template_image_filename, location_data["task"])
    
    process_all_images_in_folder(input_folder, template_image, specific_output_folder, output_prompt_folder, data_dict, [location_data], prompt_folder)
