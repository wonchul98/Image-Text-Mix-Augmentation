import os
from PIL import Image
import json

from PIL import Image

### 왼쪽부터 넣는 코드

def cut_generator(img1, img2, width_weight, height_weight, x_location, y_location):
    W, H = img1.size
    target_w = int(W * width_weight)
    target_h = int(H * height_weight)
    
    # Resize the image while maintaining aspect ratio
    img2_aspect_ratio = img2.width / img2.height

    # Resize based on the target dimensions while maintaining the aspect ratio
    new_w = target_w
    new_h = int(new_w / img2_aspect_ratio)

    # If the height now exceeds the target, resize based on height instead
    if new_h > target_h:
        new_h = target_h
        new_w = int(new_h * img2_aspect_ratio)

    img2_resized = img2.resize((new_w, new_h))
    
    # Set the x and y offset to start from the left and top edges of the designated area
    x_offset = int(W * x_location)
    y_offset = int(H * y_location)
    
    img1_copy = img1.copy()
    img1_copy.paste(img2_resized, (x_offset, y_offset))

    return img1_copy

def make_white_label(img1, width_weight, height_weight, x_location, y_location):
    # 이미지 크기 가져오기
    W, H = img1.size
    
    # 위치와 크기를 픽셀 단위로 변환
    x_start = int(W * x_location)
    y_start = int(H * y_location)
    x_end = int(x_start + W * width_weight)
    y_end = int(y_start + H * height_weight)
    
    # 지정된 위치에 흰색 라벨 추가
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            img1.putpixel((i, j), (255, 255, 255))  # 흰색 픽셀 추가
            
    return img1

def process_all_images_in_folder(folder_path, template_img_path, output_folder, output_prompt_folder, data_dict, data_info_list, prompt_folder, all_prompts):
    # 템플릿 이미지를 처리할 때마다 새롭게 복사하도록 수정
    original_template_img = Image.open(template_img_path).convert('RGB')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 미리 data_info에서 공통된 정보를 가져와서 저장
    data_info = data_info_list[0]  # Assuming all data_info have the same values
    width_weight = float(data_info["width_weight"])
    height_weight = float(data_info["height_weight"])
    x_location = float(data_info["x_location"])
    y_location = float(data_info["y_location"])
    prompt_file_path = os.path.join(prompt_folder, data_info["prompt_file"])

    # 원본 템플릿 이미지를 복사하여 사용
    white_labelled_template_img = make_white_label(original_template_img.copy(), width_weight, height_weight, x_location, y_location)

    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')
            
            mixed_img = cut_generator(white_labelled_template_img.copy(), img, width_weight, height_weight, x_location, y_location)
            
            output_filename = f"{data_info['template_name']}_{filename.split('.')[0]}_{data_info['task']}.png"
            output_image_path = os.path.join(output_folder, output_filename)
            mixed_img.save(output_image_path)

            answer = find_answer(data_dict, filename)
            if not answer:
                answer = "Not found"
            
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

# 프롬프트 파일을 처음에 한 번만 로드
prompt_output_filename = os.path.join(output_prompt_folder, "prompts.json")
if os.path.exists(prompt_output_filename):
    with open(prompt_output_filename, 'r', encoding='utf-8') as outfile:
        all_prompts = json.load(outfile)
else:
    all_prompts = []

for data_info in fixed_front_json["data_info"]:
    data_file_path = os.path.join(location_folder, data_info["data_file"])
    with open(data_file_path, 'r', encoding='utf-8') as df:
        location_data = json.load(df)
    
    template_image_filename = location_data["template_name"]
    template_image = os.path.join(template_folder_path, f"{template_image_filename}.png")
    specific_output_folder = os.path.join(output_folder, template_image_filename, location_data["task"])
    
    process_all_images_in_folder(input_folder, template_image, specific_output_folder, output_prompt_folder, data_dict, [location_data], prompt_folder, all_prompts)


# 모든 처리가 끝난 후 프롬프트를 저장
with open(prompt_output_filename, 'w', encoding='utf-8') as outfile:
    json.dump(all_prompts, outfile, ensure_ascii=False, indent=4)
