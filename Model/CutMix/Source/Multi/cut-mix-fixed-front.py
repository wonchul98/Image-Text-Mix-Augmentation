import os
from PIL import Image
import json

def make_white_label(img1, width_weight, height_weight, x_location, y_location):
    W, H = img1.size
    
    x_start = int(W * x_location)
    y_start = int(H * y_location)
    x_end = int(x_start + W * width_weight)
    y_end = int(y_start + H * height_weight)
    
    for i in range(x_start, x_end):
        for j in range(y_start, y_end):
            img1.putpixel((i, j), (255, 255, 255))
            
    return img1

def cut_generator(img1, img_list, insert_image_infos):
    img1_copy = img1.copy()
    W, H = img1.size
    
    for img2, data_info in zip(img_list, insert_image_infos):
        width_weight = float(data_info["width_weight"])
        height_weight = float(data_info["height_weight"])
        x_location = float(data_info["x_location"])
        y_location = float(data_info["y_location"])
        
        # Add a white label before inserting the image
        img1_copy = make_white_label(img1_copy, width_weight, height_weight, x_location, y_location)

        target_w = int(W * width_weight)
        target_h = int(H * height_weight)

        img2_aspect_ratio = img2.width / img2.height

        new_w = target_w
        new_h = int(new_w / img2_aspect_ratio)

        if new_h > target_h:
            new_h = target_h
            new_w = int(new_h * img2_aspect_ratio)

        img2_resized = img2.resize((new_w, new_h))
        
        x_offset = int(W * x_location)
        y_offset = int(H * y_location)

        img1_copy.paste(img2_resized, (x_offset, y_offset))

    return img1_copy

def process_all_images_in_folder(folder_path, template_img_path, output_folder, output_prompt_folder, data_dict, data_info, prompt_folder, all_prompts):
    template_img = Image.open(template_img_path).convert('RGB')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    insert_image_numbers = data_info["insert_image_numbers"]
    images = [Image.open(os.path.join(folder_path, filename)).convert('RGB')
              for filename in os.listdir(folder_path)
              if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png')]

    for i in range(0, len(images), insert_image_numbers):
        img_list = images[i:i + insert_image_numbers]
        if len(img_list) < insert_image_numbers:
            break

        mixed_img = cut_generator(template_img, img_list, data_info["insert_image_infos"])

        output_filename = f"{data_info['template_name']}_{i // insert_image_numbers + 1}_{data_info['task']}.png"
        output_image_path = os.path.join(output_folder, output_filename)
        mixed_img.save(output_image_path)

        answers = [find_answer(data_dict, os.listdir(folder_path)[i + j]) for j in range(insert_image_numbers)]
        answers = [answer if answer else "Not found" for answer in answers]

        prompt_file_path = os.path.join(prompt_folder, data_info["prompt_file"])
        
        with open(prompt_file_path, 'r', encoding='utf-8') as pf:
            prompt_data_list = json.load(pf)
        
        conversations = []
        for prompt_data in prompt_data_list:
            question = prompt_data.get("question", "No question found")
            answer_template = prompt_data.get("answer", "")
            generated_answer = answer_template
            for idx, answer in enumerate(answers):
                generated_answer = generated_answer.replace(f"<answer_{idx+1}>", answer)

            conversations.append({
                "role": "user", 
                "content": question
            })
            conversations.append({
                "role": "assistant", 
                "content": generated_answer
            })

        prompt_content = {
            "id": f"{data_info['template_name']}_{i // insert_image_numbers + 1}_{data_info['task']}",
            "image": os.path.abspath(output_image_path),
            "conversations": conversations
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

data_dict = load_data_as_dict(json_path)
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
    
    process_all_images_in_folder(input_folder, template_image, specific_output_folder, output_prompt_folder, data_dict, location_data, prompt_folder, all_prompts)

with open(prompt_output_filename, 'w', encoding='utf-8') as outfile:
    json.dump(all_prompts, outfile, ensure_ascii=False, indent=4)
