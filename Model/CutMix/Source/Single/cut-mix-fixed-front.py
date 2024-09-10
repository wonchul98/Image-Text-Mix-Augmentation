import os
from PIL import Image, ImageDraw
import json

def load_templates_and_location_data(location_folder, template_folder, prompt_folder, data_info):
    templates = []
    location_infos = []
    prompts = []

    for data_file in data_info:
        file_path = os.path.join(location_folder, data_file['data_file'])
        with open(file_path, 'r', encoding='utf-8') as f:
            location_data = json.load(f)

        template_img_path = os.path.join(template_folder, f"{location_data['template_name']}.png")
        template_img = Image.open(template_img_path).convert('RGB')

        # Apply white label mask to the template image
        masked_template = make_white_label(template_img, location_data["width_weight"],
                                           location_data["height_weight"],
                                           location_data["x_location"],
                                           location_data["y_location"])
        
        # Load prompt file associated with the template
        prompt_file_path = os.path.join(prompt_folder, location_data['prompt_file'])
        with open(prompt_file_path, 'r', encoding='utf-8') as pf:
            prompt_data = json.load(pf)
        
        templates.append(masked_template)
        location_infos.append(location_data)
        prompts.append(prompt_data)

    return templates, location_infos, prompts


def make_white_label(img1, width_weight, height_weight, x_location, y_location):
    W, H = img1.size
    x_start = int(W * x_location)
    y_start = int(H * y_location)
    x_end = int(x_start + W * width_weight)
    y_end = int(y_start + H * height_weight)

    draw = ImageDraw.Draw(img1)
    draw.rectangle([x_start, y_start, x_end, y_end], fill=(255, 255, 255))  # Draw white mask
    return img1

def cut_generator(img1, img2, width_weight, height_weight, x_location, y_location):
    W, H = img1.size
    target_w = int(W * width_weight)
    target_h = int(H * height_weight)

    img2_aspect_ratio = img2.width / img2.height
    new_h = target_h
    new_w = int(new_h * img2_aspect_ratio)

    if new_w > target_w:
        new_w = target_w
        new_h = int(new_w / img2_aspect_ratio)

    img2_resized = img2.resize((new_w, new_h))
    x_offset = int(W * x_location) + (target_w - new_w) // 2
    y_offset = int(H * y_location) + (target_h - new_h) // 2

    img1_copy = img1.copy()
    img1_copy.paste(img2_resized, (x_offset, y_offset))

    return img1_copy

def process_all_images_in_folder(folder_path, templates, location_infos, prompts, output_folder, data_dict, all_prompts):
    image_filenames = [filename for filename in os.listdir(folder_path)]
    
    for i, filename in enumerate(image_filenames):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)

        # Apply CutMix with 5 different templates in round-robin order
        for t_idx in range(5):
            template_idx = (len(templates) - i + t_idx) % len(templates)
            print(template_idx)
            location_data = location_infos[template_idx]
            template_img = templates[template_idx].copy()
            prompt_data = prompts[template_idx]

            mixed_img = cut_generator(template_img, img, location_data["width_weight"], location_data["height_weight"],
                                      location_data["x_location"], location_data["y_location"])

            # Save the CutMix image
            output_filename = f"{location_data['template_name']}_{filename.split('.')[0]}_{location_data['task']}.png"
            specific_output_folder = os.path.join(output_folder, location_data['template_name'], location_data['task'])
            os.makedirs(specific_output_folder, exist_ok=True)
            output_image_path = os.path.join(specific_output_folder, output_filename)
            mixed_img.save(output_image_path)

            # Find the answer and prepare prompt content
            answer = find_answer(data_dict, filename)
            if not answer:
                answer = "Not found"

            question = prompt_data.get("question", "No question found").replace("<location>", location_data["task"])
            answer_format = prompt_data.get("answer_format1", "") + answer + prompt_data.get("answer_format2", "")


            prompt_content = {
                "id": f"{location_data['template_name']}_{filename.split('.')[0]}_{location_data['task']}",
                "image": os.path.abspath(output_image_path),
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer_format}
                ]
            }
            all_prompts.append(prompt_content)

def load_data_as_dict(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {annotation['id']: annotation['text'] for annotation in data['annotations']}

def find_answer(data_dict, image_name):
    image_id = image_name.split('.')[0]
    return data_dict.get(image_id, None)

# Main execution code
with open('fixed_front.json', 'r', encoding='utf-8') as f:
    fixed_front_json = json.load(f)

template_folder = fixed_front_json["template_folder"]
location_folder = fixed_front_json["location_folder"]
image_folder = fixed_front_json["image_folder"]
output_folder = fixed_front_json["output_folder"]
output_prompt_folder = fixed_front_json["output_prompt_folder"]
metadata_path = os.path.join(fixed_front_json["metadata_folder"], fixed_front_json["metadata_name"])
prompt_folder = fixed_front_json["prompt_folder"]

# Load metadata, templates, location info, and prompts
data_dict = load_data_as_dict(metadata_path)
templates, location_infos, prompts = load_templates_and_location_data(location_folder, template_folder, prompt_folder, fixed_front_json["data_info"])

# Initialize or load existing prompts
prompt_output_filename = os.path.join(output_prompt_folder, "prompts.json")
if os.path.exists(prompt_output_filename):
    with open(prompt_output_filename, 'r', encoding='utf-8') as outfile:
        all_prompts = json.load(outfile)
else:
    all_prompts = []

# Process all handwriting images
process_all_images_in_folder(image_folder, templates, location_infos, prompts, output_folder, data_dict, all_prompts)

# Save the updated prompts
with open(prompt_output_filename, 'w', encoding='utf-8') as outfile:
    json.dump(all_prompts, outfile, ensure_ascii=False, indent=4)
