import os
from PIL import Image, ImageDraw
import json
from collections import defaultdict
import random

def load_templates_and_location_data(location_folder, template_folder, prompt_folder, data_info):
    templates = []
    location_infos = []
    prompts = []

    for data_file in data_info:
        file_path = os.path.join(location_folder, data_file['data_file'])
        with open(file_path, 'r', encoding='utf-8') as f:
            location_data = json.load(f)

        template_img_path = os.path.join(template_folder, f"{location_data['template_name']}.jpg")
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

import os
import json

def append_to_json_file(filepath, new_data):
    # Check if the file exists
    if os.path.exists(filepath):
        # If it exists, load the existing data
        with open(filepath, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)
        # Append the new data to the existing data
        existing_data.extend(new_data)
    else:
        # If the file does not exist, start with the new data
        existing_data = new_data
    
    # Save the combined data back to the file
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, ensure_ascii=False, indent=4)

def process_words_with_templates(word_allocations, templates, location_infos, prompts, output_folder, data_dict, word_image_folder, output_prompt_folder):
    # Ensure the output_prompt_folder exists
    os.makedirs(output_prompt_folder, exist_ok=True)
    
    # Extract train, eval, and test words
    train_words = word_allocations['train']
    eval_words = word_allocations['eval']
    test_words = word_allocations['test']

    # Initialize separate lists for train, eval, and test prompts
    train_prompts = []
    eval_prompts = []
    test_prompts = []

    # Round-robin for templates
    def apply_cutmix(words, dataset_name, prompt_list):
        total_words = len(words)
        progress_step = total_words // 10  # To track 10% progress increments

        for i, word in enumerate(words):
            # Print progress every 10%
            if (i + 1) % progress_step == 0 or (i + 1) == total_words:
                progress = ((i + 1) / total_words) * 100
                print(f"{dataset_name} {int(progress)}% 완료")

            template_idx = i % len(templates)  # Round-robin 
            template_img = templates[template_idx].copy()
            location_data = location_infos[template_idx]
            prompt_data = prompts[template_idx]

            # Fetch the image for the word
            word_img_path = os.path.join(word_image_folder, f"{word}.png")
            word_img = Image.open(word_img_path)

            # Apply CutMix
            mixed_img = cut_generator(template_img, word_img, location_data["width_weight"],
                                      location_data["height_weight"], location_data["x_location"],
                                      location_data["y_location"])

            # Save the CutMix image as JPG to reduce size
            output_filename = f"{location_data['template_name']}_{location_data['task']}_{random.randint(10000, 99999)}.jpg"
            specific_output_folder = os.path.join(output_folder, location_data['template_name'], location_data['task'])
            os.makedirs(specific_output_folder, exist_ok=True)
            output_image_path = os.path.join(specific_output_folder, output_filename)
            mixed_img.save(output_image_path, "JPEG", quality=90)  # Save as JPG with quality reduction

            # Prepare prompt content
            question = prompt_data.get("question", "No question found")
            answer = find_answer(data_dict, word)
            answer_format = prompt_data.get("answer_format1", "") + answer + prompt_data.get("answer_format2", "")

            # Create prompt content
            prompt_content = {
                "id": f"{location_data['template_name']}_{word}_{location_data['task']}",
                "image": os.path.abspath(output_image_path),
                "conversations": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer_format}
                ]
            }
            # Append to the respective prompt list
            prompt_list.append(prompt_content)

    # Process words for train, eval, and test datasets
    apply_cutmix(train_words, "train", train_prompts)
    apply_cutmix(eval_words, "eval", eval_prompts)
    apply_cutmix(test_words, "test", test_prompts)

    # Append or save separate JSON files for train, eval, and test
    append_to_json_file(os.path.join(output_prompt_folder, "train.json"), train_prompts)
    append_to_json_file(os.path.join(output_prompt_folder, "eval.json"), eval_prompts)
    append_to_json_file(os.path.join(output_prompt_folder, "test.json"), test_prompts)

    print("Processing complete and train, eval, test prompts saved.")


def load_data_as_dict(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {annotation['id']: annotation['text'] for annotation in data['annotations']}

def find_answer(data_dict, image_name):
    image_id = image_name.split('.')[0]
    return data_dict.get(image_id, "Not Found")

def count_words(metadata, word_folder_path):
    word_annotations = defaultdict(list)
    
    # image_filenames 리스트를 set으로 변환하여 탐색 속도 향상
    cnt = 0
    image_filenames = {filename.split('.')[0] for filename in os.listdir(word_folder_path)}
    
    # Iterate through the metadata and only count words that have corresponding image files
    for annotation_id, word in metadata.items():
        word = word.strip()  # Clean up whitespace
        if annotation_id in image_filenames:  # Check if the annotation_id has a corresponding image
            cnt+=1
            if word:  # Ensure word is not empty
                word_annotations[word].append(annotation_id)  # Group by word
    print(f"총 단어 수: {cnt}")
    return word_annotations

def get_specific_numbers(word_annotations, train_data_number, eval_data_number, test_data_number):
    word_distribution = {}
    word_count = len(word_annotations)
    
    # 정수 출력 시 f-string 사용
    print(f"word_count: {word_count}")
    
    train_count = train_data_number // word_count
    train_offset = train_data_number % word_count
    
    print(f"train_count: {train_count}")
    print(f"train_offset: {train_offset}")

    for i, word in enumerate(word_annotations):
        word_distribution[word] = {}  # 기존 데이터를 덮어쓰지 않도록 빈 딕셔너리 생성
        if i < train_offset:
            word_distribution[word]["train"] = train_count + 1
        else:
            word_distribution[word]["train"] = train_count
            
    eval_count = (train_data_number + eval_data_number) // word_count
    eval_offset = (train_data_number + eval_data_number) % word_count
    
    print(f"eval_count: {eval_count}")
    print(f"eval_offset: {eval_offset}")
    
    for i, word in enumerate(word_annotations):
        if i < eval_offset:
            word_distribution[word]["eval"] = eval_count + 1 - word_distribution[word]["train"]
        else:
            word_distribution[word]["eval"] = eval_count - word_distribution[word]["train"]
            
    test_count = (train_data_number + eval_data_number + test_data_number) // word_count
    test_offset = (train_data_number + eval_data_number + test_data_number) % word_count
    
    print(f"test_count: {test_count}")
    print(f"test_offset: {test_offset}")
    
    for i, word in enumerate(word_annotations):
        if i < test_offset:
            word_distribution[word]["test"] = test_count + 1 - word_distribution[word]["train"] - word_distribution[word]["eval"]
        else:
            word_distribution[word]["test"] = test_count - word_distribution[word]["train"] - word_distribution[word]["eval"]

    return word_distribution


def allocate_data(word_annotations, word_distribution):
    # Data structure to store the final allocations
    allocations = {
        "train": [],
        "eval": [],
        "test": []
    }

    # Iterate through each word and its distribution requirements
    for word, counts in word_distribution.items():
        # Fetch the annotations for this word
        annotations = word_annotations.get(word, [])

        # Check if there are enough annotations for the required allocation
        total_required = counts["train"] + counts["eval"] + counts["test"]
        if len(annotations) < total_required:
            print(f"Error: Not enough annotations for the word '{word}'")
            print(f"Required: {total_required}, Available: {len(annotations)}")
            print(f"Word distribution: {counts}")
            # Stop the function by raising an exception or returning early
            raise ValueError(f"Insufficient annotations for word '{word}'")

        # Shuffle annotations to ensure random selection
        random.shuffle(annotations)

        # Determine the index limits for each data type
        train_idx = counts["train"]
        eval_idx = train_idx + counts["eval"]
        test_idx = eval_idx + counts["test"]

        # Allocate annotations based on the indices calculated
        allocations["train"].extend(annotations[:train_idx])
        allocations["eval"].extend(annotations[train_idx:eval_idx])
        allocations["test"].extend(annotations[eval_idx:test_idx])

    return allocations


# Main execution code
def main():
    # Load the fixed_front.json configuration file
    with open('fixed_front.json', 'r', encoding='utf-8') as f:
        fixed_front_json = json.load(f)

    # Extract necessary paths from the JSON configuration
    template_folder = fixed_front_json["template_folder"]
    location_folder = fixed_front_json["location_folder"]
    image_folder = fixed_front_json["image_folder"]
    output_folder = fixed_front_json["output_folder"]
    output_prompt_folder = fixed_front_json["output_prompt_folder"]
    metadata_path = os.path.join(fixed_front_json["metadata_folder"], fixed_front_json["metadata_name"])
    prompt_folder = fixed_front_json["prompt_folder"]
    os.makedirs(output_prompt_folder, exist_ok=True)

    # Define the number of data points for train, eval, and test datasets
    train_data_number = 32000
    eval_data_number = 8000
    test_data_number = 10000

    # Step 1 메타 데이터 로드
    data_dict = load_data_as_dict(metadata_path)

    # Step 2 단어 별 데이터 분포 파악 (폴더에 있는 데이터만)
    word_annotations = count_words(data_dict, image_folder)

    # Step 3
    word_distribution = get_specific_numbers(word_annotations, train_data_number, eval_data_number, test_data_number)

    # Step 4
    word_allocations = allocate_data(word_annotations, word_distribution)

    # Step 5
    templates, location_infos, prompts = load_templates_and_location_data(location_folder, template_folder, prompt_folder, fixed_front_json["data_info"])

    # Step 6
    process_words_with_templates(word_allocations, templates, location_infos, prompts, output_folder, data_dict, image_folder, output_prompt_folder)


if __name__ == "__main__":
    main()
