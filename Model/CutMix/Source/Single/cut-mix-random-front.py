import json
import os
from PIL import Image
import numpy as np


def cut_generator(img1, img2, width_weight, height_weight):
  # Convert PIL images to NumPy arrays
  img1_array = np.array(img1)

  # Resize img2 based on width_weight and height_weight
  W, H = img1.size
  new_w = int(W * width_weight)
  new_h = int(H * height_weight)
  img2_resized = img2.resize((new_w, new_h))
  img2_resized_array = np.array(img2_resized)

  # Determine random position to place img2_resized within img1
  max_x = W - new_w
  max_y = H - new_h
  if max_x < 0 or max_y < 0:
    raise ValueError("Resized image is larger than the base image dimensions.")

  x_offset = np.random.randint(0, max_x + 1)
  y_offset = np.random.randint(0, max_y + 1)

  # Create a copy of img1 and place the resized img2 on it
  image = img1_array.copy()
  image[y_offset:y_offset + new_h,
  x_offset:x_offset + new_w] = img2_resized_array

  # Convert back to PIL Image
  return Image.fromarray(image)


def read_json(filename):
  """
  JSON 파일을 읽고 각 템플릿당 이미지 합성
  """
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
    width_weight = float(location_data["width_weight"])
    height_weight = float(location_data["height_weight"])
    task = location_data["task"]
    prompt_file = location_data["prompt_file"]
    specific_output_folder = os.path.join(output_folder, template_name, task)

    if not os.path.exists(specific_output_folder):
      os.makedirs(specific_output_folder)

    if not os.path.exists(output_prompt_folder):
      os.makedirs(output_prompt_folder)

    template_path = os.path.join(template_folder, f"{template_name}.png")
    template = Image.open(template_path).convert('RGB').resize((256, 256))

    if template is None:
      print(f"Failed to load template image: {template_path}")
      continue

    for image_name in os.listdir(image_folder):
      image_path = os.path.join(image_folder, image_name)
      image = Image.open(image_path).convert('RGB').resize((256, 256))



      if image is None:
        print(f"Failed to load image: {image_path}")
        continue

      result = cut_generator(template, image, width_weight, height_weight);

      output_image_name = os.path.join(specific_output_folder,
                                       f"{template_name.split('.')[0]}_{image_name.split('.')[0]}_{task}.png")
      result.save(output_image_name)

      answer = find_answer(data_dict, image_name)
      if not answer:
        answer = "Not found"

      prompt_file_path = os.path.join(prompt_folder, prompt_file)

      with open(prompt_file_path, 'r', encoding='utf-8') as pf:
        prompt_data = json.load(pf)

      question = prompt_data.get("question", "No question found")
      answer_format1 = prompt_data.get("answer_format1", "")
      answer_format2 = prompt_data.get("answer_format2", "")
      generated_answer = f"{answer_format1}{answer}{answer_format2}"

      # 프롬프트 생성
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

  data_dict = {annotation['id']: annotation['text'] for annotation in
               data['annotations']}
  return data_dict


def find_answer(data_dict, image_name):
  image_id = image_name.split('.')[0]
  return data_dict.get(image_id, None)

read_json("random_front.json")