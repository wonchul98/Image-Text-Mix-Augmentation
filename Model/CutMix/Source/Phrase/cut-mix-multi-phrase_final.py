import json
import os
import cv2
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager


def masking(template, x, y, rectangle_width, height_per_line_px, max_lines, padding_px):
  template_copy = template.copy()
  color = (255, 255, 255)  # 흰색
  thickness = -1
  for i in range(max_lines):
    start_point = (x, y)  # 시작 점
    end_point = (x + rectangle_width, y + height_per_line_px)  # 끝 점
    cv2.rectangle(template_copy, start_point, end_point, color, thickness)
    y = y + height_per_line_px + padding_px
  return template_copy


def cut_phrase(template, image, x, y, start, end, height_per_line_px, padding_px):
  """
  img1: 템플릿 이미지 (OpenCV image)
  img2: 삽입할 이미지 (OpenCV image)
  x, y : 삽입 좌표
  start: 문장 시작 x 좌표 (고정)
  """

  H, W = image.shape[:2]

  ratio = height_per_line_px / H

  # print("height_per_line_px", height_per_line_px)
  # print("ratio", ratio)

  # Calculate new dimensions based on the selected ratio
  new_w = int(W * ratio)
  new_h = int(H * ratio)

  # Resize the second image while keeping its transparency
  img2_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

  while img2_resized.shape[1] > 0:
    # Calculate how much of the image fits in the current line
    space_remaining = end - x
    # print("end", end)
    # print("remaining", space_remaining)
    if img2_resized.shape[1] <= space_remaining:
      # If the whole image fits in the remaining space
      template[y:y + img2_resized.shape[0],
      x:x + img2_resized.shape[1]] = img2_resized
      x += img2_resized.shape[1]
      break
    else:
      # If the image extends beyond the current line, cut and paste the portion that fits
      template[y:y + img2_resized.shape[0], x:end] = img2_resized[:, :space_remaining]

      # Move to the next line
      img2_resized = img2_resized[:, space_remaining:]
      x = start
      y += height_per_line_px + padding_px

  return template, x, y





def read_json(filename, epoch):
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
  count_per_temp = epoch // len(data_info)
  count = 0

  prompt_output_filename = os.path.join(output_prompt_folder, "prompts.json")
  validation_output_filename = os.path.join(output_prompt_folder, "validation.json")
  image_list = os.listdir(image_folder)
  random_indices = random.sample(range(len(image_list)), epoch)

  if os.path.exists(prompt_output_filename):
    with open(prompt_output_filename, 'r', encoding='utf-8') as outfile:
      all_prompts = json.load(outfile)
  else:
    all_prompts = []

  if os.path.exists(validation_output_filename):
    with open(validation_output_filename, 'r', encoding='utf-8') as outfile:
      validations = json.load(outfile)
  else:
    validations = []

  data_dict = load_data_as_dict(json_path)

  with Manager() as manager:
    all_prompts = manager.list(all_prompts)
    validations = manager.list(validations)
    for info in data_info:
      data_file_path = os.path.join(location_folder, info["data_file"])
      with open(data_file_path, 'r', encoding='utf-8') as df:
        location_data = json.load(df)
        process_template(location_data, output_folder, output_prompt_folder, template_folder,
                         image_folder, image_list, count_per_temp, data_dict, prompt_folder,
                         all_prompts, validations, random_indices, count)
        count += count_per_temp
      # ListProxy를 일반 리스트로 변환 후 저장
    with open(prompt_output_filename, 'w', encoding='utf-8') as outfile:
      json.dump(list(all_prompts), outfile, ensure_ascii=False, indent=4)
    with open(validation_output_filename, 'w', encoding='utf-8') as outfile:
      json.dump(list(validations), outfile, ensure_ascii=False, indent=4)


def process_template(location_data, output_folder, output_prompt_folder, template_folder,
                         image_folder, image_list, count_per_temp, data_dict,
                        prompt_folder, all_prompts, validations, random_indices, count):
    template_name = location_data["template_name"]
    width_weight = float(location_data["width_weight"])
    height_weight = float(location_data["height_weight"])
    x_location = float(location_data["x_location"])
    y_location = float(location_data["y_location"])
    height_per_line = float(location_data["height_per_line"])
    max_lines = int(location_data["max_lines"])
    padding_px = int(location_data["padding"])
    task = location_data["task"]
    prompt_file = location_data["prompt_file"]
    specific_output_folder = os.path.join(output_folder, template_name, task)

    if not os.path.exists(specific_output_folder):
      os.makedirs(specific_output_folder)

    if not os.path.exists(output_prompt_folder):
      os.makedirs(output_prompt_folder)

    template_path = os.path.join(template_folder, template_name + ".png")
    template = cv2.imread(template_path)

    if template is None:
      print(f"Failed to load template image: {template_path}")
      return

    # Get image dimensions
    template_height, template_width = template.shape[:2]

    # Convert ratio coordinates to pixel values
    x_pixel_location = int(x_location * template_width)
    y_pixel_location = int(y_location * template_height)
    rectangle_width = int(width_weight * template_width)
    rectangle_height = int(height_weight * template_height)
    height_per_line_px = int(height_per_line * template_height)
    end = x_pixel_location + rectangle_width

    total_length = rectangle_width * max_lines
    template_masked = masking(template, x_pixel_location, y_pixel_location, rectangle_width, height_per_line_px, max_lines, padding_px)
    prompt_file_path = os.path.join(prompt_folder, prompt_file)

    with open(prompt_file_path, 'r', encoding='utf-8') as pf:
      prompt_data = json.load(pf)

    question = prompt_data.get("question", "No question found")
    answer_format1 = prompt_data.get("answer_format1", "")
    answer_format2 = prompt_data.get("answer_format2", "")
    # print(template_name, count)

    # image_list len : 90,000
    # count_per_temp : 10000
    # range : 0
    workers = 32
    batches = []
    batchSize = count_per_temp // workers
    batchStart = count
    #  이 부분 문제있음
    for b in range(workers):
      batchEnd = batchStart + batchSize
      if b == (workers - 1):
        batches.append((batchStart, count + count_per_temp))
      else:
        batches.append((batchStart, batchEnd))
      batchStart += batchSize
    print(batches)

    with ProcessPoolExecutor(max_workers = workers) as executor:
      for (batchStart, batchEnd) in batches:
        executor.submit(generate_images, batchStart, batchEnd, image_list,
                        image_folder, template_masked,
                        x_pixel_location,
                        y_pixel_location, end, specific_output_folder,
                        template_name,
                        data_dict, answer_format1, answer_format2, question,
                        validations, all_prompts,
                        total_length, height_per_line_px, padding_px, random_indices)

def generate_images(batchStart, batchEnd, image_list,
                          image_folder, template_masked,
                          x_coord, y_coord, end, specific_output_folder,
                          template_name,
                          data_dict, answer_format1, answer_format2, question,
                          validations, all_prompts,
                          total_length, height_per_line_px, padding_px, random_indices):
  for idx in range(batchStart, batchEnd):
    len_sum = 0
    answer = ""
    result = template_masked.copy()
    interval = 1
    x_start = x_coord
    x = x_coord
    y = y_coord
    curr = random_indices[idx]

    while len_sum < total_length:
      image_name = image_list[curr]
      # print(image_name)
      image_path = os.path.join(image_folder, image_name)
      image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

      if image is None:
        print(f"Failed to load image: {image_path}")
        continue

      (h, w) = image.shape[:2]
      img_length = w * (height_per_line_px / h)
      len_sum += img_length

      if len_sum > total_length:
        break

      # Cut and paste the image
      result, x, y = cut_phrase(result, image, x, y, x_start, end, height_per_line_px, padding_px)
      answer += find_answer(data_dict, image_name)
      curr = (curr + interval) % len(image_list)

    # print("answer", answer)
    output_image_name = os.path.join(specific_output_folder,
                                     f"{template_name.split('.')[0]}_phrase_{random_indices[idx]}.jpg")
    cv2.imwrite(output_image_name, result, [cv2.IMWRITE_JPEG_QUALITY, 90])

    generated_answer = f"{answer_format1}{answer}{answer_format2}"

    # 프롬프트 생성
    prompt_content = {
      "id": f"{template_name.split('.')[0]}_phrase_{random_indices[idx]}",
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
    if idx % 5 == 0:
      validations.append(prompt_content)
    else:
      all_prompts.append(prompt_content)






def load_data_as_dict(json_path):
  with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

  data_dict = {annotation['id']: annotation['text'] for annotation in
               data['annotations']}
  return data_dict


def find_answer(data_dict, image_name):
  image_id = image_name.split('.')[0]
  return data_dict.get(image_id, None)


if __name__ == '__main__':
  read_json("fixed_phrase.json", 10000)
