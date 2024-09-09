import json
import os
import cv2
import numpy as np



def masking(img_path, width, height):
  image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
  if image is None:
    raise ValueError(f"Failed to load image from path: {img_path}")
  H, W = image.shape[:2]
  # Calculate scaling ratios for width and height
  x_ratio = width / W
  y_ratio = height / H

  # Determine the appropriate scaling ratio to ensure the image fits within the specified area
  ratio = min(x_ratio, y_ratio)

  # Calculate new dimensions based on the selected ratio
  new_w = int(W * ratio)
  new_h = int(H * ratio)
  resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
  # 6. 흰색 배경 이미지 생성
  result_image = np.full((height, width, 3), (255, 255, 255),
                         dtype=np.uint8)

  # 7. 리사이즈된 이미지를 중앙에 배치
  x_offset = (width - new_w) // 2
  y_offset = (height - new_h) // 2

  result_image[y_offset:y_offset + new_h,
  x_offset:x_offset + new_w] = resized_image

  return result_image

def cut_generator(img1, img2, x_pixels, y_pixels, x_offset, y_offset):
  """
  img1: 템플릿 이미지 (OpenCV image)
  img2: 삽입할 이미지 (OpenCV image)
  x_pixels, y_pixels: img2가 들어갈 사각형의 폭과 높이 (픽셀 단위)
  x_offset, y_offset: img1에서의 삽입 위치 (픽셀 단위)
  """
  H, W = img2.shape[:2]
  #
  # # Calculate scaling ratios for width and height
  # x_ratio = x_pixels / W
  # y_ratio = y_pixels / H
  #
  # # Determine the appropriate scaling ratio to ensure the image fits within the specified area
  # ratio = min(x_ratio, y_ratio)
  #
  # # Calculate new dimensions based on the selected ratio
  # new_w = int(W * ratio)
  # new_h = int(H * ratio)

  # Resize the second image while keeping its transparency
  # img2_resized = cv2.resize(img2, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

  # Create a copy of img1 to paste the resized img2
  img1_copy = img1.copy()

  # Handle case when img2_resized has an alpha channel
  if img2.shape[2] == 4:
    # Separate alpha channel from img2
    alpha_s = img2[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):  # Loop over the color channels
      img1_copy[y_offset:y_offset + H, x_offset:x_offset + W, c] = (
          alpha_s * img2[:, :, c] +
          alpha_l * img1_copy[y_offset:y_offset + H,
                    x_offset:x_offset + W, c]
      )
  else:
    # If no alpha channel, just paste the resized image
    img1_copy[y_offset:y_offset + H,
    x_offset:x_offset + W] = img2

  return img1_copy

def detect_skew_angle(image):
    # 이미지 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 이진화 (임계값 처리)
    _, binary = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 모폴로지 연산을 통해 노이즈 제거
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 에지 검출
    edges = cv2.Canny(cleaned, 50, 150)

    # 허프 변환을 이용하여 직선 검출
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return 0

    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = (theta * 180 / np.pi) - 90
        angles.append(angle)

    if angles:
        # 평균 각도를 계산하여 skew 각도로 반환
        return np.mean(angles)

    return 0


def add_padding(image, padding=200):
  h, w = image.shape[:2]
  # Create a transparent padded image
  padded_image = np.zeros((h + 2 * padding, w + 2 * padding, 4), dtype=np.uint8)
  padded_image[padding:padding + h, padding:padding + w, :3] = image
  padded_image[:, :, 3] = 255  # Fully opaque alpha channel
  return padded_image


def calculate_bounding_box(src_pts, matrix):
  transformed_pts = cv2.perspectiveTransform(np.array([src_pts]), matrix)[0]
  min_x, min_y = np.min(transformed_pts, axis=0)
  max_x, max_y = np.max(transformed_pts, axis=0)

  new_width = int(max_x - min_x)
  new_height = int(max_y - min_y)

  # Ensure non-negative dimensions
  new_width = max(new_width, 1)
  new_height = max(new_height, 1)

  # Ensure offset is non-negative
  offset_x = max(0, int(-min_x))
  offset_y = max(0, int(-min_y))

  return new_width, new_height, (offset_x, offset_y)


def apply_perspective_skew(masked_image, skew_angle, padding=200):
  padded_image = add_padding(masked_image, padding)
  (h, w) = padded_image.shape[:2]

  angle_radians = np.deg2rad(skew_angle)
  skew_factor = np.tan(angle_radians)

  src_pts = np.float32([
    [0, 0],
    [w - 1, 0],
    [0, h - 1],
    [w - 1, h - 1]
  ])

  dst_pts = np.float32([
    [0, 0],
    [w, skew_factor * h],
    [0, h],
    [w, h + skew_factor * h]
  ])

  matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

  new_w, new_h, offset = calculate_bounding_box(src_pts, matrix)

  # print(f"Source points: {src_pts}")
  # print(f"Destination points: {dst_pts}")
  # print(f"Transformation matrix: \n{matrix}")
  # print(f"New width: {new_w}, New height: {new_h}, Offset: {offset}")

  if new_w <= 0 or new_h <= 0:
    raise ValueError("Invalid new width or height calculated.")

  # Create a blank (transparent) image with an alpha channel
  skewed_image = np.zeros((new_h, new_w, 4), dtype=np.uint8)

  # Apply the perspective transform to the padded image
  skewed_image_rgb = cv2.warpPerspective(padded_image, matrix, (new_w, new_h),
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0, 0))

  # Set RGB channels
  skewed_image[:, :, :3] = skewed_image_rgb[:, :, :3]

  # Set alpha channel based on original image
  skewed_image[:, :, 3] = skewed_image_rgb[:, :, 3]

  return skewed_image


def crop_skewed_image(skewed_image):
  # Convert the image to grayscale to find the bounding box
  gray_image = cv2.cvtColor(skewed_image, cv2.COLOR_RGBA2GRAY)

  # Find contours to locate the bounding box of the skewed content
  contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)

  if not contours:
    raise ValueError("No contours found in the skewed image.")

  # Get the largest contour
  contour = max(contours, key=cv2.contourArea)

  # Create a mask with the largest contour
  mask = np.zeros_like(gray_image)
  cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

  # Find the bounding box of the contour
  x, y, w, h = cv2.boundingRect(contour)

  # Crop the image based on the mask
  cropped_image = cv2.bitwise_and(skewed_image, skewed_image, mask=mask)
  cropped_image = cropped_image[y:y + h, x:x + w]

  return cropped_image



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
    x_location = float(location_data["x_location"])
    y_location = float(location_data["y_location"])
    task = location_data["task"]
    prompt_file = location_data["prompt_file"]
    specific_output_folder = os.path.join(output_folder, template_name, task)

    if not os.path.exists(specific_output_folder):
      os.makedirs(specific_output_folder)

    if not os.path.exists(output_prompt_folder):
      os.makedirs(output_prompt_folder)

    template_path = os.path.join(template_folder, f"{template_name}.png")
    template = cv2.imread(template_path)

    if template is None:
      print(f"Failed to load template image: {template_path}")
      continue

    skew_angle = detect_skew_angle(template)
    print("template_name:", template_name, "skew:", skew_angle)

    # Get image dimensions
    template_height, template_width = template.shape[:2]

    # Convert ratio coordinates to pixel values
    x_pixel_location = int(x_location * template_width)
    y_pixel_location = int(y_location * template_height)
    rectangle_width = int(width_weight * template_width)
    rectangle_height = int(height_weight * template_height)

    for image_name in os.listdir(image_folder):
      image_path = os.path.join(image_folder, image_name)
      masked_image = masking(image_path, rectangle_width, rectangle_height)
      skewed_image = apply_perspective_skew(masked_image, skew_angle)
      modified_image = crop_skewed_image(skewed_image)


      if skewed_image is None:
        print(f"Failed to apply skew for image: {image_path}")
        continue

      result = cut_generator(template, modified_image, rectangle_width,
                             rectangle_height, x_pixel_location, y_pixel_location)

      output_image_name = os.path.join(specific_output_folder,
                                       f"{template_name}_{image_name.split('.')[0]}_{task}.png")
      cv2.imwrite(output_image_name, result)

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
                        "id": f"{template_name}_{image_name.split('.')[0]}_{task}",
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
read_json("fixed-skewed.json")