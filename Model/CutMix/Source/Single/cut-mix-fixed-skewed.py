import cv2
import numpy as np
import json
import os


def cut_generator(img1, img2, width_weight, height_weight, x_location,
    y_location):
  """
  img1: 템플릿 이미지 (OpenCV image)
  img2: 삽입할 이미지 (OpenCV image)
  width_weight, height_weight: img2의 크기 비율
  x_location, y_location: img1에서의 삽입 위치 비율
  """
  # Get dimensions of the template image
  H, W = img1.shape[:2]

  # Calculate new size for img2 based on the weights
  new_w = int(W * width_weight)
  new_h = int(H * height_weight)

  # Resize the second image while keeping its transparency
  img2_resized = cv2.resize(img2, (new_w, new_h),
                            interpolation=cv2.INTER_CUBIC)

  # Calculate the insertion point on img1
  x_offset = int(W * x_location)
  y_offset = int(H * y_location)

  # Create a copy of img1 to paste the resized img2
  img1_copy = img1.copy()

  # Handle case when img2_resized has an alpha channel
  if img2_resized.shape[2] == 4:
    alpha_s = img2_resized[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
      img1_copy[y_offset:y_offset + new_h, x_offset:x_offset + new_w, c] = (
          alpha_s * img2_resized[:, :, c] +
          alpha_l * img1_copy[y_offset:y_offset + new_h,
                    x_offset:x_offset + new_w, c]
      )
  else:
    img1_copy[y_offset:y_offset + new_h,
    x_offset:x_offset + new_w] = img2_resized

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


def apply_perspective_skew(image_path, skew_angle, padding=200):
  image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
  if image is None:
    raise ValueError(f"Failed to load image from path: {image_path}")

  padded_image = add_padding(image, padding)
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

  print(f"Source points: {src_pts}")
  print(f"Destination points: {dst_pts}")
  print(f"Transformation matrix: \n{matrix}")
  print(f"New width: {new_w}, New height: {new_h}, Offset: {offset}")

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

  template_folder = json_data["template_folder"]
  image_folder = json_data["image_folder"]
  output_folder = json_data["output_folder"]
  data_info = json_data["data_info"]

  for info in data_info:
    template_name = info["template_name"]
    width_weight = float(info["width_weight"])
    height_weight = float(info["height_weight"])
    x_location = float(info["x_location"])
    y_location = float(info["y_location"])

    template_path = os.path.join(template_folder, template_name)
    template = cv2.imread(template_path)

    if template is None:
      print(f"Failed to load template image: {template_path}")
      continue

    skew_angle = detect_skew_angle(template)

    for image_name in os.listdir(image_folder):
      image_path = os.path.join(image_folder, image_name)
      skewed_image = apply_perspective_skew(image_path, -skew_angle)
      modified_image = crop_skewed_image(skewed_image);

      if skewed_image is None:
        print(f"Failed to apply skew for image: {image_path}")
        continue

      result = cut_generator(template, modified_image, width_weight,
                             height_weight, x_location, y_location)

      output_image_name = os.path.join(output_folder,
                                       f"{template_name}_{image_name}")
      cv2.imwrite(output_image_name, result)

read_json("fixed-skewed.json")