import cv2
import numpy as np


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


# Example usage
first_image_path = 'sign.jpg'
image = cv2.imread(first_image_path)
skew_angle = detect_skew_angle(image)
print(f"Detected skew angle: {skew_angle} degrees")

skewed_image = apply_perspective_skew('source.JPG', -skew_angle)

# Crop the skewed image to remove padding and only keep the relevant area
cropped_skewed_image = crop_skewed_image(skewed_image)

# Save the cropped skewed image as PNG to include alpha channel
cv2.imwrite('cropped_skewed_image.png', cropped_skewed_image)
