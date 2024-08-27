from PIL import Image
import numpy as np

def rand_bbox(lam):
    W = 256
    H = 256
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    print(bbx1, bby1, bbx2, bby2)
    return bbx1, bby1, bbx2, bby2

def cut_generator(img1, img2, lam=0.5):
    bbx1, bby1, bbx2, bby2 = rand_bbox(lam)
    print(img1.shape)
    print(img2.shape)
    
    # img2를 박스 영역에 맞게 축소하면서 원본 정보 유지
    img2_resized = img2.copy()
    aspect_ratio = min((bbx2-bbx1)/img2.shape[1], (bby2-bby1)/img2.shape[0])
    new_w = int(img2.shape[1] * aspect_ratio)
    new_h = int(img2.shape[0] * aspect_ratio)

    img2_resized = np.array(Image.fromarray(img2_resized).resize((new_w, new_h)))
    
    # 중앙에 배치하여 이미지를 삽입
    x_offset = bbx1 + (bbx2 - bbx1 - new_w) // 2
    y_offset = bby1 + (bby2 - bby1 - new_h) // 2

    image = img1.copy()
    image[y_offset:y_offset+new_h, x_offset:x_offset+new_w, :] = img2_resized
    
    return image



# Test용 코드 
dir1 = 'source.JPG'
dir2 = '문가네_진국_170628_0002.jpg' 
img1 = np.array(Image.open(dir2).convert('RGB').resize((256,256)))
img2 = np.array(Image.open(dir1).convert('RGB').resize((256,256)))

img = cut_generator(img1, img2, lam=0.5)
Image.fromarray(img).show()