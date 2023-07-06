import cv2
import os

def preprocess_images(image_paths):
    preprocessed_images = []
    for image_path in image_paths:
        # 使用 cv2.imread 讀取圖片
        image = cv2.imread(image_path)
        
        # 將圖片尺寸調整為指定大小
        desired_width = 512
        desired_height = 512
        image = cv2.resize(image, (desired_width, desired_height))
        
        # 正規化圖片像素值到 0 到 1 的範圍
        image = image / 255.0
        
        preprocessed_images.append(image)
    return preprocessed_images

folder_path = 'original_images'
output_folder = 'preprocessed_images'

# 讀取資料夾中的圖片檔案
image_paths = []
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg'):
        image_path = os.path.join(folder_path, filename)
        image_paths.append(image_path)

# 呼叫圖片預處理函數
preprocessed_images = preprocess_images(image_paths)

# 檢查輸出資料夾是否存在，如果不存在則創建新的資料夾
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 保存處理後的圖片到新資料夾
for i, image in enumerate(preprocessed_images):
    # 將像素值範圍調整為 0 到 255 的整數型態
    image = (image * 255).astype('uint8')

    output_path = os.path.join(output_folder, f'preprocessed_{i}.jpg')
    cv2.imwrite(output_path, image)
    print(f'保存圖片 {output_path}')

print('圖片處理完成並儲存到新資料夾。')
