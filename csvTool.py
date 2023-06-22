import csv
import os

folder_path = 'gestures'
csv_file = './gestures/annotations.csv'

# 檢查CSV檔案是否存在，如果不存在則創建新的檔案
if not os.path.isfile(csv_file):
    with open(csv_file, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Label'])  # 寫入標題行

# 讀取資料夾中的圖片檔案並將其加入到CSV檔案中
with open(csv_file, 'a') as file:
    writer = csv.writer(file)
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)
            gesture_label = 'your_label'  # 根據您的需要指定手勢的標籤
            writer.writerow([image_path, gesture_label])
