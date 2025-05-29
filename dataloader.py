import tarfile
import os
from sklearn.model_selection import train_test_split
import shutil

base_dir = 'data/caltech-101/101_ObjectCategories'

# 解压缩函数
def extract_files(input_path, output_path):
    with tarfile.open(input_path, 'r:gz') as tar:
        tar.extractall(path=output_path)

# 如果文件未解压，解压缩文件
if not os.path.exists(base_dir):
    extract_files('data/caltech-101/caltech-101/101_ObjectCategories.tar.gz', 'data/caltech-101/')

def split_dataset(base_dir, train_dir, val_dir, test_dir, val_size=0.1, test_size=0.2):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    categories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Number of categories: {len(categories)}")

    for category in categories:
        category_path = os.path.join(base_dir, category)
        
        images = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        
        if len(images) == 0:
            print(f"No images found in category: {category}")
            continue
        
        # Split images
        train_val_images, test_images = train_test_split(images, test_size=test_size, random_state=42)
        train_images, val_images = train_test_split(train_val_images, test_size=val_size / (1 - test_size), random_state=42)

        for image_set, target_dir in [(train_images, train_dir), (val_images, val_dir), (test_images, test_dir)]:
            target_category_dir = os.path.join(target_dir, category)
            os.makedirs(target_category_dir, exist_ok=True)
            for image in image_set:
                shutil.move(os.path.join(category_path, image), os.path.join(target_category_dir, image))


# 用法示例
train_dir = base_dir + '/train'
val_dir = base_dir + '/val'
test_dir = base_dir + '/test'
split_dataset(base_dir, train_dir, val_dir, test_dir)

