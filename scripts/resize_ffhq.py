import os
import cv2
import tqdm

source_dir = "/mnt/petrelfs/share_data/wangyaohui/datasets/ffhq/images1024x1024"
target_dir = "/mnt/petrelfs/wangyufei/datasets/ffhq/images256x256"
image_size=(256,256)

def resize_images(img_path):
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, image_size)
    return resized_img

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
            img_path = os.path.join(root, file)
            resized_img = resize_images(img_path)
            
            # create the corresponding target directory
            target_path = os.path.join(target_dir, root[len(source_dir)+1:])
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            
            # save the resized image
            target_img_path = os.path.join(target_path, file)
            cv2.imwrite(target_img_path, resized_img)
