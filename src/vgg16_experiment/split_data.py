# python ./src/vgg16_experiment/split_data.py GTOS 0
# python ./src/vgg16_experiment/split_data.py GTOS 1
# python ./src/vgg16_experiment/split_data.py GTOS 2
# python ./src/vgg16_experiment/split_data.py GTOS 3
# python ./src/vgg16_experiment/split_data.py GTOS 4
# python ./src/vgg16_experiment/split_data.py GTOS 5
import os
import sys
from shutil import copyfile

# Parse arguments
dataset = sys.argv[1]
split = sys.argv[2]

train_split_dir = "./DS_{}/splits/train_split_{}.txt".format(dataset, split)
test_split_dir = "./DS_{}/splits/test_split_{}.txt".format(dataset, split)

train_folder = './DS_{}/data/split_{}/train'.format(dataset, split) 
test_folder = './DS_{}/data/split_{}/test'.format(dataset, split) 

# Create train and test folder
if not os.path.exists(train_folder):
    os.makedirs(train_folder)
if not os.path.exists(test_folder):
    os.makedirs(test_folder)

# Split train images
with open(train_split_dir) as f:
    content = f.readlines()
content = [x.strip() for x in content]
for line in content:
    if line == "":
        continue
    src_path = line.split()[0]
    class_name = src_path.split('/')[4]
    file_name = src_path.split('/')[6]
    dst_path = "{}/{}/{}".format(train_folder, class_name, file_name)
    class_folder = "{}/{}".format(train_folder, class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    if not os.path.exists(dst_path):
        copyfile(src_path, dst_path)
    print(dst_path)

# Split test images
with open(test_split_dir) as f:
    content = f.readlines()
content = [x.strip() for x in content]
for line in content:
    if line == "":
        continue
    src_path = line.split()[0]
    class_name = src_path.split('/')[4]
    file_name = src_path.split('/')[6]
    dst_path = "{}/{}/{}".format(test_folder, class_name, file_name)
    class_folder = "{}/{}".format(test_folder, class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    if not os.path.exists(dst_path):
        copyfile(src_path, dst_path)
    print(dst_path)

print("Done")
