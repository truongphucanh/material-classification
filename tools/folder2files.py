# python ./tools/folder2files.py ./DS_GTOS/splits/test_split_1.txt

import os
import sys

def get_files_in(folder, label):
    list_files = []
    for root, dirs, files in os.walk(folder):
        for filename in [f for f in files if f.endswith(".jpg")]:
            jpg = os.path.join(root, filename).replace("\\","/")
            if filename not in list_files:
                list_files.append([jpg , label])
    return list_files

split_file = sys.argv[1]

with open(split_file) as f:
    content = f.readlines()
content = [x.strip() for x in content] 
new_content = ""
for line in content:
    folder = "./DS_GTOS/data/original/" + line.split()[0]
    label = line.split()[1]
    list_files = get_files_in(folder, label)
    for f in list_files:
        new_content = new_content + f[0] + " " + f[1] + "\n"
with open(split_file, 'w') as f:
    f.write(new_content)

