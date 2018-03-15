# python ./tools/get_data_filenames.py <dataset_name> <dataset_type>
# python ./tools/get_data_filenames.py FMD original
# output file: ./<dataset_name>_dataset_type.txt

import os
import sys

dataset_name = sys.argv[1]
dataset_type = sys.argv[2]

data_folder = './DS_{}/data/{}'.format(dataset_name, dataset_type)
for root, dirs, files in os.walk(data_folder):
    for filename in [f for f in files if f.endswith(".jpg")]:
        jpg = os.path.join(root, filename).replace("\\","/")
        print(jpg)
        with open('./{}_{}.txt'.format(dataset_name, dataset_type), 'a') as f:
            f.write('{}\n'.format(jpg))
