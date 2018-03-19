# Format: python ./tools/split.py <dataset_name> <n_splits> <train_rate>
# Example: python ./tools/split.py FMD 5 0.8

import sys
import os

def split(dataset_name, train_rate, out_file_train, out_file_test):
    data_folder = './DS_{}/data/original'.format(dataset_name)
    got_folder = []
    for root, dirs, files in os.walk(data_folder):
        print(dirs)
        for filename in [f for f in files if f.endswith(".jpg")]:
            jpg = os.path.join(root, filename).replace("\\", "/")

def main(argv):
    if len(argv) < 4:
        print("Missing arguments. Please try again.")
        print("Format: python ./tools/split.py <dataset_name> <n_splits> <train_rate>")
        print("Example: python ./tools/split.py FMD 5 0.8")
        return
    dataset_name = argv[1]
    n_splits = int(argv[2])
    train_rate = argv[3]
    for i in range(1, n_splits + 1):
        out_file_train = "./DS_{}/splits/trainlist{}.txt".format(dataset_name, i)
        out_file_test = "./DS_{}/splits/testlist{}.txt".format(dataset_name, i)
        split(dataset_name, train_rate, out_file_train, out_file_test)

if __name__ == '__main__':
    main(sys.argv)