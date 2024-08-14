import os
import random
import shutil

train_set_path = 'D:/Data/Cat-Dog/train1'
val_set_path = 'D:/Data/Cat-Dog/val'


def create_valset(source_path, target_path, ratio=0.2):
    # 创建验证集
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        print("Val set directory created.")
    else:
        print("Val set directory already exists, please check.")
        return
    for file in os.listdir(source_path):
        file_path = os.path.join(source_path, file)
        if random.random() < ratio:
            shutil.move(file_path, os.path.join(target_path, file))


def create_dataset_txt(set_path, txt_path):
    labels = ['cat', 'dog']
    with open(txt_path, 'w') as f:
        for file in os.listdir(set_path):
            file_path = os.path.join(set_path, file)
            label: int = labels.index(file.split('.')[0])
            f.write(file_path + '\t' + str(label) + '\n')
    print("Dataset txt file created.")


if __name__ == '__main__':
    create_valset(train_set_path, val_set_path, ratio=0.2)
    create_dataset_txt(train_set_path, './train.txt')
    create_dataset_txt(val_set_path, './val.txt')
