import shutil
import os

file_List = ["train", "val"]
for file in file_List:
    if not os.path.exists('../my_dataset/data/images/%s' % file):
        os.makedirs('../my_dataset/data/images/%s' % file)
    if not os.path.exists('../my_dataset/data/labels/%s' % file):
        os.makedirs('../my_dataset/data/labels/%s' % file)
    print(os.path.exists('../tmp/%s.txt' % file))
    f = open('%s.txt' % file, 'r')
    lines = f.readlines()
    for line in lines:
        print(line)
        line = "/".join(line.split('/')[-5:]).strip()
        shutil.copy(line, "../my_dataset/data/images/%s" % file)
        line = line.replace('JPEGImages', 'labels')
        line = line.replace('jpg', 'txt')
        # my_dataset/data/images
        shutil.copy(line, "../my_dataset/data/labels/%s/" % file)