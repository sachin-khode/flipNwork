import shutil
import os
import requests
import pandas as pd
import csv


def make_directory(dirname):
    # current_path = "/home/sk-ji/Desktop/Web_Scrp_Flip/image_data/training"
    current_path = os.getcwd()
    path = os.path.join(current_path, dirname)
    if not os.path.exists(path):
        os.makedirs(path)


def make_directory2(dirname,existing_dir):
    # current_path = str(existing_dir/dirname)
    current_path = os.getcwd()
    path = os.path.join(current_path, existing_dir)
    if not os.path.exists(path):
        os.makedirs(path)




def save_images(data, dirname, page):
    for index, link in enumerate(data['image_urls']):
        print("Downloading {0} of {1} images".format(index + 1, len(data['image_urls'])))
        response = requests.get(link)
        # dirn ame = make_directory(dirname)
        with open('{0}/img_{1}{2}.jpeg'.format(dirname, page, index), "wb") as file:
            file.write(response.content)


def save_data_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, mode="a", encoding="utf-8-sig")

dirrr = "kjdskjhfdkfjs"
dirr2 = "dfdssfsdddddddddddd"
make_directory(dirrr)
make_directory2(dirr2,dirrr)