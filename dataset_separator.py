#!/usr/bin

# objectives : sort datasets between gender, ages and races
# 2 datasets :
#   - jessicali9530/celeba-dataset
#   - ashwingupta3012/human-faces

import shutil, os
import glob
import matplotlib.image as mpimg
import cv2
from deepface import DeepFace

## HUMAN FACES
dir = "./Humans"
maledir = "./Humans/Male"
femaledir = "./Humans/Female"

hdir_glob = glob.glob("./Humans/*")
images_nb = len(hdir_glob)

if not os.path.exists(dir):
    print("Install/Unzip corresponding dataset")

def place_corresponding_folder(image_file, gender):
    shutil.copy(image_file, maledir if gender == 'Man' else femaledir)

for i in range(images_nb):
    img_file = hdir_glob[i]
    # img_file = os.path.join(dir, filename)
    # if not os.path.isfile(img_file):
    #     print("invalid_file : {} {}".format(filename, img_file))
    #     continue
    img = cv2.imread(img_file)
    try:
        result = DeepFace.analyze(img, actions=['gender', 'age', 'race'], enforce_detection=False)
        gender = result['gender']
        age = result['age']
        main_race = result['dominant_race']
        races = result['race']
        if age < 18 or age > 50:
            print("image {} has a person with invalid age".format(img_file))
            continue
        elif main_race == 'black':
            print("invalid race, go drink on another fountain.")
            continue
        
        place_corresponding_folder(img_file, gender)

    except:
        print("Error reading file {}".format(img_file))