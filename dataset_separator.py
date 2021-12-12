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
folder = "Humans"

hdir_glob = glob.glob("./Humans/*")
images_nb = len(hdir_glob)

# backends = []

if not os.path.exists(dir):
    print("Install/Unzip corresponding dataset")

def place_corresponding_folder(image_file, isMale):
    shutil.copy(image_file, maledir if isMale == True else femaledir)


for filename in os.listdir(folder):
    img_file = os.path.join(folder, filename)
    print("IMREAD file {}...".format(img_file))
    img = cv2.imread(img_file)
    print("imread done, proceeding to the if")
    if img is not None:
        print("img not None")
        try:
            result = DeepFace.analyze(img, actions=['gender', 'age', 'race'], enforce_detection=False)

            age = result['age']

            if age < 18 or age > 50:
                print("image {} has a person with invalid determined age".format(img_file))
                continue
            place_corresponding_folder(img_file, result['gender'] == 'Man')

        except Exception as e:
            print("DeepFace.analyze error on {}".format(img_file))
            print("## {}".format(e))



# for i in range(images_nb):
#     img_file = hdir_glob[i]
#     # img_file = os.path.join(dir, filename)
#     # if not os.path.isfile(img_file):
#     #     print("invalid_file : {} {}".format(filename, img_file))
#     #     continue
#     img = cv2.imread(img_file)
#     try:
#         result = DeepFace.analyze(img, actions=['gender', 'age', 'race'], enforce_detection=False)
#         gender = result['gender']
#         age = result['age']
#         main_race = result['dominant_race']
#         races = result['race']
#         if age < 18 or age > 50:
#             print("image {} has a person with invalid age".format(img_file))
#             continue
#         elif main_race == 'black':
#             print("invalid race, go drink on another fountain.")
#             continue
        
#         place_corresponding_folder(img_file, gender)

#     except:
#         print("Error reading file {}".format(img_file))