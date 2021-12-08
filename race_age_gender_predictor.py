#!/usr/bin

# Using the dataset of 'ashwingupta3012/human-faces' in ./Humans

from deepface import DeepFace
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random,os
import pandas as pd

import glob
import cv2
import traceback

# Download dataset :
if not os.path.exists("./Humans"):
    os.system("kaggle datasets download 'ashwingupta3012/human-faces' --unzip")

### age gender
img_len = len(glob.glob('./Humans'))
print("found {} files in ./Humans".format(img_len))

def calculate_race_age_gender(image, img_name):
    name = {}
    race = 'NA'
    age = 'NA'
    gender = 'NA'

    agebucket = 'NA'

    try:
        img_arr=cv2.imread(image)
        ## get gender
        response=DeepFace.analyze(img_arr,actions=["race", "gender","age"],enforce_detection=False)
        race = response['race']
        gender = response['gender']
        age = response['age']
        ## Bucket the age
        if int(age) >= 13 and int(age) <= 17:
            agebucket = '13-17years'
        elif int(age) > 17 and int(age) <= 24:
            agebucket = '18-24years'
        elif int(age) > 24 and int(age) <= 34:
            agebucket = '25-34years'
        elif int(age) > 34 and int(age) <= 44:
            agebucket = '35-44years'
        elif int(age) > 44 and int(age) <= 54:
            agebucket = '45-54years'
        elif int(age) > 54 and int(age) <= 64:
            agebucket = '55-64years'
        elif int(age) > 64:
            agebucket = 'above 65years'
        else:
            agebucket = 'NA'
        ## store in dictionary
        name[img_name]=(race, age, gender, agebucket)
    except:
        name[img_name]='NA' ## If the image is not a front facing image
        traceback.print_exc()
    return name


count = 0
img_list = {}
for i in range(img_len):
    image = glob.glob("./Humans/*")[i]
    image_name = str(image)
    print("image is {}".format(image))
    count += 1
    img_list[image_name] = calculate_race_age_gender(image, str(image))
    if (count < 50):
        continue
    else:
        break

print(img_list[:10])
g   = plt.figure(figsize=(15, 15))

rows = 6
columns = 3

for i in range(18):
    image = glob.glob("./Humans/*")[i]
    image_name = str(image)
    img_arr = cv2.imread(image)
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, i+1)
    # showing image
    plt.imshow(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image:"+glob.glob("./Humans/*")[i][7:]+"\n" +
        "Gender:" + img_list[image_name][i]["gender"] + "\n" +
        "Age: " + img_list[image_name][i]["age"] + "\n" +
        "Race: " + img_list[image_name][i]["race"] + "\n" +
        "Age Bucket:"+img_list[image_name][i]["agebucket"])

# ### race_prediction
# random_file = random.choice(os.listdir("./Humans"))
# print("random file is {}".format(random_file))
# img = mpimg.imread("./Humans/{}".format(random_file))
# imgplot = plt.imshow(img)
# plt.axis('off')
# plt.show()
# result=DeepFace.analyze(img,actions=['race'])
# result = DeepFace.analyze(img, actions=['race'], models={}, enforce_detection=False)
# new = pd.DataFrame.from_dict(result) 
# new["race"]
# print("Dominant Race for this person is {} with likelihood as {} %".format((new.dominant_race.iloc[0].capitalize()), (round(new.race.max(),2))))