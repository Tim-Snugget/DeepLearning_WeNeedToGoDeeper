import shutil, os
import matplotlib.image as mpimg
import cv2
from deepface import DeepFace

## HUMAN FACES

maledir = "./Humans/Male"
femaledir = "./Humans/Female"
folder = "Humans"

def place_corresponding_folder(image_file, isMale):
    shutil.copy(image_file, maledir if isMale == True else femaledir)

for filename in os.listdir(folder):
    img_file = os.path.join(folder, filename)
    print("IMREAD file {}...".format(img_file))
    img = cv2.imread(img_file)
    if img is not None:
        try:
            result = DeepFace.analyze(img, actions=['gender', 'age'], enforce_detection=False)
            # result = DeepFace.analyze(img, actions=['gender', 'age', 'race'], enforce_detection=False)

            age = result['age']
            if age < 18 or age > 50:
                print("image {} has a person with invalid determined age".format(img_file))
                continue
            place_corresponding_folder(img_file, result['gender'] == 'Man')

        except:
            print("DeepFace.analyze error on {}".format(img_file))
    else:
      print("Image reading on {} is None".format(img_file))

## CELEBA
import pandas as pd

csv = pd.read_csv("./list_attr_celeba.csv")
gender_csv = csv[['image_id', 'Male']]
gender_csv.Male = gender_csv.Male == 1 # making it compliant with the place_corresponding_folder function
celeba_folder = "img_align_celeba/img_align_celeba"

for i in range(len(gender_csv)):
  img_file = os.path.join(celeba_folder, gender_csv.image_id[i])
  print("[{}] - {}".format(gender_csv.image_id[i], gender_csv.Male[i]))
  place_corresponding_folder(img_file, gender_csv.Male[i])