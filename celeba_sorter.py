#!/usr/bin/

import pandas as pd

csv = pd.read_csv("./list_attr_celeba.csv")
gender_csv = csv[['image_id', 'Male']]
gender_csv.Male = gender_csv.Male == 1

for i in range(len(gender_csv)):
    print("[{}] - {}".format(gender_csv.image_id[i], gender_csv.Male[i]))

# for row in gender_csv.iterrows():
#     print("[{}] - {}".format(row.image_id, row.Male))