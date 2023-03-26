import pandas as pd
import numpy as np
import os

original_path = os.path.join(os.getcwd(),  'SketchData', 'original')
corrupted_path = os.path.join(os.getcwd(), 'SketchData', 'corrupted')

image_paths={0:[],1:[]}

for image in os.listdir(original_path):
    image_paths[0].append(os.path.join('original', image))
    image_paths[1].append(os.path.join('corrupted', image[:len(image) - 4]+'c'+".png"))

# for image in os.listdir(corrupted_path):
#     image_paths[1].append(os.path.join('corrupted', image))

df=pd.DataFrame(image_paths)

df.to_csv('image_paths_train.csv', index=False, header=False)
