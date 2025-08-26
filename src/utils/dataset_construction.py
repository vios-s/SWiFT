import os
import pandas as pd
import cv2 as cv
from image_similarity_measures.quality_metrics import rmse
# from sklearn.metrics.pairwise import cosine_similarity


# Read csv
file_path = './data/Skin/csv/fitzpatrick17k.csv'
fitz_data = pd.read_csv(file_path)

# attribute
attribute = ['size', 'fitzpatrick']

# fitzpatrick dataset process
fitz_data['race'] = fitz_data['fitzpatrick'].apply(lambda x: 1 if x <= 3 else 0)
fitz_data['counter_sim'] = 0
white_race_data = fitz_data[fitz_data['race'] == 1]
black_race_data = fitz_data[fitz_data['race'] == 0]
for i in range(white_race_data.shape[0]):
    whiterow = fitz_data.iloc[i]
    try:
        image1_path = whiterow.filepath
        image1 = cv.imread(whiterow.filepath)
        image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
        image1_name = whiterow.image_name
        for j in range(black_race_data.shape[0]):
            blackrow = black_race_data.iloc[j]
            image2_path = blackrow.filepath
            # image_sim = evaluation(image1_path, image2_path, metrics=["rmse", "psnr"])
            image2 = cv.imread(blackrow.filepath)
            image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

            image2_name = blackrow.image_name
            new_image_sim = rmse(image1, image2)
            if new_image_sim > whiterow.counter_sim:
                white_race_data.loc[whiterow[0], 'counter_sim'] = new_image_sim
                fitz_data.loc[whiterow[0], 'counter_image'] = blackrow.filepath
            if new_image_sim > blackrow.counter_sim:
                black_race_data.loc[blackrow[0], 'counter_sim'] = new_image_sim
                fitz_data.loc[blackrow[0], 'counter_image'] = whiterow.filepath
    except:
        pass


pd.DataFrame.to_csv(fitz_data, os.path.join('./data/Skin/csv/', 'fitzpatrick17k_counter.csv'), index=False)







