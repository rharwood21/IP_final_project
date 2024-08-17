import os
import csv
import cv2 as cv
import sklearn
from sklearn.svm import LinearSVC
import sys
import pandas as pd
from image_descriptors import *
from pathlib import Path
# for two labels from the OG csv, make a list of all images associated with those labels

# open the images, run descriptors for the regions in the labeled regions

# save descriptions

#def extract_features(img):
    # get texture features from image

if __name__ == '__main__':
    # initialize some paths
    archive_path = Path('./archive')
    all_pngs = list(archive_path.glob('*/*/*.png'))

    labels = []
    data = []

    lbp_desc = LocalBinaryPatterns(24, 8)
    hog_desc = HOG(8, (16,16), (1,1), False, False)

    # read csv into dataframe or something
    with open('BBox_List_2017.csv', 'r', newline='') as csv_data:
        #csv_reader = csv.reader(csv_data)
        findings_df = pd.read_csv('BBox_List_2017.csv')
    disease_classes = ['Mass', 'Nodule', 'Pneumonia']
    # collect just the Findings = 'Mass' or 'Nodule'
    findings_df = findings_df.loc[findings_df['Finding Label'].isin(disease_classes)]
    # sort by image index for ease of folder access
    findings_df = findings_df.sort_values('Image Index')
    # in each bounding box for Mass or Nodule, find LBP
    for index, img in findings_df.iterrows():
        for filename in all_pngs:
            #print(f'filename: {filename} == image index: {img["Image Index"]}? {filename == img["Image Index"]}')
            if filename.name == img["Image Index"]:
                this_img = cv.imread(filename)
                this_img_gray = cv.cvtColor(this_img, cv.COLOR_BGR2GRAY)
                # now that we've found the image, need to get some features and save them off.
                x, y, w, h = img["Bbox [x"], img["y"], img["w"], img["h]"]
                this_img_region = this_img_gray[int(x):int(x+w), int(y):int(y+h)]

                lbp_hist = lbp_desc.describe(this_img_region)
                labels.append(img['Finding Label'])
                data.append(lbp_hist)

                hog_hist = hog_desc.describe(this_img_region)
                labels.append(img['Finding Label'])
                data.append(hog_hist)
    # now that labels and histograms have been collected, train an SVM
    texture_model = sklearn.svm.LinearSVC(C=100.0, random_state=7)
    texture_model.fit(data, labels)

    # SVM has been trained. Let's test it out.
    correct_predicts = 0
    num_predicts = 0
    mass_predicts = 0
    nodule_predicts = 0
    correct_nodule_predicts = 0
    correct_mass_predicts = 0
    pneumonia_predicts = 0

    for index, img in findings_df.iterrows():
        for filename in all_pngs:
            if filename.name == img["Image Index"]:
                this_img = cv.imread(filename)
                this_img_gray = cv.cvtColor(this_img, cv.COLOR_BGR2GRAY)
                hist = lbp_desc.describe(this_img_gray)
                label_prediction = texture_model.predict(hist.reshape(1, -1))
                num_predicts += 1
                if label_prediction == img['Finding Label']:
                    correct_predicts += 1
                if label_prediction == 'Nodule':
                    nodule_predicts += 1
                    print(f"{filename.name} predicted as Nodule!")
                if label_prediction == 'Mass':
                    mass_predicts += 1
                if label_prediction == 'Pneumonia':
                    pneumonia_predicts += 1
                    #print(f"{filename.name} predicted as Pneumonia!")

    print(f"Correct Prediction Rate: {correct_predicts/num_predicts}")


                #cv.putText(this_img, label_prediction[0], (10, 30), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 3)
                #cv.putText(this_img, img['Finding Label'], (10, 60), cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 3)
                #cv.imshow("Prediction", this_img)
                #cv.waitKey(0)

