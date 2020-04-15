import numpy as np
import csv
import pandas as pd
import os

images_dic = {1: 'Early_Renaissance',
                2: 'Naïve_Art_(Primitivism)',
                3: 'Expressionism',
                4: 'Magic_Realism',
                5: 'Northern_Renaissance',
                6: 'Rococo',
                7: 'Ukiyo-e',
                8: 'Art_Nouveau_(Modern)',
                9: 'Pop_Art',
                10: 'High_Renaissance',
                11: 'Minimalism',
                12: 'Mannerism_(Late_Renaissance)',
                13: 'Art_Informel',
                14: 'Neoclassicism',
                15: 'Color_Field_Painting',
                16: 'Symbolism',
                17: 'Realism',
                18: 'Romanticism',
                19: 'Surrealism',
                20: 'Cubism',
                21: 'Impressionism',
                22: 'Baroque',
                23: 'Abstract_Expressionism',
                24: 'Post-Impressionism',
                25: 'Abstract_Art'}

def mean_class(date, class_data, label):
    sum = 0
    count = 0
    for i in range(0, class_data.shape[0]):
        if class_data[i] == label and date[i] != -1:
            sum = sum + date[i]
            count = count + 1
    mean = int(sum / count)
    for i in range(0, class_data.shape[0]):
        if class_data[i] == label and date[i] == -1:
            date[i] = mean

    return date

def fortmatData(array_data):

    class_data = array_data[:,0]
    author_name_data = array_data[:,1]
    data = []

    #Get data
    valid_number = []
    for i in range(0, author_name_data.shape[0]):
        res = author_name_data[i].split("_")
        res1 = res[1].split(".")
        res2 = res1[0].split("-")
        for token in res2:
            if token.isnumeric() == True:
                n_str = str(token)
                if(len(n_str) == 4):
                    valid_number.append(token)

        valid_number = np.array(valid_number)

        if valid_number.shape[0] == 0:
            data.append(-1)
        else:
            data.append(valid_number[valid_number.shape[0]-1])

        valid_number = []

    data = np.array(data)
    data = data.astype('int')

    for i in range(1,26):
        data = mean_class(data, class_data, images_dic.get(i))

    rows = []
    key_list = np.asarray(list(images_dic.keys()))
    val_list = np.asarray(list(images_dic.values()))
    for i in range(0, array_data.shape[0]):
        row = []
        key = key_list[np.where(val_list == array_data[i, 0])][0]
        row.append(key)
        row.append(array_data[i, 1])
        row.append(data[i])
        for j in range(2, array_data.shape[1]):
            row.append(array_data[i, j])

        rows.append(row)

    return np.array(rows)


input_data = pd.read_csv('../outputs/img_dataset_testing.csv').values
output_data  = fortmatData(input_data)
with open('../outputs/img_dataset.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    for i in range(0, output_data.shape[0]):
        writer.writerow(output_data[i, :])