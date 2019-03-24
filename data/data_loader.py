import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn import tree
import graphviz
import sklearn.model_selection as ms
import numpy as np
from sklearn.pipeline import Pipeline

def load_mushroom_data(return_cols=False):
    mushrooms = pd.read_csv('mushrooms_old.csv')
    mushrooms = mushrooms.drop("veil-type",axis=1)
    # mushrooms = pd.get_dummies(mushrooms,
    #                            columns=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
    #                                     'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
    #                                     'stalk-surface-above-ring', 'stalk-surface-below-ring',
    #                                     'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
    #                                     'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'])
    lencoder = LabelEncoder()
    for col in mushrooms.columns:
        mushrooms[col] = lencoder.fit_transform(mushrooms[col])

    # mushrooms["class"] = lencoder.fit_transform(mushrooms["class"])
    mushroomsX = mushrooms.drop('class', axis=1).copy().values
    mushroomsY = mushrooms['class'].copy().values

    if return_cols:
        return mushroomsX, mushroomsY,mushrooms.columns[1:]
    return mushroomsX, mushroomsY


def load_diabetes_data(return_cols=False):
    diabetes = pd.read_csv('diabetes.csv')
    diabetesX = diabetes.drop('Outcome', axis=1).copy().values
    diabetesY = diabetes['Outcome'].copy().values

    if return_cols:
        return diabetesX, diabetesY, diabetes.columns[:-1]
    return diabetesX, diabetesY



def load_data(dataset,return_cols=False):
    if dataset=="mushrooms":
        return load_mushroom_data(return_cols=return_cols)
    elif dataset=="diabetes":
        return load_diabetes_data(return_cols=return_cols)




mushrooms = pd.read_csv('mushrooms_old.csv')
mushrooms = mushrooms.drop("veil-type",axis=1)
lencoder = LabelEncoder()
for col in mushrooms.columns:
    mushrooms[col] = lencoder.fit_transform(mushrooms[col])


cols = mushrooms.columns.tolist()
print(cols)
cols = cols[1:] + cols[0:1]
print(cols)
mushrooms = mushrooms[cols]
print(mushrooms)
mushrooms.to_csv("mushrooms.csv",index=False)







# dataset = "diabetes" # SET DATASET TO USE. "mushrooms" or "credit_cards"
# data_x, data_y,cols = load_data(dataset,return_cols=True)
#
# ds_train_x, ds_test_x, ds_train_y, ds_test_y = ms.train_test_split(data_x, data_y,
#                                                                    test_size=.3,
#                                                                    random_state=100,
#                                                                    stratify=data_y)
# pipe = Pipeline([('Scale', StandardScaler())])
# train_x = pipe.fit_transform(ds_train_x, ds_train_y)
# train_y = np.atleast_2d(ds_train_y).T
# test_x = pipe.transform(ds_test_x)
# test_y = np.atleast_2d(ds_test_y).T
#
# train_x, validate_x, train_y, validate_y = ms.train_test_split(train_x, train_y,
#                                                                test_size=0.3, random_state=100,
#                                                                stratify=train_y)
#
# test_y = pd.DataFrame(np.where(test_y == 0, -1, 1))
# train_y = pd.DataFrame(np.where(train_y == 0, -1, 1))
# validate_y = pd.DataFrame(np.where(validate_y == 0, -1, 1))
#
# tst = pd.concat([pd.DataFrame(test_x), test_y], axis=1)
# trg = pd.concat([pd.DataFrame(train_x), train_y], axis=1)
# val = pd.concat([pd.DataFrame(validate_x), validate_y], axis=1)
#
# print("Test validation and training")
# tst.to_csv('{}_test.csv'.format('diabetes'), index=False, header=False)
# trg.to_csv('{}_train.csv'.format('diabetes'), index=False, header=False)
# val.to_csv('{}_validate.csv'.format('diabetes'), index=False, header=False)
#
# data_train_x, data_test_x, data_train_y, data_test_y = ms.train_test_split(data_x, data_y, test_size=0.3, stratify=data_y, random_state=0)
