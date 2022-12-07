from itertools import chain

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing

train_file = "./input/train.csv"
test_file = "./input/test.csv"
sample_submission = "./input/sample_submission.csv"
submission_filename = "god_of_overfitting_spare_us.csv"

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.8,
    'subsample': .85,
    'eta': 0.0275,
    'objective': 'binary:logitraw',
    'num_parallel_tree': 7,
    'max_depth': 5,
    'nthread': 22,
    'eval_metric': 'auc',
    'verbosity':0
}

top111 = ['Field12', 'PersonalField52', 'PersonalField80', 'PersonalField44', 'Field9',
          'PropertyField7', 'PropertyField12', 'CoverageField5B', 'PersonalField42', 'PersonalField45',
          'PersonalField81', 'PropertyField8', 'PersonalField79', 'GeographicField45B', 'PropertyField22',
          'PersonalField75', 'PersonalField31', 'PropertyField19', 'PropertyField31', 'GeographicField11A',
          'PersonalField23', 'GeographicField21B', 'PersonalField4A', 'Field10', 'GeographicField16B',
          'GeographicField20A', 'PersonalField25', 'PersonalField4B', 'PropertyField3', 'GeographicField17A',
          'GeographicField59B', 'GeographicField7B', 'GeographicField8A', 'Year', 'GeographicField6B',
          'PersonalField14',
          'GeographicField45A', 'GeographicField14B', 'SalesField12', 'CoverageField11A', 'CoverageField5A', 'Month',
          'PropertyField33', 'PersonalField5', 'CoverageField11B', 'GeographicField11B', 'GeographicField23B',
          'PropertyField39B', 'CoverageField3A', 'GeographicField1B', 'GeographicField17B', 'PropertyField39A',
          'GeographicField41B', 'CoverageField6A', 'SalesField9', 'PersonalField16', 'PersonalField26',
          'PropertyField24A', 'Field8', 'GeographicField28A', 'CoverageField3B', 'SalesField2A', 'GeographicField19B',
          'GeographicField43A', 'PropertyField16B', 'PropertyField16A', 'PropertyField1B', 'CoverageField1B',
          'PropertyField1A', 'GeographicField48B', 'PersonalField11', 'CoverageField1A', 'PersonalField15',
          'GeographicField5B', 'PropertyField34', 'CoverageField8', 'PersonalField82', 'SalesField2B',
          'PropertyField35', 'CoverageField2B', 'SalesField10', 'PropertyField21A', 'SalesField3', 'CoverageField9',
          'SalesField7', 'Weekday', 'PersonalField13', 'PropertyField21B', 'SalesField6', 'SalesField1A',
          'PersonalField9', 'SalesField4', 'PersonalField12', 'PersonalField27', 'PersonalField10B', 'Field7',
          'SalesField1B', 'PersonalField84', 'PersonalField2', 'PersonalField1', 'SalesField5', 'PersonalField10A',
          'PropertyField37', 'PropertyField29', 'GeographicField4B', 'PropertyField2B', 'GeographicField1A',
          'GeographicField61B', 'Field11', 'PersonalField76', 'PropertyField30']

drop_out = ['GeographicField19B', 'PropertyField7', 'GeographicField17A', 'GeographicField28A',
            'GeographicField21B', 'GeographicField7B', 'CoverageField11B', 'GeographicField6B', 'GeographicField45A',
            'PersonalField25', 'Month', 'CoverageField5A', 'GeographicField8A', 'GeographicField1B',
            'CoverageField6A_CoverageField6B', 'PersonalField23', 'Field11', 'PropertyField2B', 'SalesField12',
            'GeographicField41B',
            'PropertyField16A', 'Field10', 'PropertyField3', 'PropertyField16B', 'GeographicField1A',
            'GeographicField20A', 'PersonalField81', 'GeographicField16B', 'GeographicField59B', 'PersonalField79',
            'CoverageField1A_CoverageField3A', 'CoverageField3B_CoverageField4B', 'PropertyField22',
            'GeographicField61B',
            'CoverageField3A_PropertyField21A', 'PropertyField12', 'CoverageField2A_CoverageField3A',
            'CoverageField2B_CoverageField3B', 'PropertyField8', 'PropertyField30', 'GeographicField14B',
            'PersonalField31',
            'PropertyField21A', 'CoverageField3A_CoverageField4A', 'PropertyField31', 'CoverageField11A',
            'PropertyField19', 'GeographicField45B', 'CoverageField1A', 'PersonalField75',
            'GeographicField8A_GeographicField13A', 'CoverageField3B_PropertyField21B',
            'CoverageField1B_CoverageField3B', 'GeographicField6A_GeographicField13A', 'CoverageField5B',
            'PersonalField42', 'PersonalField45', 'PersonalField76', 'GeographicField6A_GeographicField8A',
            'PersonalField80', 'Field9', 'CoverageField3A', 'CoverageField3B',
            'GeographicField8A_GeographicField11A', 'GeographicField11A_GeographicField13A',
            'GeographicField4B',
            'CoverageField2B', 'Field12', 'PropertyField21B', 'CoverageField1B', 'PersonalField44',
            'GeographicField6A_GeographicField11A', 'PersonalField52']

interactions2way = [
    ("CoverageField1B", "PropertyField21B"),
    ("GeographicField6A", "GeographicField8A"),
    ("GeographicField6A", "GeographicField13A"),
    ("GeographicField8A", "GeographicField13A"),
    ("GeographicField11A", "GeographicField13A"),
    ("GeographicField8A", "GeographicField11A"),
    ("GeographicField6A", "GeographicField11A"),
    ("CoverageField6A", "CoverageField6B"),
    ("CoverageField3A", "CoverageField4A"),
    ("CoverageField2B", "CoverageField3B"),
    ("CoverageField1A", "CoverageField3A"),
    ("CoverageField3B", "CoverageField4B"),
    ("CoverageField2A", "CoverageField3A"),
    ("CoverageField1B", "CoverageField3B"),
    ("CoverageField3B", "PropertyField21B"),
    ("CoverageField3A", "PropertyField21A"),
    ("CoverageField1B", "PropertyField16B"),
    ("Weekday", "SalesField7"),
    ("PersonalField9", "CoverageField6B"),
    ("PersonalField12", "CoverageField6A"),
    ("PropertyField16B", "PropertyField21A"),
    ("PersonalField12", "Field8"),
    ("PropertyField32", "PersonalField9"),
    ("Field6", "CoverageField6A"),
    ("PersonalField12", "CoverageField6A"),
    ("CoverageField6A", "PropertyField34"),
    ("PersonalField33", "PropertyField8"),
    ("CoverageField2A", "CoverageField3B")
]

interactions3way = [('PersonalField23', 'PersonalField9', 'PropertyField37'),
                    ('CoverageField3A', 'PersonalField63', 'PropertyField21A'),
                    ('CoverageField3A', 'CoverageField4A', 'PersonalField76'),
                    ('CoverageField3A', 'CoverageField4A', 'GeographicField62A'),
                    ('CoverageField6A', 'PersonalField69', 'PersonalField9'),
                    ('CoverageField6A', 'PersonalField71', 'PersonalField9'),
                    ('GeographicField10B', 'GeographicField13A', 'PersonalField9'),
                    ('GeographicField8A', 'PersonalField71', 'PersonalField9'),
                    ('CoverageField2B', 'PersonalField75', 'PropertyField16B'),
                    ('CoverageField6A', 'PersonalField49', 'PropertyField29'),
                    ('CoverageField4B', 'PersonalField39', 'PropertyField16B'),
                    ('CoverageField11B', 'PersonalField6', 'SalesField2B'),
                    ('CoverageField11B', 'PersonalField36', 'SalesField2B'),
                    ('CoverageField2B', 'PropertyField16B', 'PropertyField8'),
                    ('CoverageField3A', 'GeographicField21A', 'PropertyField21B'),
                    ('GeographicField11A', 'PersonalField48', 'PersonalField9'),
                    ('CoverageField11B', 'PersonalField26', 'SalesField2B'),
                    ('CoverageField1B', 'CoverageField3A', 'PersonalField61'),
                    ('CoverageField1A', 'PropertyField16A', 'PropertyField36'),
                    ('PersonalField9', 'PropertyField10', 'PropertyField32'),
                    ('GeographicField11A', 'GeographicField62A', 'PersonalField12'),
                    ('Field10', 'PersonalField9', 'PropertyField34'),
                    ('CoverageField2B', 'CoverageField3A', 'PersonalField8'),
                    ('Field11', 'PropertyField34', 'SalesField6'),
                    ('PersonalField19', 'PersonalField60', 'PropertyField8')]

interactions4way = [('Field8', 'PersonalField12', 'PersonalField75', 'PropertyField37'),
                    ('CoverageField6A', 'PersonalField12', 'PropertyField37', 'PropertyField8'),
                    ('Field8', 'PersonalField9', 'PropertyField3', 'PropertyField37'),
                    ('CoverageField6A', 'Field8', 'PersonalField84', 'PersonalField9'),
                    ('CoverageField8', 'PersonalField12', 'PersonalField80', 'PropertyField37'),
                    ('CoverageField8', 'Field8', 'PersonalField12', 'PersonalField84'),
                    ('CoverageField5A', 'GeographicField11A', 'PersonalField9', 'PropertyField37'),
                    ('CoverageField1B', 'CoverageField3B', 'CoverageField5A', 'PropertyField22'),
                    ('CoverageField1A', 'CoverageField3A', 'PersonalField82', 'PropertyField19'),
                    ('CoverageField1A', 'CoverageField3A', 'PersonalField11', 'PropertyField19'),
                    ('CoverageField5A', 'Field8', 'PersonalField12', 'PersonalField42'),
                    ('CoverageField6A', 'Field11', 'PersonalField9', 'PropertyField12'),
                    ('CoverageField6A', 'CoverageField8', 'PropertyField35', 'SalesField3'),
                    ('CoverageField3A', 'PersonalField82', 'PropertyField21A', 'Year'),
                    ('CoverageField1B', 'CoverageField3B', 'PersonalField42', 'PropertyField8'),
                    ('CoverageField1B', 'CoverageField3A', 'PersonalField1', 'PropertyField16A'),
                    ('CoverageField1B', 'CoverageField3B', 'PropertyField22', 'PropertyField8'),
                    ('CoverageField6A', 'PersonalField45', 'PersonalField9', 'PropertyField29'),
                    ('CoverageField5A', 'PersonalField1', 'PropertyField35', 'SalesField3'),
                    ('CoverageField1A', 'CoverageField3A', 'Field12', 'PersonalField27'),
                    ('CoverageField5A', 'CoverageField8', 'Field11', 'PropertyField29'),
                    ('CoverageField3B', 'PersonalField25', 'PersonalField45', 'PropertyField21B'),
                    ('CoverageField2B', 'CoverageField3B', 'GeographicField17A', 'PersonalField5'),
                    ('CoverageField1A', 'CoverageField3A', 'PersonalField75', 'Year'),
                    ('Field11', 'PersonalField12', 'PersonalField25', 'PropertyField30')]

interactions2way_list = list(np.unique(list(chain(*interactions2way))))
interactions3way_list = list(np.unique(list(chain(*interactions3way))))
interactions4way_list = list(np.unique(list(chain(*interactions4way))))

interactions_list = interactions2way_list + interactions3way_list + interactions4way_list
tmp_features = list(np.setdiff1d(interactions_list, top111))

tc_features = []


def get_data():
    global tc_features

    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    y_train = train.QuoteConversion_Flag

    train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
    test = test.drop('QuoteNumber', axis=1)

    ntrain = train.shape[0]

    train_test = pd.concat((train, test), axis=0)

    train_test['Date'] = pd.to_datetime(train_test['Original_Quote_Date'])

    train_test['Year'] = train_test['Date'].dt.year
    train_test['Month'] = train_test['Date'].dt.month
    train_test['Day'] = train_test['Date'].dt.day
    train_test['Weekday'] = train_test['Date'].dt.dayofweek

    train_test['Field10'] = train_test['Field10'].apply(lambda x: x.replace(',', '')).astype(np.int32)
    train_test['PropertyField37'] = train_test['PropertyField37'].apply(lambda x: -1 if x == ' ' else x)
    train_test['GeographicField63'] = train_test['GeographicField63'].apply(lambda x: -1 if x == ' ' else x)

    train_test = train_test.drop(['Date', 'Original_Quote_Date'], axis=1)
    train_test = train_test.fillna(-1)

    categoricals = [x for x in train_test.columns if train_test[x].dtype == 'object']

    for c in categoricals:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_test[c].values))
        train_test[c] = lbl.transform(list(train_test[c].values))

    train = train_test.iloc[:ntrain, :].copy().reset_index(drop=True)
    test = train_test.iloc[ntrain:, :].copy().reset_index(drop=True)

    features = list(train.columns)
    features = np.intersect1d(features, top111 + tmp_features)

    x_train = train[features].copy()
    x_test = test[features].copy()

    x_train['NaNCount'] = x_train.apply(lambda x: np.sum(x == -1), axis=1)
    x_test['NaNCount'] = x_test.apply(lambda x: np.sum(x == -1), axis=1)

    for A, B in interactions2way:
        feat = "_".join([A, B])
        x_train[feat] = x_train[A] - x_train[B]
        x_test[feat] = x_test[A] - x_test[B]

    for A, B, C in interactions3way:
        feat = "_".join([A, B, C])
        tc_features += [feat]
        x_train[feat] = x_train[A] - x_train[B] - x_train[C]
        x_test[feat] = x_test[A] - x_test[B] - x_test[C]

    for A, B, C, D in interactions4way:
        feat = "_".join([A, B, C, D])
        tc_features += [feat]
        x_train[feat] = x_train[A] - x_train[B] - x_train[C] - x_train[D]
        x_test[feat] = x_test[A] - x_test[B] - x_test[C] - x_test[D]

    x_train.drop(tmp_features, axis=1, inplace=True)
    x_test.drop(tmp_features, axis=1, inplace=True)

    x_train.drop(drop_out[-25:], axis=1, inplace=True)
    x_test.drop(drop_out[-25:], axis=1, inplace=True)

    return np.array(x_train), np.array(y_train), np.array(x_test)


if __name__ == "__main__":
    x_train, y_train, x_test = get_data()
    print(x_train.shape, x_test.shape)

    x_train_tc = x_train.copy()
    ntcfeat = len(tc_features)
    print('1')
    x_train[:, -ntcfeat:] = 0

    ntrain = x_train.shape[0]
    best_nrounds = 2500
    print('2')
    xgb.config_context(use_rmm=True)
    dtrain = xgb.DMatrix(x_train, label=y_train)
    dtrain_tc = xgb.DMatrix(x_train_tc, label=y_train)
    print('3')
    gbdt = xgb.train(xgb_params, dtrain, best_nrounds - 100)
    xgb_params['eta'] = 0.01
    gbdt = xgb.train(xgb_params, dtrain_tc, 300, xgb_model=gbdt)
    print('4')
    dtest = xgb.DMatrix(x_test)
    print('5')
    submission = pd.read_csv(sample_submission)
    submission.iloc[:, 1] = gbdt.predict(dtest).reshape((-1, 1))
    submission.to_csv(submission_filename, index=False)