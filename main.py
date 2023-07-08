import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression



def load_dataset(file_path):
    try:
        # Load the dataset using pandas
        dataset = pd.read_csv(file_path)
        return dataset
    except FileNotFoundError:
        print("Error: File not found.")
    except Exception as e:
        print("Error:", str(e))


def dataset_prep(df):
    # drop duplicates
    df = df.drop_duplicates()
    # replace any null value at connection type to "unknown"
    df.loc[df.connectionType.isna(), 'connectionType'] = 'UNKNOWN'
    # now drop the few null values in other columns
    df = df.dropna()
    # delete all rows with winBid > sentPrice
    df = df[~(df.bidFloorPrice > df.sentPrice)]
    return df


def add_time_based_features(df):
    df['eventHour'] = pd.to_datetime(df.eventTimestamp, unit='ms').dt.hour
    df['eventDayOfWeek'] = pd.to_datetime(df.eventTimestamp, unit='ms').dt.dayofweek
    return df


def drop_columns(df, col_names_lst):
    df = df.drop(col_names_lst, axis=1)
    return df


def convert_size(df, column_name):
    # Extract the two size values from the column
    sizes = df[column_name].str.split('X')

    # Convert the sizes to integers and calculate the product
    converted_sizes = sizes.apply(lambda x: int(x[0]) * int(x[1]))

    # Replace the column with the converted sizes
    df[column_name] = converted_sizes

    return df


def one_hot_encode(df, column_list):
    # Perform one-hot encoding using pandas' get_dummies function for each column in the list
    encoded_dfs = [pd.get_dummies(df[col], prefix=col, dtype=int) for col in column_list]

    # Concatenate the encoded columns with the original DataFrame
    df_encoded = pd.concat([df] + encoded_dfs, axis=1)

    # Drop the original columns from the DataFrame
    df_encoded.drop(column_list, axis=1, inplace=True)

    return df_encoded


def predict_in_batches(model, X, batch_size=1000):
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    predictions = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_samples)

        batch_X = X[start_idx:end_idx]
        batch_pred = model.predict(batch_X)
        predictions.append(batch_pred)

    return np.concatenate(predictions, axis=0)


if __name__ == '__main__':
    train_set = load_dataset('/Users/laurentobaly/PycharmProjects/DT_Home_Assignment/data/train_data.csv')
    test_set = load_dataset('/Users/laurentobaly/PycharmProjects/DT_Home_Assignment/data/test_data.csv')
    test_device_id = test_set['deviceId']

    train_set = dataset_prep(train_set)
    test_set = dataset_prep(test_set)

    # train_set = convert_size(train_set, 'size')
    # test_set = convert_size(test_set, 'size')

    train_set = add_time_based_features(train_set)
    test_set = add_time_based_features(test_set)

    train_set = drop_columns(train_set, ['size', 'eventTimestamp', 'correctModelName', 'appVersion', 'osAndVersion',
                                         'deviceId', 'mediationProviderVersion', 'countryCode', 'brandName', "c1", "c2", "c3", "c4",
                                         'sentPrice', 'has_won'])
    test_set = drop_columns(test_set, ['size', 'eventTimestamp', 'correctModelName', 'appVersion', 'osAndVersion',
                                       'deviceId', 'mediationProviderVersion','countryCode', 'brandName', "c1", "c2", "c3", "c4",
                                       'sentPrice'])

    train_set = one_hot_encode(train_set, ['unitDisplayType', 'bundleId', 'connectionType'])
    test_set = one_hot_encode(test_set, ['unitDisplayType',  'bundleId', 'connectionType'])

    y_train = train_set['winBid']
    X_train = train_set.drop('winBid', axis=1)
    X_test = test_set

    xgb = XGBRegressor()
    xgb.fit(X_train, y_train)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Predict on the test set
    y_pred_lr = lr.predict(X_train)
    y_pred_xgb = xgb.predict(X_train)

    # Evaluate the model's performance
    rmse_lr = mean_squared_error(y_train, y_pred_lr, squared=False)
    rmse_xgb = mean_squared_error(y_train, y_pred_xgb, squared=False)
    print("Root Mean Squared Error Of Linear Regression:", rmse_lr)
    print("Root Mean Squared Error Of XGBRegressor:", rmse_xgb)

    y_results = xgb.predict(test_set)
    results = pd.concat([test_device_id, pd.Series(y_results)], axis=1)
    results.to_csv('results.csv')




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
