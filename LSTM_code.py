# multivariate multistep forecasting

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed



def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    # define parameters
    verbose, epochs, batch_size = 0, 50, 16
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    # reshape output into samples, timesteps, features
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    # define model
    model = Sequential()
    model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
    # fit network
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model
    

def forecast(model, history, n_input):
    data = array(history)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    input_x = data[-n_input: :]
    # reshape into [1, n_input, n]
    input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    # dorecast the next week
    yhat = model.predict(input_x, verbose = 0)
    yhat = yhat[0]
    return yhat

def evaluate_forecasts(actual, predicted):
    scores = list()
    # calculate rmse score
    for i in range(actual.shape[1]):
        
        mse = mean_squared_error(actual[:, i], predicted[:, i])
        rmse = sqrt(mse)
        scores.append(mse)
    # calculate overall rmse
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s+= (actual[row, col] - predicted[row, col])**2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores
  

def evaluate_model(train, test, n_input):
    # fit mode
    model = build_model(train, n_input)
    # history is a list of weekly data
    history = [x for x in train]
    # walk forwarf validation over each week
    predictions = list()
    for i in range(len(test)):
        yhat_sequence = forecast(model, history, n_input)
        predictions.append(yhat_sequence)
        # get real observation and add to history for predicting next week
        history.append(test[i, :])
    predictions = array(predictions)
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)
    return score, scores, predictions
        

# split data into train and test
def split_dataset(data):
    # split data into standard weeks
    train, test = data[0:11690], data[11690:]
    # restructure into windows of weekly data
    train = array(split(train, len(train)/7))
    test = array(split(test, len(test)/7))
    return train, test

# convert to supervised format with reqiuired shape
def to_supervised(train, n_input, n_out=7):
    # reshaping to 2D to flatten
    data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
    X, y = list(), list()
    in_start = 0
    # step over entire history one time step at a time
    for _ in range(len(data)):
        # define end of input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
        # ensure we have enough data for this instance
        if out_end < len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, -1])
        # move along one time step
        in_start = in_start + 1
    return array(X), array(y)
  
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s' % (name, score, s_scores))


#####################################################

#
## load dataset
#dataset = read_csv('household_power_consumption_days.csv',
#                   header=0,
#                   infer_datetime_format=True,
#                   parse_dates=['datetime'],
#                   index_col=['datetime']
#                   )
#dataset.head()
#dataset = dataset.values


dataset = read_csv('final_LSTM_input_data.csv')
dataset = dataset.iloc[0:16688, :]

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
categorical_features_boolean = dataset.dtypes==object
categorical_features = dataset.columns[categorical_features_boolean].tolist()
dataset[categorical_features] = dataset[categorical_features].apply(lambda col: le.fit_transform(col))


# sort values
dataset.sort_values(['fiscal_year_nbr', 'fiscal_week_nbr'], inplace=True)


# split into train and test
train, test = split_dataset(dataset.values)

# evaluate model and get scores
n_input = 14
score, scores, predictions = evaluate_model(train, test, n_input)

# summarize_scores
summarize_scores('lstm', score, scores)


