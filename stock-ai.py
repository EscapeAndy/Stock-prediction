import yfinance as yf


#SYMBOL = "2330.TW"
SYMBOL = input("Please enter StockID: ")

HISTORY = "10y"

all_day_k = yf.Ticker(SYMBOL).history(period=HISTORY, interval="1d")
all_day_k = all_day_k[:-1]
all_day_k
import numpy as np
import pandas as pd


PAST_WIN_LEN = 100
CLASSES = ["Bull", "Bear"]
LABEL_BULL = CLASSES.index("Bull")
LABEL_BEAR = CLASSES.index("Bear")

x, y = [], []
for today_i in range(len(all_day_k)):
    day_k_past = all_day_k[:today_i+1]
    day_k_forward = all_day_k[today_i+1:]
    if len(day_k_past) < PAST_WIN_LEN or len(day_k_forward) < 1:
        continue
    day_k_past_win = day_k_past[-PAST_WIN_LEN:]
    day_k_forward_win = day_k_forward[:1]

    today_price = day_k_past_win.iloc[-1]["Close"]
    tomorrow_price = day_k_forward_win.iloc[0]["Close"]
    label = LABEL_BULL if tomorrow_price > today_price else LABEL_BEAR

    x.append(day_k_past_win.values)
    y.append(label)

x, y = np.array(x), np.array(y)

TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.2, 0.1

test_split_index = -round(len(x) * TEST_SPLIT)
x_other, x_test = np.split(x, [test_split_index])
y_other, y_test = np.split(y, [test_split_index])

train_split_index = round(len(x) * TRAIN_SPLIT)
indexes = np.arange(len(x_other))
np.random.shuffle(indexes)
train_indexes, val_indexes = np.split(indexes, [train_split_index])
x_train, x_val = x_other[train_indexes], x_other[val_indexes]
y_train, y_val = y_other[train_indexes], y_other[val_indexes]
label_distribution = pd.DataFrame([{"Dataset": "train",
                                   "Bull" : np.count_nonzero(y_train == LABEL_BULL),
                                   "Bear" : np.count_nonzero(y_train == LABEL_BEAR)},
                                   {"Dataset": "val",
                                    "Bull" : np.count_nonzero(y_val == LABEL_BULL),
                                    "Bear" : np.count_nonzero(y_val == LABEL_BEAR)},
                                   {"Dataset": "test",
                                    "Bull" : np.count_nonzero(y_test == LABEL_BULL),
                                    "Bear" : np.count_nonzero(y_test == LABEL_BEAR)}])
label_distribution
x_test_bull = x_test[y_test == LABEL_BULL]
x_test_bear = x_test[y_test == LABEL_BEAR]

min_n_labels = min(len(x_test_bull), len(x_test_bear))

x_test_bull = x_test_bull = x_test_bull[np.random.choice(len(x_test_bull), min_n_labels, replace=False), :]
x_test_bear = x_test_bear = x_test_bear[np.random.choice(len(x_test_bear), min_n_labels, replace=False), :]
x_test = np.vstack([x_test_bull, x_test_bear])

y_test = np.array([LABEL_BULL] * min_n_labels + [LABEL_BEAR] * min_n_labels)

pd.DataFrame([{"Dataset": "test",
              "Bull": np.count_nonzero(y_test == LABEL_BULL),
              "Bear": np.count_nonzero(y_test == LABEL_BEAR)}])
np.savez("datasets.npz", x_train=x_train, y_train=y_train,
        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 載入數據集
data = np.load("datasets.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_val, y_val = data["x_val"], data["y_val"]
x_test, y_test = data["x_test"], data["y_test"]

# 建立模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

# 編譯模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 訓練模型
model.fit(x_train, y_train, batch_size=1, epochs=1, validation_data=(x_val, y_val))

# 測試模型
predictions = model.predict(x_test)


# 載入新的股市數據
new_data = yf.Ticker(SYMBOL).history(period="1d", interval="1d")

# 選取最近100天的數據
new_data = new_data[-100:]

# 使用模型來預測明天的股價走勢
prediction = model.predict(new_data.values.reshape(1, -1, new_data.shape[1]))

prediction

# 判斷明天的股價走勢
if prediction > 0.5:
    print(SYMBOL+" 明天的股價將上漲。")
else:
    print(SYMBOL+" 明天的股價將下跌。")
    
model.save('stock_prediction_model.h5')
    
    
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 預測測試集
predictions = model.predict(x_test)

# 將預測結果轉換為標籤
predictions_labels = (predictions > 0.5).astype(int)

# 計算混淆矩陣
cm = confusion_matrix(y_test, predictions_labels)

# 繪製混淆矩陣
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()

   
