from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('stock_prediction_model.h5')

@app.route('/', methods=['GET', 'POST'])
#@app.route('/')


def index():
    if request.method == 'POST':
        stock_id = request.form['stock_id']
        prediction = predict_stock(stock_id)
        return render_template('index.html', prediction=prediction)
    return render_template('index.html')

def predict_stock(stock_id):
    # 載入新的股市數據
    new_data = yf.Ticker(stock_id).history(period="1d", interval="1d")
    new_data = new_data[-100:]

    # 使用模型來預測明天的股價走勢
    prediction = model.predict(new_data.values.reshape(1, -1, new_data.shape[1]))

    if prediction > 0.5:
        return f"{stock_id} 明天的股價將上漲。"
    else:
        return f"{stock_id} 明天的股價將下跌。"

if __name__ == '__main__':
    app.run(debug=True)
    
