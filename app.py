from flask import Flask, render_template, request
import joblib
import yfinance as yf
import pandas as pd
import pytz
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
import locale
import os

# Locale
try:
    locale.setlocale(locale.LC_ALL, 'id_ID.UTF-8')
except locale.Error:
    locale.setlocale(locale.LC_ALL, '')

app = Flask(__name__)

@app.template_filter('format_price')
def format_price(value):
    try:
        return locale.currency(value, grouping=True).replace("Rp", "Rp ")
    except (ValueError, TypeError):
        return value

stocks = {
    'ADRO.JK': 'PT Alamtri Resources Indonesia Tbk (ADRO)',
    'ASII.JK': 'Astra International Tbk (ASII)',
    'BBCA.JK': 'Bank Central Asia Tbk (BBCA)',
    'BBNI.JK': 'Bank Negara Indonesia Persero Tbk (BBNI)',
    'BBRI.JK': 'Bank Rakyat Indonesia Persero Tbk (BBRI)',
    'BMRI.JK': 'Bank Mandiri Persero Tbk (BMRI)',
    'ICBP.JK': 'Indofood CBP Sukses Makmur Tbk (ICBP)',
    'PTBA.JK': 'Bukit Asam Tbk (PTBA)',
    'TLKM.JK': 'Telkom Indonesia Persero Tbk (TLKM)',
    'TOWR.JK': 'Sarana Menara Nusantara Tbk (TOWR)',
}

wib = pytz.timezone('Asia/Jakarta')
history_file = 'prediksi_history.csv'

MARKET_CLOSE_TIME = (16, 30)  # 16:30 WIB

def is_market_open(now=None):
    now = now or datetime.now(wib)
    return now.time() < datetime(now.year, now.month, now.day, *MARKET_CLOSE_TIME).time()

def calculate_accuracy(actual, predicted):
    try:
        if len(actual) == 0 or len(predicted) == 0:
            return "-"
        mape = np.mean(np.abs((np.array(actual) - np.array(predicted)) / actual)) * 100
        return round(mape, 2)
    except Exception:
        return "-"

def get_next_trading_day(date):
    while date.weekday() >= 5:  # sabtu/minggu
        date += timedelta(days=1)
    return date

@app.route('/')
def index():
    return render_template('index.html', stocks=stocks)

def download_and_check_data(stock_symbol):
    data = yf.download(stock_symbol, period='6mo', interval='1d')
    if data.empty:
        return None, "Data tidak tersedia."
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data, None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        stock_symbol = request.form['stock_symbol']
        model_path = f'models/{stock_symbol}_model.pkl'

        if not os.path.exists(model_path):
            return f"Model untuk {stock_symbol} tidak ditemukan."

        model = joblib.load(model_path)
        data, error_message = download_and_check_data(stock_symbol)
        if error_message:
            return error_message

        data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
        data['Return'] = data['Close'].pct_change()
        data_clean = data.dropna(subset=['Close', 'EMA_10', 'Return'])

        if data_clean.empty:
            return "Data tidak cukup untuk prediksi."

        input_data = data_clean[['Close', 'EMA_10', 'Return']].iloc[-1:]
        prediction_value = float(model.predict(input_data)[0])

        now = datetime.now(wib)
        if is_market_open(now):
            close_price = float(data['Close'].iloc[-2])
            prediction_date_obj = now  # prediksi untuk hari ini
        else:
            close_price = float(data['Close'].iloc[-1])
            prediction_date_obj = get_next_trading_day(now + timedelta(days=1))  # prediksi besok

        percentage_change = round(((prediction_value - close_price) / close_price) * 100, 2)

        n = min(len(data_clean), 10)
        accuracy = calculate_accuracy(
            data_clean['Close'][-n:], 
            model.predict(data_clean[['Close', 'EMA_10', 'Return']][-n:])
        )

        prediction_date_str = prediction_date_obj.strftime('%Y-%m-%d')

        # Simpan histori
        new_data = pd.DataFrame([{
            'tanggal': prediction_date_str,
            'saham': stock_symbol,
            'harga_prediksi': prediction_value,
            'before_close': is_market_open(now)  # flag penting untuk history
        }])

        if os.path.exists(history_file):
            history_df = pd.read_csv(history_file)
            history_df = pd.concat([history_df, new_data], ignore_index=True)
        else:
            history_df = new_data

        history_df.drop_duplicates(subset=['tanggal', 'saham'], keep='last', inplace=True)
        history_df.to_csv(history_file, index=False)

        # Grafik
        pred_history = history_df[history_df['saham'] == stock_symbol].copy()
        pred_history['tanggal'] = pd.to_datetime(pred_history['tanggal'])
        pred_history = pred_history.sort_values('tanggal')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data_clean.index.tz_localize(None),
            y=data_clean['Close'],
            mode='lines',
            name='Harga Aktual',
            line=dict(color='blue')
        ))
        if not pred_history.empty:
            fig.add_trace(go.Scatter(
                x=pred_history['tanggal'],
                y=pred_history['harga_prediksi'],
                mode='lines+markers',
                name='Prediksi Harga',
                line=dict(color='red'),
                marker=dict(size=6)
            ))
        fig.update_layout(
            title=f'Grafik Harga Saham dan Prediksi: {stocks[stock_symbol]}',
            xaxis_title="Tanggal",
            yaxis_title="Harga Saham (Rp)",
            legend=dict(x=0, y=1)
        )

        return render_template(
            'result.html',
            stock_name=stocks[stock_symbol],
            stock_symbol=stock_symbol,
            prediction=prediction_value,
            close_price=close_price,
            percentage_change=percentage_change,
            accuracy=accuracy,
            graph_html=fig.to_html(full_html=False),
            prediction_date=prediction_date_str
        )

    except Exception as e:
        return f"Terjadi error: {e}"

@app.route('/history/<stock_symbol>')
def history(stock_symbol):
    if not os.path.exists(history_file):
        return f"Tidak ada histori prediksi untuk {stocks[stock_symbol]}"

    history_df = pd.read_csv(history_file)
    history_df = history_df[history_df['saham'] == stock_symbol].copy()

    if history_df.empty:
        return f"Tidak ada histori prediksi untuk {stocks[stock_symbol]}"

    actual_prices = []
    for idx, row in history_df.iterrows():
        try:
            tanggal_pred = datetime.strptime(row['tanggal'], '%Y-%m-%d')
            if bool(row.get('before_close', False)):
                # prediksi sebelum tutup, ambil harga hari itu
                target_date = tanggal_pred
            else:
                # prediksi setelah tutup, ambil harga besok
                target_date = get_next_trading_day(tanggal_pred)

            data = yf.download(
                stock_symbol,
                start=target_date.strftime('%Y-%m-%d'),
                end=(target_date + timedelta(days=1)).strftime('%Y-%m-%d')
            )
            harga_aktual = float(data['Close'].iloc[0]) if not data.empty else None
        except:
            harga_aktual = None
        actual_prices.append(harga_aktual)

    history_df['harga_aktual'] = actual_prices
    history_df = history_df.sort_values('tanggal', ascending=False)

    return render_template(
        'history.html',
        stock_name=stocks[stock_symbol],
        history_data=history_df.to_dict(orient='records')
    )

if __name__ == '__main__':
    app.run(debug=True)
