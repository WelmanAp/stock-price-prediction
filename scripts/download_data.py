import yfinance as yf
import pandas as pd
import os

def download_and_process_stock_data(ticker, start_date, end_date, output_folder):
    # Unduh data
    stock_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

    if stock_data.empty:
        print(f"Data untuk {ticker} tidak tersedia.")
        return

    stock_data = stock_data.dropna()

    # Jika kolom MultiIndex, ubah ke single index
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)

    # Tambahkan kolom fitur
    stock_data['EMA_10'] = stock_data['Close'].ewm(span=10, adjust=False).mean()
    stock_data['Return'] = stock_data['Close'].pct_change()
    stock_data = stock_data.dropna()

    # Reset index agar kolom Date jadi kolom biasa
    stock_data = stock_data.reset_index()
    stock_data['Date'] = stock_data['Date'].dt.date

    # Pembulatan
    stock_data['EMA_10'] = stock_data['EMA_10'].round(2)
    stock_data['Return'] = stock_data['Return'].round(3)

    # Hapus kolom Adj Close karena tidak digunakan
    if 'Adj Close' in stock_data.columns:
        stock_data = stock_data.drop(columns=['Adj Close'])

    # Simpan ke Excel
    output_file = os.path.join(output_folder, f"{ticker}.xlsx")
    stock_data.to_excel(output_file, index=False)
    print(f"Data untuk {ticker} telah diproses dan disimpan ke {output_file}.")

# Daftar saham
stocks = [
    'ADRO.JK', 'ASII.JK', 'BBCA.JK', 'BBNI.JK', 'BBRI.JK',
    'BMRI.JK', 'ICBP.JK', 'PTBA.JK', 'TLKM.JK', 'TOWR.JK'
]

start_date = "2020-06-30"
end_date = "2025-08-01"
#1 juli 2020 - 31 juli 2025
output_folder = "data"
os.makedirs(output_folder, exist_ok=True)

# Proses semua saham
for stock in stocks:
    download_and_process_stock_data(stock, start_date, end_date, output_folder)
