# STOCK PRICE PREDICTION by WelmanApridius

## ✨ Key Features
- ✅ **Stock Data Retrieval** → Uses **Yahoo Finance API** to fetch historical and **real-time** stock price data.  
- ✅ **Stock Price Prediction** → **Random Forest** model with **EMA-10 Daily** as a technical indicator.  
- ✅ **Interactive Data Visualization** → Compares actual vs. predicted prices using **Plotly**.  
- ✅ **Dynamic Dashboard** → Users can select stocks and instantly view predictions through a **Bootstrap-based interface**.  
- ✅ **Model Evaluation** → Performance measured using **Mean Absolute Percentage Error (MAPE)**.  
- ✅ **Prediction History** → Prediction results are automatically stored in a **CSV file (`prediksi_history.csv`)** and can be revisited on the *History* page.  

> ⚡ **Note**  
> - The contents of the **`data`** and **`models`** folders are dynamically generated.  
> - You can configure the stock list and date range inside **`scripts/download_data.py`**:
>   ```python
>   start_date = "2020-06-30"
>   end_date = "2025-08-01"
>   output_folder = "data"
>   os.makedirs(output_folder, exist_ok=True)
>   ```
> - Adjust these values as needed.  
> - The script **`scripts/train_models.py`** will then automatically use the updated `data` to train the models and save them into the **`models`** folder.
