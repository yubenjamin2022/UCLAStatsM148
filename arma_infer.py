import os
import warnings

# 1. Thread limitations to prevent Anaconda deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# 2. Force Matplotlib to run in headless mode
import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore") 

def main():
    # ==========================================
    # --- HYPERPARAMETERS TO EDIT ---
    # ==========================================
    P_ORDER = 1                      # AR order (p)
    D_ORDER = 1                      # Differencing order (d)
    Q_ORDER = 1                      # MA order (q)
    
    SEASONAL_P = 1                   # Seasonal AR order (P)
    SEASONAL_D = 1                   # Seasonal Differencing (D)
    SEASONAL_Q = 1                   # Seasonal MA order (Q)
    SEASONAL_PERIODS = 52            # Weeks in a year (S)
    
    FORECAST_STEPS = 52              # Weeks to predict into the future
    # ==========================================

    print("--- Starting ARMA Inference ---")
    
    target_dir = os.path.join("outputs", f"ARMA_p{P_ORDER}_d{D_ORDER}_q{Q_ORDER}_sp_{SEASONAL_P}_sd_{SEASONAL_D}_sq_{SEASONAL_Q}_W")
    print(f"Target Directory: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    data_path = os.path.join("data", "order_shipped_dates.csv")
    print(f"Loading data from {data_path}...")
    
    orders_shipped = pd.read_csv(data_path, parse_dates=['event_timestamp'])

    weekly_totals = (
        orders_shipped
            .set_index('event_timestamp')
            .resample('W')
            .size()
            .asfreq('W', fill_value=0)
    )

    print(f"Data points after resampling: {len(weekly_totals)}")

    print("Fitting model (this may take a moment)...")
    try:
        model = ARIMA(
            weekly_totals,
            order=(P_ORDER, D_ORDER, Q_ORDER),
            seasonal_order=(SEASONAL_P, SEASONAL_D, SEASONAL_Q, SEASONAL_PERIODS),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        res = model.fit(method_kwargs={'method': 'nm', 'maxiter': 1000})
        
        print("\n" + "="*40)
        print("MODEL FITTED SUCCESSFULLY! SUMMARY:")
        print("="*40)
        print(res.summary())
        
        print("\nGenerating fitted values and future forecasts...")
        
        fitted_values = res.predict()
        
        forecast_obj = res.get_forecast(steps=FORECAST_STEPS)
        future_predictions = forecast_obj.predicted_mean
        
        print("\nSaving results to target directory...")
        
        summary_path = os.path.join(target_dir, "model_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(res.summary().as_text())
            
        fitted_csv_path = os.path.join(target_dir, "fitted_values.csv")
        fitted_values.to_csv(fitted_csv_path, header=["Fitted_Values"])
        
        future_csv_path = os.path.join(target_dir, "future_predictions.csv")
        future_predictions.to_csv(future_csv_path, header=["Future_Forecast"])
        
        plt.figure(figsize=(14, 5))
        weekly_totals.plot(label='Actual Data', color='blue')
        fitted_values.plot(color='red', linestyle='--', label='In-Sample Fitted')
        future_predictions.plot(color='orange', linewidth=2, label=f'Future Forecast ({FORECAST_STEPS} Weeks)')
        
        plt.xlabel('Date')
        plt.ylabel('Total Completed Journeys')
        plt.title('Completed Journeys: Actuals vs Fitted vs Future Forecast')
        plt.legend(loc='upper left')
        
        plot_path = os.path.join(target_dir, "model_forecast_plot.png")
        plt.savefig(plot_path)
        plt.close()

        print("\n--- Inference Complete! Check the outputs folder for your files. ---")
        
    except Exception as e:
        import traceback
        print("\n!!! ERROR DURING MODEL FITTING !!!")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()