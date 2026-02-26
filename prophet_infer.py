import os
import warnings
import logging

# 1. Thread limitations to prevent Anaconda deadlocks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# 2. Force Matplotlib to run in headless mode
import matplotlib
matplotlib.use('Agg') 

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Suppress Prophet's verbose C++ logging
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
warnings.filterwarnings("ignore") 

def main():
    # ==========================================
    # --- HYPERPARAMETERS TO EDIT ---
    # ==========================================
    GROWTH = 'linear'                  # Trend growth type ('linear' or 'logistic')
    YEARLY_SEASONALITY = False           # Built-in yearly seasonality
    WEEKLY_SEASONALITY = False           # Built-in weekly seasonality
    DAILY_SEASONALITY = False            # Built-in daily seasonality
    
    # Custom Annual Seasonality
    CUSTOM_SEASON_PERIOD = 365.25        # Length of custom season
    FOURIER_ORDER = 1                    # Fourier order 
    CUSTOM_PRIOR_SCALE = 0.1             # Prior scale 
    
    # Holiday settings
    INCLUDE_WINTER_HOLIDAYS = True       # Adds December holidays like the notebook
    
    FORECAST_STEPS = 24                  # Number of months to forecast into the future
    # ==========================================

    print("--- Starting Prophet Inference ---")
    
    target_dir = os.path.join("outputs", f"Prophet_{GROWTH.capitalize()}_FO{FOURIER_ORDER}_Prior{CUSTOM_PRIOR_SCALE}_Hol{INCLUDE_WINTER_HOLIDAYS}")
    print(f"Target Directory: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    data_path = os.path.join("data", "order_shipped_dates.csv")
    print(f"Loading data from {data_path}...")
    
    orders_shipped = pd.read_csv(data_path, parse_dates=['event_timestamp'])

    monthly_orders = (
        orders_shipped
            .set_index('event_timestamp')
            .resample('MS')
            .size()
    )

    df_monthly = monthly_orders.reset_index().rename(columns={'event_timestamp': 'ds', 0: 'y'})
    df_monthly = df_monthly.loc[df_monthly['ds'] >= '2021-01-01'].reset_index(drop=True)
    df_monthly['ds'] = df_monthly['ds'].dt.tz_localize(None) 

    if GROWTH == 'logistic':
        df_monthly['cap'] = int(1.1 * df_monthly['y'].max())
        df_monthly['floor'] = 4000

    print(f"Data points after resampling: {len(df_monthly)}")

    print("Fitting Prophet model (this may take a moment)...")
    try:
        # Set up holidays if enabled
        holidays_df = None
        if INCLUDE_WINTER_HOLIDAYS:
            holidays_df = pd.DataFrame({
                'holiday': 'winter_holiday',
                # Adding extra future years to safely cover the forecast period
                'ds': pd.to_datetime(['2021-12-01', '2022-12-01', '2023-12-01', '2024-12-01', '2025-12-01', '2026-12-01'])
            })

        model = Prophet(
            growth=GROWTH, 
            yearly_seasonality=YEARLY_SEASONALITY, 
            weekly_seasonality=WEEKLY_SEASONALITY, 
            daily_seasonality=DAILY_SEASONALITY,
            holidays=holidays_df
        )
        
        model.add_seasonality(
            name='annual',
            period=CUSTOM_SEASON_PERIOD,
            fourier_order=FOURIER_ORDER,
            prior_scale=CUSTOM_PRIOR_SCALE
        )
        
        model.fit(df_monthly)
        print("MODEL FITTED SUCCESSFULLY!")
        
        print("\nGenerating fitted values and future forecasts...")
        
        future_dates = model.make_future_dataframe(periods=FORECAST_STEPS, freq='MS')
        
        if GROWTH == 'logistic':
            future_dates['cap'] = df_monthly.loc[0, 'cap']
            future_dates['floor'] = df_monthly.loc[0, 'floor']
        
        forecast_df = model.predict(future_dates)
        
        fitted_values = forecast_df.iloc[:-FORECAST_STEPS]
        future_predictions = forecast_df.iloc[-FORECAST_STEPS:]
        
        print("\nSaving results to target directory...")
        
        summary_path = os.path.join(target_dir, "model_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("--- Prophet Model Parameters ---\n")
            f.write(f"Growth Type: {GROWTH}\n")
            f.write(f"Yearly Seasonality: {YEARLY_SEASONALITY}\n")
            f.write(f"Weekly Seasonality: {WEEKLY_SEASONALITY}\n")
            f.write(f"Daily Seasonality: {DAILY_SEASONALITY}\n")
            f.write(f"Custom Annual Fourier Order: {FOURIER_ORDER}\n")
            f.write(f"Winter Holidays Included: {INCLUDE_WINTER_HOLIDAYS}\n")
            if 'k' in model.params:
                f.write(f"\nFitted Growth rate (k): {model.params['k'][0][0]}\n")
            if 'm' in model.params:
                f.write(f"Fitted Offset (m): {model.params['m'][0][0]}\n")
            f.write(f"\nForecasted future periods: {FORECAST_STEPS} months\n")
            
        fitted_csv_path = os.path.join(target_dir, "fitted_values.csv")
        fitted_values[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(fitted_csv_path, index=False)
        
        future_csv_path = os.path.join(target_dir, "future_predictions.csv")
        future_predictions[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(future_csv_path, index=False)
        
        fig1 = model.plot(forecast_df, figsize=(12, 6))
        plt.title(f'Prophet Forecast ({GROWTH.capitalize()}): Actuals vs Fitted vs Future')
        plt.xlabel('Date')
        plt.ylabel('Total Completed Journeys')
        plot_path = os.path.join(target_dir, "prophet_forecast_plot.png")
        fig1.savefig(plot_path)
        plt.close(fig1)

        fig2 = model.plot_components(forecast_df, figsize=(10, 6))
        components_path = os.path.join(target_dir, "prophet_components_plot.png")
        fig2.savefig(components_path)
        plt.close(fig2)

        print("\n--- Inference Complete! Check the outputs folder for your files. ---")
        
    except Exception as e:
        import traceback
        print("\n!!! ERROR DURING MODEL FITTING !!!")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()