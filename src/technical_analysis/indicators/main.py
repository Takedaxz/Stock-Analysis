import papermill as pm

#Configuration
TICKER = "NVDA"
start_date = "2024-01-01"
end_date = "2025-07-09"


#Notebook list
notebook_list = [
    "bollinger_bands.ipynb",
    "chart_patterns.ipynb",
    "ema.ipynb",
    "elliott_wave.ipynb",
    "ichimoku_cloud.ipynb",
    "macd.ipynb",
    "momentum.ipynb",
    "rsi.ipynb",
    "sma.ipynb",
    "stochastic_oscillator.ipynb",
    "support_resistance.ipynb",
    "volume.ipynb",
]

Ouput_list = [
    "bollinger_bands_output.ipynb",
    "chart_patterns_output.ipynb",
    "ema_output.ipynb",
    "elliott_wave_output.ipynb",
    "ichimoku_cloud_output.ipynb",
    "macd_output.ipynb",
    "momentum_output.ipynb",
    "rsi_output.ipynb",
    "sma_output.ipynb",
    "stochastic_oscillator_output.ipynb",
    "support_resistance_output.ipynb",
    "volume_output.ipynb",
]

#Run all notebooks
for i in range(len(notebook_list)):
    input_notebook_path = f"{notebook_list[i]}"
    output_notebook_path = f"{Ouput_list[i]}"
    print(f"Running {input_notebook_path}...")
    pm.execute_notebook(
        input_path=input_notebook_path,
        output_path='output/'+output_notebook_path,
        parameters={"ticker": TICKER, "start_date": start_date, "end_date": end_date}
    )
    print(f"--- Successfully executed '{input_notebook_path}'. ---")
