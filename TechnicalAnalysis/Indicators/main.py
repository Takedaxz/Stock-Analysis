import papermill as pm

#Configuration
TICKER = "AMZN"
start_date = "2025-01-01"
end_date = "2025-05-25"


#Notebook list
notebook_list = [
    "BollingerBands.ipynb",
    "Chart_Patterns.ipynb",
    "EMA.ipynb",
    "Elliott_Wave.ipynb",
    "Ichimoku_Cloud.ipynb",
    "MACD.ipynb",
    "Momentum.ipynb",
    "RSI.ipynb",
    "SMA.ipynb",
    "Stochastic_Oscillator.ipynb",
    "SupportResistance.ipynb",
    "Volume.ipynb",
]

Ouput_list = [
    "BollingerBands_output.ipynb",
    "Chart_Patterns_output.ipynb",
    "EMA_output.ipynb",
    "Elliott_Wave_output.ipynb",
    "Ichimoku_Cloud_output.ipynb",
    "MACD_output.ipynb",
    "Momentum_output.ipynb",
    "RSI_output.ipynb",
    "SMA_output.ipynb",
    "Stochastic_Oscillator_output.ipynb",
    "SupportResistance_output.ipynb",
    "Volume_output.ipynb",
]

#Run all notebooks
for i in range(len(notebook_list)):
    input_notebook_path = f"{notebook_list[i]}"
    output_notebook_path = f"{Ouput_list[i]}"
    print(f"Running {input_notebook_path}...")
    pm.execute_notebook(
        input_path=input_notebook_path,
        output_path='Output/'+output_notebook_path,
        parameters={"ticker": TICKER, "start_date": start_date, "end_date": end_date}
    )
    print(f"--- Successfully executed '{input_notebook_path}'. ---")
