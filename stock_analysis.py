# Calculate MACD components
macd_data = ta.macd(df.Close)
df['MACD'] = macd_data['MACD_12_26_9']  # MACD line
df['MACD_signal'] = macd_data['MACDs_12_26_9']  # Signal line
df['MACD_hist'] = macd_data['MACDh_12_26_9']  # MACD histogram 