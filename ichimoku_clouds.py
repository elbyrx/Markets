# -*- coding: utf-8 -*-
"""Ichimoku Clouds.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WrZufyNSH1ITchQhOrt1LsCTZlqfNtey
"""

import yfinance as yf
import pandas as pd

# Define stock symbols
#NASDAQ:
#stocks = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'GOOGL', 'AVGO', 'GOOG', 'TSLA', 'PEP', 'COST', 'ADBE', 'CSCO', 'CMCSA', 'NFLX', 'AMD', 'TMUS', 'TXN', 'AMGN', 'INTC', 'INTU', 'AMAT', 'HON', 'QCOM', 'BKNG', 'SBUX', 'ADP', 'ISRG', 'MDLZ', 'GILD', 'REGN', 'VRTX', 'LRCX', 'ADI', 'PANW', 'MU', 'PYPL', 'KLAC', 'SNPS', 'MELI', 'CHTR', 'MAR', 'CSX', 'CDNS', 'MNST', 'ASML', 'ORLY', 'ABNB', 'NXPI', 'MRVL', 'CTAS', 'PDD', 'KDP', 'WDAY', 'LULU', 'FTNT', 'ODFL', 'PCAR', 'MRNA', 'MCHP', 'PAYX', 'ADSK', 'CPRT', 'DXCM', 'KHC', 'AZN', 'IDXX', 'AEP', 'ROST', 'ON', 'EXC', 'BIIB', 'SGEN', 'BKR', 'CTSH', 'CEG', 'CRWD', 'VRSK', 'TTD', 'EA', 'FAST', 'CSGP', 'XEL', 'GEHC', 'DLTR', 'WBD', 'GFS', 'TEAM', 'DDOG', 'ALGN', 'FANG', 'ILMN', 'ANSS', 'EBAY', 'WBA', 'ZS', 'ENPH', 'SIRI', 'ZM', 'JD', 'LCID']
#S&P500
stocks = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'UNH', 'XOM', 'LLY', 'JNJ', 'JPM', 'V', 'PG', 'AVGO', 'MA', 'HD', 'CVX', 'MRK', 'ABBV', 'PEP', 'COST', 'ADBE', 'KO', 'CSCO', 'WMT', 'PFE', 'MCD', 'TMO', 'CRM', 'BAC', 'ACN', 'CMCSA', 'LIN', 'NFLX', 'ABT', 'ORCL', 'AMD', 'DHR', 'DIS', 'WFC', 'TXN', 'PM', 'COP', 'VZ', 'CAT', 'AMGN', 'INTC', 'INTU', 'NEE', 'UNP', 'LOW', 'BA', 'BMY', 'IBM', 'SPGI', 'NKE', 'RTX', 'AMAT', 'HON', 'QCOM', 'GE', 'UPS', 'NOW', 'BKNG', 'PLD', 'MDT', 'SBUX', 'ELV', 'MS', 'GS', 'DE', 'ADP', 'TJX', 'LMT', 'T', 'ISRG', 'BLK', 'MDLZ', 'GILD', 'AXP', 'MMC', 'SYK', 'REGN', 'VRTX', 'LRCX', 'ADI', 'ETN', 'CVS', 'ZTS', 'SCHW', 'CI', 'CB', 'AMT', 'SLB', 'C', 'BDX', 'TMUS', 'MO', 'PGR', 'EOG', 'FI', 'SO', 'BSX', 'CME', 'PANW', 'EQIX', 'MU', 'DUK', 'PYPL', 'KLAC', 'SNPS', 'AON', 'ITW', 'ATVI', 'SHW', 'ICE', 'APD', 'NOC', 'CSX', 'CDNS', 'CL', 'MPC', 'HUM', 'FDX', 'WM', 'TGT', 'ORLY', 'MCK', 'HCA', 'FCX', 'PXD', 'EMR', 'MMM', 'MAR', 'PSX', 'CMG', 'NXPI', 'ROP', 'MCO', 'PH', 'APH', 'GD', 'USB', 'PNC', 'AJG', 'NSC', 'VLO', 'F', 'ANET', 'MSI', 'EW', 'GM', 'OXY', 'TT', 'AZO', 'CARR', 'SRE', 'ECL', 'TDG', 'ADM', 'PCAR', 'MCHP', 'MNST', 'ADSK', 'PSA', 'KMB', 'CHTR', 'CCI', 'NUE', 'HES', 'MSCI', 'WMB', 'CTAS', 'STZ', 'AIG', 'DXCM', 'TEL', 'ROST', 'IDXX', 'AFL', 'GIS', 'AEP', 'JCI', 'HLT', 'D', 'ON', 'EXC', 'WELL', 'IQV', 'MET', 'MRNA', 'COF', 'PAYX', 'O', 'BIIB', 'DOW', 'FTNT', 'CPRT', 'TFC', 'TRV', 'ODFL', 'DHI', 'YUM', 'SPG', 'DLR', 'SYY', 'AME', 'CTSH', 'CTVA', 'BKR', 'CNC', 'DG', 'HAL', 'A', 'CEG', 'EL', 'OTIS', 'AMP', 'LHX', 'KMI', 'DD', 'VRSK', 'ROK', 'PRU', 'CMI', 'FIS', 'GPN', 'FAST', 'PPG', 'CSGP', 'DVN', 'XEL', 'GWW', 'HSY', 'EA', 'ED', 'BK', 'NEM', 'KR', 'URI', 'VICI', 'PEG', 'RSG', 'LEN', 'PWR', 'DLTR', 'WST', 'OKE', 'ABC', 'VMC', 'KDP', 'ALL', 'WBD', 'ACGL', 'CDW', 'FANG', 'MLM', 'IR', 'FTV', 'PCG', 'GEHC', 'HPQ', 'WEC', 'EXR', 'AWK', 'DAL', 'EIX', 'KHC', 'IT', 'APTV', 'MTD', 'ANSS', 'ILMN', 'CBRE', 'ALGN', 'AVB', 'LYB', 'GLW', 'ZBH', 'WY', 'TROW', 'RMD', 'TSCO', 'XYL', 'EFX', 'SBAC', 'EBAY', 'KEYS', 'CHD', 'DFS', 'MPWR', 'TTWO', 'ES', 'STE', 'HIG', 'STT', 'CAH', 'ULTA', 'ALB', 'DTE', 'RCL', 'HPE', 'GPC', 'EQR', 'WTW', 'MTB', 'FICO', 'CTRA', 'BAX', 'BR', 'AEE', 'MKC', 'WAB', 'ETR', 'RJF', 'DOV', 'FE', 'FLT', 'HOLX', 'INVH', 'TDY', 'DRI', 'WBA', 'LH', 'TRGP', 'LUV', 'VRSN', 'CLX', 'PPL', 'MOH', 'NVR', 'COO', 'ARE', 'HWM', 'CNP', 'PHM', 'NDAQ', 'EXPD', 'J', 'FSLR', 'LVS', 'ENPH', 'IRM', 'RF', 'IFF', 'FITB', 'PFG', 'BRO', 'STLD', 'SWKS', 'VTR', 'ATO', 'BG', 'NTAP', 'IEX', 'MRO', 'CMS', 'FDS', 'BALL', 'CINF', 'MAA', 'UAL', 'OMC', 'TER', 'WAT', 'JBHT', 'GRMN', 'EQT', 'CBOE', 'CCL', 'TYL', 'K', 'EXPE', 'NTRS', 'TSN', 'AKAM', 'TXT', 'HBAN', 'PTC', 'CF', 'ESS', 'SJM', 'EG', 'DGX', 'BBY', 'AVY', 'RVTY', 'SNA', 'CAG', 'AXON', 'AMCR', 'PAYC', 'SYF', 'ZBRA', 'EPAM', 'PODD', 'LW', 'SWK', 'POOL', 'DPZ', 'VTRS', 'MOS', 'APA', 'LKQ', 'CFG', 'PKG', 'EVRG', 'LDOS', 'TRMB', 'WDC', 'MGM', 'STX', 'KMX', 'LNT', 'NDSN', 'MAS', 'MTCH', 'IPG', 'WRB', 'TECH', 'IP', 'BF', 'INCY', 'AES', 'L', 'LYV', 'GEN', 'TAP', 'UDR', 'CE', 'CPT', 'JKHY', 'HST', 'KIM', 'HRL', 'FMC', 'CHRW', 'CZR', 'PEAK', 'PNR', 'CDAY', 'NI', 'HSIC', 'TFX', 'CRL', 'GL', 'QRVO', 'WYNN', 'EMN', 'KEY', 'AAL', 'ALLE', 'BWA', 'MKTX', 'REG', 'FFIV', 'ETSY', 'SEDG', 'ROL', 'JNPR', 'AOS', 'FOXA', 'PNW', 'BXP', 'HII', 'NRG', 'HAS', 'CPB', 'UHS', 'RHI', 'XRAY', 'BIO', 'NWSA', 'WRK', 'CTLT', 'TPR', 'BBWI', 'WHR', 'AIZ', 'PARA', 'BEN', 'NCLH', 'GNRC', 'FRT', 'IVZ', 'VFC', 'CMA', 'DVA', 'OGN', 'ALK', 'SEE', 'ZION', 'MHK', 'DXC', 'RL', 'FOX', 'AAP', 'LNC', 'NWL', 'NWS', 'FTRE']


# Define Ichimoku Cloud parameters
conversion_period = 9
base_period = 26
lagging_span2_period = 52
displacement = 26

# Create an empty dictionary to store results
all_stocks_data = {}

# Download and process data for each stock
for stock_symbol in stocks:
    stock_data = yf.download(stock_symbol, start='2022-01-01', end=pd.Timestamp.now() - pd.Timedelta(days=1))

    # Calculate Ichimoku Cloud indicators
    stock_data['ConversionLine'] = (stock_data['High'].rolling(window=conversion_period).max() + stock_data['Low'].rolling(window=conversion_period).min()) / 2
    stock_data['BaseLine'] = (stock_data['High'].rolling(window=base_period).max() + stock_data['Low'].rolling(window=base_period).min()) / 2
    stock_data['LeadingSpanA'] = ((stock_data['ConversionLine'] + stock_data['BaseLine']) / 2).shift(displacement)
    stock_data['LeadingSpanB'] = ((stock_data['High'].rolling(window=lagging_span2_period).max() + stock_data['Low'].rolling(window=lagging_span2_period).min()) / 2).shift(displacement)
    stock_data['LaggingSpan'] = stock_data['Close'].shift(-displacement)

    # Define a function to generate buy or sell signals (TENKEN x KIJUN)
    def generate_signals(row):
        if row['ConversionLine'] > row['BaseLine']:
            return 'Buy'
        elif row['ConversionLine'] < row['BaseLine']:
            return 'Sell'
        else:
            return 'Hold'

    # # Define a function to generate buy or sell signals (CLOUD CROSSING)
    # def generate_signals(row):
    #     if row['Close'] > row['LeadingSpanA'] and row['Close'] > row['LeadingSpanB']:
    #         return 'Buy'
    #     elif row['Close'] < row['LeadingSpanA'] and row['Close'] < row['LeadingSpanB']:
    #         return 'Sell'
    #     else:
    #         return 'Hold'



    # Apply the signal generation function to each row
    stock_data['Signal'] = stock_data.apply(generate_signals, axis=1)

    all_stocks_data[stock_symbol] = stock_data

# Find and print stocks that changed signals in the last 2 days
for stock_symbol, stock_data in all_stocks_data.items():
    signal_changes = stock_data['Signal'].ne(stock_data['Signal'].shift())
    if signal_changes.tail(2).any() and stock_data['Signal'].iloc[-1] == 'Buy':
        last_signal_change = signal_changes[signal_changes].index[-1]
        last_signal_change_type = stock_data.loc[last_signal_change, 'Signal']
        print(f"{stock_symbol} - Last Signal Change: {last_signal_change}, Type: {last_signal_change_type}")
        print("\n")