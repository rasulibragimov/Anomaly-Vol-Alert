import pandas as pd
import numpy as np


def Q(array, n, alpha):
	'''Quantile alpha-level of [the absolute value of] last n observations'''
	return abs(pd.Series(array)).rolling(n).quantile(alpha)


def ATR(df, window, ohlc=['Open', 'High', 'Low', 'Close']):
	atr = 'ATR_' + str(window)
	if not 'TR' in df.columns:
		df['h-l'] = df[ohlc[1]] - df[ohlc[2]]
		df['h-yc'] = abs(df[ohlc[1]] - df[ohlc[3]].shift())
		df['l-yc'] = abs(df[ohlc[2]] - df[ohlc[3]].shift())
		df['TR'] = df[['h-l', 'h-yc', 'l-yc']].max(axis=1)
		df.drop(['h-l', 'h-yc', 'l-yc'], inplace=True, axis=1)
	EMA(df, 'TR', atr, window, alpha=True)
	return df


def SuperTrend(df, window, multiplier, ohlc=['Open', 'High', 'Low', 'Close']):
	ATR(df, window, ohlc=ohlc)
	atr = 'ATR_' + str(window)
	st = 'ST_' + str(window) + '_' + str(multiplier)
	stx = 'STX_' + str(window) + '_' + str(multiplier)
	df['basic_ub'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 + multiplier * df[atr]
	df['basic_lb'] = (df[ohlc[1]] + df[ohlc[2]]) / 2 - multiplier * df[atr]
	df['final_ub'] = 0.00
	df['final_lb'] = 0.00
	for i in range(window, len(df)):
		df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or \
														 df[ohlc[3]].iat[i - 1] > df['final_ub'].iat[i - 1] else \
			df['final_ub'].iat[i - 1]
		df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or \
														 df[ohlc[3]].iat[i - 1] < df['final_lb'].iat[i - 1] else \
			df['final_lb'].iat[i - 1]
	df[st] = 0.00
	for i in range(window, len(df)):
		df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[
			i] <= df['final_ub'].iat[i] else \
			df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df[ohlc[3]].iat[i] > \
									 df['final_ub'].iat[i] else \
				df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] >= \
										 df['final_lb'].iat[i] else \
					df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df[ohlc[3]].iat[i] < \
											 df['final_lb'].iat[i] else 0.00
	df[stx] = np.where((df[st] > 0.00), np.where((df[ohlc[3]] < df[st]), 'down', 'up'), np.NaN)
	df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)
	df.fillna(0, inplace=True)
	return df


def EMA(df, base, target, window, alpha=False):
	"""
	Function to compute Exponential Moving Average (EMA)

	Args :
		df : Pandas DataFrame which contains ['date', 'open', 'high', 'low', 'close', 'volume'] columns
		base : String indicating the column name from which the EMA needs to be computed from
		target : String indicates the column name to which the computed data needs to be stored
		window : Integer indicates the window of computation in terms of number of candles
		alpha : Boolean if True indicates to use the formula for computing EMA using alpha (default is False)

	Returns :
		df : Pandas DataFrame with new column added with name 'target'
	"""

	con = pd.concat([df[:window][base].rolling(window=window).mean(), df[window:][base]])

	if alpha:
		# (1 - alpha) * previous_val + alpha * current_val where alpha = 1 / window
		df[target] = con.ewm(alpha=1 / window, adjust=False).mean()
	else:
		# ((current_val - previous_val) * coeff) + previous_val where coeff = 2 / (window + 1)
		df[target] = con.ewm(span=window, adjust=False).mean()

	df[target].fillna(0, inplace=True)
	return df


def ROCP(array):
	'''Rate of change Percentage: (price-prev_price)/prev_price * 100'''
	return ((array - pd.Series(array).shift(1)) / pd.Series(array).shift(1)) * 100


def SMA(array, n):
	'''Simple moving average'''
	return pd.Series(array).rolling(n).mean()


def SUMROCP(array, n):
	'''Summ of ROCP on n period'''
	return pd.Series(array).rolling(n).sum()


def RSI(array, n):
	"""Relative strength index"""
	gain = pd.Series(array).diff()
	loss = gain.copy()
	gain[gain < 0] = 0
	loss[loss > 0] = 0
	rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
	return 100 - 100 / (1 + rs)
