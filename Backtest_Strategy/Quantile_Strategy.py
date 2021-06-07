from backtesting import Backtest, Strategy
from Get_Data import Trades
import pandas as pd
import numpy as np
from Indicators import Q, ATR, SuperTrend, EMA, ROCP, SMA, SUMROCP, RSI
import asyncio
import os


class Quantile_Simple(Strategy):
	n_volume = 125
	n_rocp = 175
	volume_quantile = 0.9
	rocp_quantile = 0.9995
	ema_w1 = 10
	n_summrocp = 2
	summ_rocp_t = 0
	m = 5
	n = 14
	rsi_threshold = 60
	rsi_period = 165
	N = 4
	M = 1

	def init(self):
		self.volume_quantile = self.I(Q, self.data.Volume, self.n_volume, self.volume_quantile)
		self.rocp = self.I(ROCP, self.data.Close)
		self.rocp_quantile = self.I(Q, self.rocp, self.n_rocp, self.rocp_quantile)
		self.sum_rocp = self.I(SUMROCP, self.rocp, self.n_summrocp)
		self.rsi = self.I(RSI, self.data.Close, self.rsi_period)

	def next(self):
		vol = self.data.Volume[-1]
		price = self.data.Close[-1]
		rocp = self.rocp[-1]
		rocp_quantile = self.rocp_quantile[-1]
		sum_rocp = self.sum_rocp[-1]
		super_trend = self.data.STX_10_1
		rsi = self.rsi[-1]
		if (not self.position and
				vol > self.volume_quantile[-1] * self.N and
				# super_trend == 'up' and
				rocp > 0):
			self.buy()

		if (self.position and
				rocp > rocp_quantile * self.M):
			self.position.close()

# Мой REFRESH_TOKEN для проверки работоспособности
# должен быть передан в e-mail
REFRESH_TOKEN = os.getenv('REFRESH_TOKEN', None)
alor = Trades(refresh_token=REFRESH_TOKEN)
df = asyncio.run(alor.get_tick('AFKS', start='2021-01-01', finish='2021-05-31', tf=900))
df = SuperTrend(df, window=10, multiplier=1)

backtest = Backtest(df, Quantile_Simple, commission=.01)
stats = backtest.run()
# Полная статистика:
# print(stats)
stats.to_csv('Report.csv')
backtest.plot()

print('DIFF %', stats['Return [%]'] - stats['Buy & Hold Return [%]'])
print('Return', stats['Return [%]'])
print('BH', stats['Buy & Hold Return [%]'])
print('Trades', stats['# Trades'])

m_range = list(np.arange(1, 5, 0.5))

stats, heatmap = backtest.optimize(
	N=m_range,
	M=m_range,
	# n_rocp = range(5, 300, 10),
	# volume_quantile = [0.9, 0.99, 0.9995, 0.9999],
	# rocp_quantile = [0.9, 0.99, 0.9995, 0.9999],
	maximize='Equity Final [$]',
	max_tries=1000,
	random_state=0,
	return_heatmap=True)

print('DIFF %', stats['Return [%]'] - stats['Buy & Hold Return [%]'])
print('Return', stats['Return [%]'])
print('BH', stats['Buy & Hold Return [%]'])
print('Trades', stats['# Trades'])
print('Equity Final [$]', stats['Equity Final [$]'])
# print(heatmap)
heatmap.to_csv('Optimization.csv')
