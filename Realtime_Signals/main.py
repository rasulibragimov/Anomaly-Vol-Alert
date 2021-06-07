'''
На данный момент модуль достаточно сырой, например, некоторые функции
до сих пор синхронные. Тем не менее, модуль рабочий, он проработал на сервере heroku
более недели без падений и остановок, посты были опубликованы в телеграм канале:
https://t.me/joinchat/28bkkQ-1GqszOThi

Также стоит отметить, что стратегия, по которой скрипт отправляет сигналы в канал,
далеко не лучшая на текущий момент. Лучшая реализация есть в пакете Backtest_Strategy
'''
import requests
import json
import os
from datetime import datetime, date, timedelta
from json import JSONDecodeError
import urllib.parse
import time
import pandas as pd
from numpy import quantile as np_quantile
from collections import OrderedDict
from tickers import TICKERS
import math
import asyncio
import aiohttp
from aiogram import Bot, types


class AsyncIteratorWrapper:
	def __init__(self, obj):
		self._it = iter(obj)

	def __aiter__(self):
		return self

	async def __anext__(self):
		try:
			value = next(self._it)
		except StopIteration:
			raise StopAsyncIteration
		return value


class MMTS_Strategy:
	def __init__(self, last_vols, last_closes, new_tick, rsi_n, quantile_level=0.9975, rsi_threshold=50):
		self.new_vol = new_tick['sum_vol']
		self.new_close = new_tick['last_close']
		self.rsi_threshold = rsi_threshold
		self.current_rsi = self._rsi(last_vols, rsi_n)
		self.current_vol_q = np_quantile(last_vols, quantile_level)
		self.closes_rocp = self._current_rocp(last_close=self.new_close, prev_close=last_closes[-1])
		self.last_rocp = self._last_rocps(last_closes=last_closes)
		self.current_rocp_q = np_quantile(self.last_rocp, quantile_level)

	def buy(self):
		if (self.new_vol > self.current_vol_q and
				self.current_rsi > self.rsi_threshold and
				self.closes_rocp > 0):
			return True
		return False

	def sell(self):
		if (self.new_vol > self.current_vol_q and
				self.closes_rocp < -1 * self.current_rocp_q):
			return True
		return False

	def _rsi(self, array, n):
		gain = pd.Series(array).diff()
		loss = gain.copy()
		gain[gain < 0] = 0
		loss[loss > 0] = 0
		rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean()
		return float(list(100 - 100 / (1 + rs))[-1])

	def _last_rocps(self, last_closes):
		rocps = []
		for i in range(len(last_closes) - 1):
			rocps.append(self._current_rocp(last_closes[i], last_closes[i + 1]))
		return rocps

	def _current_rocp(self, prev_close, last_close):
		'''Rate of change Percentage: (price-prev_price)/prev_price * 100'''
		return round((last_close - prev_close) / prev_close * 100, 2)


class Trades:
	def __init__(self, refresh_token, watchlist, N):
		self.refresh_token = refresh_token
		self.watchlist = watchlist
		self.N = N
		self.data = self._gen_data()
		self.jwt_token = self._get_jwt_token()
		while self.jwt_token == None:
			time.sleep(15)
			print('Trying to get token again....')
			self.jwt_token = self._get_jwt_token()
		# Необходимо добавить токен
		self.bot = Bot(token=os.getenv('TOKEN', None), parse_mode=types.ParseMode.HTML)

	def _current_utc(self):
		return str(int(time.time()))

	def _utc_to_time(self, utc):
		that_minute = datetime.utcfromtimestamp(int(utc))
		next_minute = (that_minute + timedelta(minutes=1)).strftime('%H%M')
		return next_minute

	def _utc_msk(self, time):
		# format 1410 HHMM
		datetime_object = datetime.strptime(str(time), '%H%M')
		msk_time = (datetime_object + timedelta(hours=3)).strftime('%H%M')
		return str(msk_time)

	def _gen_data(self):
		data = {}
		for ticker in self.watchlist:
			data[ticker] = {'ticks': {'time': None},
							'minutes': OrderedDict(),
							'last_utc': None}
		return data

	def _get_jwt_token(self):
		payload = {'token': self.refresh_token}
		res = requests.post(
			url=f'https://oauth.alor.ru/refresh',
			params=payload
		)
		if res.status_code != 200:
			print(res)
			print(f'Ошибка получения JWT токена: {res.status_code}')
			return None
		try:
			token = res.json()
			jwt = token.get('AccessToken')
			return jwt
		except JSONDecodeError as e:
			print(f'Ошибка декодирования JWT токена: {e}')
			return None

	def _check_results(self, res):
		if res.status_code != 200:
			print(f'Ошибка: {res.status_code} {res.text}')
			return
		try:
			result = json.loads(res.content)
			return result
		except JSONDecodeError as e:
			if LOGGING:
				logging.error(f'Ошибка декодирования JSON: {e}')
			else:
				print(f'Ошибка декодирования JSON: {e}')

	def _is_realtime(self, signal_time):
		# Функция для проверки поступил сигнал недавно или нет,
		# на данный момент заглушка
		current_time = self._utc_to_time(self._current_utc())
		return True

	async def _aiotg_channel(self, channel_id: int, text: str):
		await self.bot.send_message(channel_id, text)

	def _num_format(self, n):
		millnames = ['', 'K', 'M', 'B', 'T']
		n = float(n)
		millidx = max(0, min(len(millnames) - 1,
							 int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))

		return '{:.0f}{}'.format(n / 10 ** (3 * millidx), millnames[millidx])

	def _gen_message(self, side, ticker, time, sum_vol, last_close, prev_close, sum_buy, sum_sell, rsi):
		buy_p = int(round(sum_buy / sum_vol * 100, 0))
		sell_p = int(round(sum_sell / sum_vol * 100, 0))
		rsi = int(round(rsi, 0))
		time = self._utc_msk(time)
		time = time[:2] + ':' + time[2:]
		rocp = round((last_close - prev_close) / prev_close * 100, 2)
		if str(rocp).count('-') == 0:
			rocp = f'+{rocp}'
		rub_vol = float(sum_vol) * float(last_close)
		rub_vol = self._num_format(rub_vol)
		if side == 'buy':
			message = f'<b>#{ticker}\n✳️ ПОКУПКА</b>\n<b>Изменение цены: {rocp}%\nОбъём: {rub_vol} ₽ ({sum_vol} лотов)\n</b>' \
					  f'Покупка: {buy_p}% Продажа: {sell_p}%\nЦена: {last_close}₽\nRSI: {rsi}\nВремя: {time}'
		if side == 'sell':
			message = f'<b>#{ticker}\n🔴️ ПРОДАЖА</b>\n<b>Изменение цены: {rocp}%\nОбъём: {rub_vol} ₽ ({sum_vol} лотов)\n</b>' \
					  f'Покупка: {buy_p}% Продажа: {sell_p}%\nЦена: {last_close}₽\nRSI: {rsi}\nВремя: {time}'
		return message

	async def _broadcast(self, ticker, side, rsi=None):
		# now in ticks we actually store all minute values
		ticks = self.data[ticker]['ticks']
		minutes = self.data[ticker]['minutes']
		if not self._is_realtime(ticks['time']):
			tick_t = ticks['time']
			current_time = self._utc_to_time(self._current_utc())
			print(f'OLD post {ticker} tick_t {tick_t} current_time {current_time}')
			return
		prev_close = minutes[next(reversed(minutes))]['last_close']

		print(side, ticker, ticks['time'], ticks['sum_vol'], ticks['last_close'], prev_close, ticks['sum_buy'],
			  ticks['sum_sell'])
		try:
			await self._aiotg_channel(-1001343960631, self._gen_message(side, ticker, ticks['time'], ticks['sum_vol'],
																		ticks['last_close'], prev_close,
																		ticks['sum_buy'],
																		ticks['sum_sell'], rsi=rsi))
		except Exception as e:
			print(f'ERROR in TG {e}')

	async def _check_sig(self, ticker):
		last_vols = [i['sum_vol'] for i in self.data[ticker]['minutes'].values()]
		# if there is no data in minutes yet
		# or number of observations is not enougth
		if len(last_vols) < self.N - 1:
			return
		last_closes = [i['last_close'] for i in self.data[ticker]['minutes'].values()]
		mmts = MMTS_Strategy(last_vols=last_vols, last_closes=last_closes,
							 new_tick=self.data[ticker]['ticks'], rsi_n=self.N)
		if mmts.buy():
			await self._broadcast(ticker, side='buy', rsi=mmts.current_rsi)
		if mmts.sell():
			await self._broadcast(ticker, side='sell', rsi=mmts.current_rsi)

	def _aggregate(self, ticker):
		ticks = self.data[ticker]['ticks']
		self.data[ticker]['minutes'][ticks['time']] = {'time': ticks['time'],
													   'sum_vol': ticks['sum_vol'],
													   'sum_buy': ticks['sum_buy'],
													   'sum_sell': ticks['sum_sell'],
													   'last_close': ticks['last_close']}

	async def _handle_trades(self, ticker, tds: list):
		utc = 0
		if not isinstance(tds, list):
			return
		for trade in tds:
			# print(trade)
			utc = trade['timestamp']
			volume = trade['qty']
			price = trade['price']
			side = trade.get('side')
			buy_vol = volume if side == 'buy' else 0
			sell_vol = volume if side == 'sell' else 0
			trade_time = self._utc_to_time(int(utc) / 1000)
			cached_time = self.data[ticker]['ticks']['time']
			# we are in the same minute
			if trade_time == cached_time:
				# update ticks
				ticks = self.data[ticker]['ticks']
				ticks['sum_vol'] += volume
				ticks['sum_buy'] += buy_vol
				ticks['sum_sell'] += sell_vol
				ticks['last_close'] = price
				self.data[ticker]['ticks'] = ticks
			# we not yet working with this ticker
			elif cached_time == None:
				# declare ticks
				self.data[ticker]['ticks']['time'] = trade_time
				self.data[ticker]['ticks']['sum_vol'] = volume
				self.data[ticker]['ticks']['sum_buy'] = buy_vol
				self.data[ticker]['ticks']['sum_sell'] = sell_vol
				self.data[ticker]['ticks']['last_close'] = price

			# we got new minute, ticks now should be aggregated to minutes
			# and also we check signals here
			else:
				# check signal
				await self._check_sig(ticker=ticker)
				# aggregate old ticks to minutes
				self._aggregate(ticker=ticker)
				# clear ticks
				# write new ticks info
				self.data[ticker]['ticks'] = {'time': trade_time,
											  'sum_vol': volume,
											  'sum_buy': buy_vol,
											  'sum_sell': sell_vol,
											  'last_close': price}
				# pop minutes
				if len(self.data[ticker]['minutes']) >= self.N:
					self.data[ticker]['minutes'].popitem(last=False)
		self.data[ticker]['last_utc'] = utc

	def _is_today_work(self):
		today = datetime.today()
		if today.weekday() == 5 or today.weekday() == 6:
			return False
		return True

	def _is_now_work(self):
		today = datetime.today()
		if today.weekday() == 5 or today.weekday() == 6:
			return False
		utc_time = int(self._utc_to_time(self._current_utc()))
		if utc_time > 2000 or utc_time < 700:
			return False
		return True

	async def get_trades(self, ticker, start, finish='1919174642'):
		bearer = self.jwt_token
		headers = {"Content-Type": "application/json",
				   "Authorization": f"Bearer {bearer}"}
		payload = {'from': start,
				   'to': finish}
		url = f'https://api.alor.ru/md/v2/Securities/MOEX/{ticker}/alltrades?{urllib.parse.urlencode(payload)}'
		async with aiohttp.ClientSession() as session:
			async with session.get(url, headers=headers) as resp:
				raw_respond: List[Dict[str, Table]] = await resp.json()
				return raw_respond

	async def run_forever(self):
		while True:
			if not self._is_now_work():
				print('now not work')
				await asyncio.sleep(60)
				continue
			# get mongo_watchlist
			await asyncio.sleep(1)
			async for ticker in AsyncIteratorWrapper(self.watchlist):
				# if ticker in mongo_watchlist or in base_tickers:
				print(ticker)
				last_utc = self.data[ticker]['last_utc']
				if not last_utc:
					last_utc = 0
				else:
					last_utc = int(last_utc / 1000)
				try:
					tds = await self.get_trades(ticker, start=last_utc)
				except Exception as e:
					print(f'ERROR get_today_trades {e}')
					continue
				await self._handle_trades(ticker, tds)
			await asyncio.sleep(0.5)
			self.jwt_token = self._get_jwt_token()
			while self.jwt_token == None:
				await asyncio.sleep(15)
				print('Trying to get token again....')
				self.jwt_token = self._get_jwt_token()

# Мой REFRESH_TOKEN для проверки работоспособности
# должен быть передан в e-mail
REFRESH_TOKEN = os.getenv('REFRESH_TOKEN', None)
alor = Trades(refresh_token=REFRESH_TOKEN,
			  watchlist=TICKERS,
			  N=100)
asyncio.run(alor.run_forever())
