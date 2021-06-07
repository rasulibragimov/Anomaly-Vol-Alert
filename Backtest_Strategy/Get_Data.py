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
import asyncio
import aiohttp
import calendar


class Trades:
	def __init__(self, refresh_token):
		self.refresh_token = refresh_token
		self.jwt_token = self._get_jwt_token()
		while self.jwt_token == None:
			time.sleep(15)
			print('Trying to get token again....')
			self.jwt_token = self._get_jwt_token()

	def _current_utc(self):
		return str(int(time.time()))

	def _date_to_utc(self, dt):
		dt = datetime.strptime(dt, '%Y-%m-%d')
		return calendar.timegm(dt.utctimetuple())

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

	async def get_tick(self, ticker, start, finish, tf):
		# tf: Длительность таймфрейма в секундах или код
		# ("D" - дни, "W" - недели, "M" - месяцы, "Y" - годы)
		bearer = self.jwt_token
		headers = {"Content-Type": "application/json",
				   "Authorization": f"Bearer {bearer}"}
		start, finish = self._date_to_utc(start), self._date_to_utc(finish)
		url = f'https://api.alor.ru/md/v2/history'
		params = {'symbol': ticker, 'exchange': 'MOEX', 'tf': tf, 'from': start, 'to': finish}
		async with aiohttp.ClientSession() as session:
			async with session.get(url, headers=headers, params=params) as resp:
				raw_respond = await resp.json()
				df = pd.DataFrame(raw_respond.get('history', {}))
				df.set_index(df['time'].apply(lambda t: datetime.fromtimestamp(t)), inplace=True)
				df.drop('time', inplace=True, axis=1)
				df.columns = ['Close', 'Open', 'High', 'Low', 'Volume']
				return df


# Мой REFRESH_TOKEN для проверки работоспособности
# должен быть передан в e-mail
# REFRESH_TOKEN = os.getenv('REFRESH_TOKEN', None)
# alor = Trades(refresh_token=REFRESH_TOKEN)
# df = asyncio.run(alor.get_tick('SBER', start='2020-05-01', finish='2020-05-31', tf=60))
# print(df)
