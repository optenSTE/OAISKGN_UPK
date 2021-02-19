# -*- coding: utf-8 -*-
# Библиотека для работы с Ethernet-реле NetPing 2/PWR-220 v12/ETH
# Основана на этой странице документации
# http://docs.netping.ru/[%d0%b4%d0%be%d0%ba%d1%83%d0%bc%d0%b5%d0%bd%d1%82%d0%b0%d1%86%d0%b8%d1%8f]-netping-2%252Fpwr-220-v12%252Feth-%2526-netping-2%252Fpwr-220-v13%252Fgsm3g/netping-2%252Fpwr-220-v12%252Feth-%2526-netping-2%252Fpwr-220-v13%252Fgsm3g%252C-%d0%be%d0%bf%d0%b8%d1%81%d0%b0%d0%bd%d0%b8%d0%b5-%d0%b2%d1%81%d1%82%d1%80%d0%be%d0%b5%d0%bd%d0%bd%d0%be%d0%b3%d0%be-%d0%bf%d0%be/14.-[dksf-53%252F203.1-iu]-%d0%bf%d0%be%d0%b4%d0%b4%d0%b5%d1%80%d0%b6%d0%ba%d0%b0-%d1%83%d1%81%d1%82%d1%80%d0%be%d0%b9%d1%81%d1%82%d0%b2%d0%be%d0%bc-http-api/14.2.-[dksf-53%252F203.1-iu]-%d1%83%d0%bf%d1%80%d0%b0%d0%b2%d0%bb%d0%b5%d0%bd%d0%b8%d0%b5-%d1%80%d0%b5%d0%bb%d0%b5/

# v19022021

import urllib.request


class NetpingRelay:
    def __init__(self, relay_ip: str, username='visor', password='ping'):
        if not isinstance(relay_ip, str):
            raise TypeError('relay_ip should be string, like "10.0.0.56"')

        if not isinstance(username, str):
            raise TypeError('username should be string, like "visor"')

        if not isinstance(password, str):
            raise TypeError('password should be string, like "ping"')

        self.ip = relay_ip
        self.username = username
        self.password = password
        self.timeout = 1

        passman = urllib.request.HTTPPasswordMgrWithDefaultRealm()
        passman.add_password(None, relay_ip, self.username, self.password)
        auth_handler = urllib.request.HTTPBasicAuthHandler(passman)
        opener = urllib.request.build_opener(auth_handler)
        urllib.request.install_opener(opener)

    def _get_relay_status(self, socket_num):
        """"
        socket_num - номер розетки (1 или 2)
        return mode, state
        mode - текущий режим работы розетки
            0 – Выключено вручную
            1 – Включено вручную
            2 – Сторож
            3 – Расписание
            4 – Расп+Cторож
            5 – Логика
            6 – Расп+Логика
        state -  текущее состояние розетки
            0 - отключена
            1 - включена
        """
        if 1 > socket_num > 2:
            raise ValueError('socket_num should be 1 or 2')

        url = f'http://{self.ip}/relay.cgi?r{socket_num}'
        resp = urllib.request.urlopen(url, timeout=self.timeout)
        data = resp.read().decode("utf-8")

        if 'error' in data:
            raise RuntimeError('Wrong request or electrical secket number')

        # relay_result('ok', 2, 1);
        _, mode, state = data[data.find("(") + 1:data.find(")")].split(', ')

        return mode, state

    def _set_relay_status(self, socket_num, state):
        if 1 > socket_num > 2:
            raise ValueError('socket_num should be 1 or 2')

        url = f'http://{self.ip}/relay.cgi?r{socket_num}={state}'
        resp = urllib.request.urlopen(url, timeout=self.timeout)
        data = resp.read().decode("utf-8")
        if 'error' in data:
            return False
        return True

    def get_state(self, socket_num):
        k = self._get_relay_status(socket_num)
        return k[1]

    def socket_off(self, socket_num):
        self._set_relay_status(socket_num, 0)

    def socket_on(self, socket_num):
        self._set_relay_status(socket_num, 1)
