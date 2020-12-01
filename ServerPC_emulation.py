# -*- coding: utf-8 -*-

# ��������� ��� �������� �������
# ������������ � ���������� ���, ���������� ��� JSON � ��������� ������������ (����������� �� �����)
# �������� ������ � ������� �� �� �����

import asyncio
import websockets
import json
# import instrument_description
import datetime


async def hello(uri):
    try:
        async with websockets.connect(uri, ping_timeout=None, ping_interval=None) as websocket:
            print('Server_PC -> new connection done')
            await websocket.send(str(data))
            while True:
                msg = await websocket.recv()
                print(datetime.datetime.utcnow(), 'UPK_PC ->', msg)
    finally:
        return



def run_server(loop=None):
    loop = asyncio.get_event_loop()
    while True:
        loop.run_until_complete(hello('ws://192.168.174.128:7681'))
        # loop.run_until_complete(hello('ws://192.168.1.216:7681'))
        print('Server_PC -> trying for a new connection to UPK_PC...')


if __name__ == "__main__":

    # print(instrument_description.si255_instrument)
    # �������� ������������ ����������� �� �����
    with open('V2_instrument_description.txt') as f:
        data = json.load(f)

    run_server()
