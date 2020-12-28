from OptenFiberOpticDevices import ODTiT
import logging
import websockets
import asyncio
import json
import hyperion
import datetime
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import sys
import socket
from pathlib import Path
import statistics
import re
import win32api

# Настроечные переменные
hostname = socket.gethostname()
# address, port = socket.gethostbyname(hostname), 7681  # адрес websocket-сервера
index_of_reflection = 1.4682
speed_of_light = 299792458.0
program_version = '25122020'
output_measurements_order2 = ['T_degC', 'Fav_N', 'Fbend_N', 'Ice_mm']  # последовательность выдачи данных
DEFAULT_TIMEOUT = 10000
instrument_description_filename = 'instrument_description.json'

# параметры распознавания пиков
peak_distance_pm = 1000  # минимальное горизонтальное расстояние между соседними пиками, пм
peak_height_dbm = 3  # минимальная высота пика, dBm
peak_width_pm = [100, 600]  # ширина пика, пм

# тайминги
asyncio_pause_sec = 0.02  # длительность паузы в корутинах, чтобы другие могли работать
x55_measurement_interval_sec = 0.1  # интервал выдачи измерений x55
data_averaging_interval_sec = 1  # интервал усреднения данных
one_spectrum_interval_sec = 60  # интервал получения единичного спектра
send_pause_sec = 0.2  # пауза между отправками пакетов

# Глобальные переменные
master_connection = None
instrument_description = dict()
h1 = None
active_channels = set()
devices = list()

# хранение длин волн
wavelengths_buffer = dict()
wavelengths_buffer['is_ready'] = True
wavelengths_buffer['data'] = dict()
'''
wavelengths_buffer
<class 'dict'>: 
{
    'is_ready': True,
    'data': 
    {
        1555318018.4627628: 
        {
            1: [1527.9679096239095, 1531.204343536572, 1537.2687415797068, 1539.6400267255165, 1552.349066084183, 1556.2750827914833, 1560.390709839396, 1564.4504135379173, 1567.9244971411479, 1571.953882396913, 1576.0631581546152, 1583.524071034909], 
            2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: []
        },
        1555318018.5627592: 
        {
            1: [1527.9679096239095, 1531.2052944832005, 1537.2702789296047, 1539.637909647133, 1552.350045392259, 1556.2740985248877, 1560.3928863110805, 1564.4482257324607, 1567.9244971411479, 1571.953882396913, 1576.0635631225148, 1583.524071034909], 
            2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [], 14: [], 15: [], 16: []
        },
        ...
    }
}
'''

# хранение пересчитанных измерений (из длин волн)
measurements_buffer = dict()
measurements_buffer['is_ready'] = True
measurements_buffer['data'] = pd.DataFrame()
'''
measurements_buffer2
<class 'dict'>: 
{
    'is_ready': False,
    'data': <class 'pd.DataFrame'>
            Time        Device0_T_degC  Device0_Fav_N   Device0_Fbend_N   Device0_Ice_mm  Device1_T_degC  Device1_Fav_N   Device1_Fbend_N   Device1_Ice_mm
        0   153459.567  16.8            2654.56         34.67             1.3             16.7            2654.56         34.67             1.3
        ...
}
'''

# хранение усредненных измерений
averaged_measurements_buffer_for_OSM = dict()
averaged_measurements_buffer_for_OSM['is_ready'] = True
averaged_measurements_buffer_for_OSM['data'] = dict()

what_to_send = dict()

averaged_measurements_buffer_for_disk = dict()
averaged_measurements_buffer_for_disk['is_ready'] = True
averaged_measurements_buffer_for_disk['data'] = dict()

raw_measurements_buffer_for_disk = dict()
raw_measurements_buffer_for_disk['is_ready'] = True
raw_measurements_buffer_for_disk['data'] = dict()

wls_buffer_for_saving = dict()
wls_buffer_for_saving['is_ready'] = True
wls_buffer_for_saving['data'] = dict()

wls_buffer_for_disk = dict()
wls_buffer_for_disk['is_ready'] = True
wls_buffer_for_disk['data'] = dict()

# сердечный ритм основных корутин - количество выполненных циклов за период опроса
coroutine_heart_rate = dict()

loop = asyncio.get_event_loop()
loop.set_debug(False)
queue = asyncio.Queue(maxsize=0, loop=loop)
peak_stream = None


def valid_instrument_description(loc_instrument_description):
    '''

    :param loc_instrument_description: dict() - what should be checked
    :return: DetectionSettings - dict('ch_num': [Boxcar (pm), Diff (pm), Lockout (pm), NTV (pm), Threshold, Mode])
    Raises NameError exception
    '''
    ip_template = re.compile("^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$")
    important_fields = ['version', 'IP_address', 'SampleRate', 'devices', 'DetectionSettings']
    device_important_fields = ['version', 'Sensor3110_1', 'Sensor3110_2', 'Sensor4100', 'x55_channel', 'CTES',
                               'E', 'Asize', 'Bsize', 'Tmin', 'Tmax', 'Fmin', 'Fmax', 'Freserve', 'Bending_sensivity']

    if 1:
        if not isinstance(loc_instrument_description, dict):
            raise NameError(f'wrong format of instrument description: {loc_instrument_description}')

        # проверка наличия важных полей
        for field in important_fields:
            if field not in loc_instrument_description:
                raise NameError(f'no "{field}" field in instrument description')

        for device in loc_instrument_description['devices']:
            for field in device_important_fields:
                if field not in device:
                    raise NameError(f'no "{field}" field in device {device["Name"]}')

        # распарсить DetectionSettings
        ds = {}
        try:
            ch = 0
            for k in loc_instrument_description['DetectionSettings'].split(', '):
                if '!' in k:
                    ch = k.replace('!', '')
                    ds.setdefault(ch, [int(ch), 'name', 'description'])
                else:
                    ds[ch].append(k)
        except Exception as e:
            raise NameError(f'wrong DetectionSettings format, should be "3!, 250, 250, 0, 10000, 15000, 1, 4!, 250, 250, 0, 10000, 6000, 1"')

        # проверка IP по шаблону
        if not re.match(ip_template, loc_instrument_description['IP_address']):
            raise NameError(f'incorrect IP address')

        return ds

        # проверяем готовность прибора
        instrument_ip = loc_instrument_description['IP_address']
        with socket.socket() as s:
            s.settimeout(1)
            instrument_address = (instrument_ip, hyperion.COMMAND_PORT)
            try:
                s.connect(instrument_address)
            except socket.error:
                raise ConnectionError(f'command port is not active {instrument_ip}:{hyperion.COMMAND_PORT}')

        return ds


async def connection_handler(connection, path):
    global master_connection, instrument_description, averaged_measurements_buffer_for_OSM

    logging.info('New connection {} - path {}'.format(connection.remote_address[:2], path))

    while True:
        await asyncio.sleep(asyncio_pause_sec)

        # ожидание входящего сообщения
        try:
            msg = await connection.recv()
        except websockets.exceptions.WebSocketException as e:
            # текущее соединение закрыто
            logging.info(f'There is no connection while receiving data - websockets.exceptions.WebSocketException, {str(e.args)}')
            break

        logging.info(f"Received a message: {msg}")

        msg1 = msg.replace("\'", "\"")
        json_msg = dict()
        try:
            json_msg = json.loads(msg1)
        except json.JSONDecodeError as e:
            logging.critical(f'wrong JSON message has been refused, error {e}')
            json_msg.clear()
            break

        # проверка поступившего задания - если оно валидно, то данное соединение будет мастером
        logging.info("Checking the instrument description...")
        try:
            json_msg['DetectionSettings'] = valid_instrument_description(json_msg)
        except NameError as e:
            # wrong instrument description
            logging.error(e)
            break
        except ConnectionError as e:
            # no connection with ITO
            logging.error(e)
            break

        logging.info('it is ok - current connection become the master')
        master_connection = connection

        try:
            if json_msg['type'] == "control":
                logging.info(f'Поступило управляющее сообщение: {msg}')
                # ToDo: сделать обработку управляющих сообщений

                continue
        except KeyError:
            # нет такого ключа - это задание
            pass

        # если на диске уже есть задание, то сохраним его под другим именем
        if Path(instrument_description_filename).is_file():
            # file exists
            try:
                new_instrument_description_filename = os.path.splitext(instrument_description_filename)[0] + datetime.datetime.now().strftime('_before_%Y%m%d%H%M%S') + os.path.splitext(instrument_description_filename)[1]
                os.rename(instrument_description_filename, new_instrument_description_filename)
            finally:
                pass

        # сохранение нового задания на диск для последующей работы без соединения
        with open(instrument_description_filename, 'w+') as file:
            json.dump(json_msg, file, ensure_ascii=False, indent=4)

        # если поступившее задание отличается от имеющегося ранее, то нужно очистить накопленный буфер
        # if json.dumps(instrument_description) != json.dumps(json_msg) and len(averaged_measurements_buffer_for_OSM['data']) > 0:
        if len(averaged_measurements_buffer_for_OSM['data']) > 0:
            while not averaged_measurements_buffer_for_OSM['is_ready']:
                await asyncio.sleep(asyncio_pause_sec)
            averaged_measurements_buffer_for_OSM['is_ready'] = False
            try:
                averaged_measurements_buffer_file_name = datetime.datetime.now().strftime('avg_buffer_%Y%m%d%H%S.txt')
                logging.info(
                    'Received another instrument decsripton - saving averaged_measurements_buffer to ' + averaged_measurements_buffer_file_name)
                # сначала идет старое задание
                if 0:
                    with open(averaged_measurements_buffer_file_name, 'w+') as file:
                        json.dump(instrument_description, file, ensure_ascii=False, indent=4)
                        file.write('\n')
                        for _, measurements in averaged_measurements_buffer_for_OSM['data'].items():
                            file.write("\t".join([str(x) for x in measurements]) + '\n')

                averaged_measurements_buffer_for_OSM['data'].clear()
            finally:
                averaged_measurements_buffer_for_OSM['is_ready'] = True

        # актуализируем задание
        instrument_description = json_msg

        await instrument_init()


async def instrument_init():
    global output_measurements_order2, instrument_description, devices, active_channels, x55_measurement_interval_sec, h1, data_averaging_interval_sec, measurements_buffer, peak_stream

    data_averaging_interval_sec = 1.0 / instrument_description['SampleRate']

    if instrument_description['version'] == '0.1' or instrument_description['version'] == '0.2':
        output_measurements_order2 = ['T_degC', 'Fav_N', 'Fbend_N', 'Ice_mm']  # последовательность выдачи данных
    elif instrument_description['version'] == '0.3':
        output_measurements_order2 = ['T_degC', 'Fav_N', 'Fbend_N', 'F1_N', 'F2_N']  # последовательность выдачи данных

    # вытаскиваем информацию об устройствах
    devices = list()
    for device_description in instrument_description['devices']:

        # ToDo перенести это в класс ODTiT
        device = None
        try:
            if device_description['version'] == '0.1' or device_description['version'] == '0.2' or device_description['version'] == '0.3':
                device = ODTiT(device_description['x55_channel'])
                device.id = device_description['ID']
                device.name = device_description['Name']
                device.channel = device_description['x55_channel']
                device.ctes = device_description['CTES']
                device.e = device_description['E']
                device.size = (device_description['Asize'], device_description['Bsize'])
                device.t_min = device_description['Tmin']
                device.t_max = device_description['Tmax']
                device.f_min = device_description['Fmin']
                device.f_max = device_description['Fmax']
                device.f_reserve = device_description['Freserve']
                device.span_rope_diameter = device_description['SpanRopeDiametr']
                device.span_len = device_description['SpanRopeLen']
                device.span_rope_density = device_description['SpanRopeDensity']
                device.span_rope_EJ = device_description['SpanRopeEJ']
                device.bend_sens = device_description['Bending_sensivity']
                device.time_of_flight = int(
                    -2E9 * device_description['Distance'] * index_of_reflection / speed_of_light)

                device.sensors[0].id = device_description['Sensor4100']['ID']
                device.sensors[0].type = device_description['Sensor4100']['type']
                device.sensors[0].name = device_description['Sensor4100']['name']
                device.sensors[0].wl0 = device_description['Sensor4100']['WL0']
                device.sensors[0].t0 = device_description['Sensor4100']['T0']
                device.sensors[0].p_max = device_description['Sensor4100']['Pmax']
                device.sensors[0].p_min = device_description['Sensor4100']['Pmin']
                device.sensors[0].st = device_description['Sensor4100']['ST']

                device.sensors[1].id = device_description['Sensor3110_1']['ID']
                device.sensors[1].type = device_description['Sensor3110_1']['type']
                device.sensors[1].name = device_description['Sensor3110_1']['name']
                device.sensors[1].wl0 = device_description['Sensor3110_1']['WL0']
                device.sensors[1].t0 = device_description['Sensor3110_1']['T0']
                device.sensors[1].p_max = device_description['Sensor3110_1']['Pmax']
                device.sensors[1].p_min = device_description['Sensor3110_1']['Pmin']
                device.sensors[1].fg = device_description['Sensor3110_1']['FG']
                device.sensors[1].ctet = device_description['Sensor3110_1']['CTET']

                device.sensors[2].id = device_description['Sensor3110_2']['ID']
                device.sensors[2].type = device_description['Sensor3110_2']['type']
                device.sensors[2].name = device_description['Sensor3110_2']['name']
                device.sensors[2].wl0 = device_description['Sensor3110_2']['WL0']
                device.sensors[2].t0 = device_description['Sensor3110_2']['T0']
                device.sensors[2].p_max = device_description['Sensor3110_2']['Pmax']
                device.sensors[2].p_min = device_description['Sensor3110_2']['Pmin']
                device.sensors[2].fg = device_description['Sensor3110_2']['FG']
                device.sensors[2].ctet = device_description['Sensor3110_2']['CTET']

            if device_description['version'] == '0.2' or device_description['version'] == '0.3':
                device.fmodel_f0 = device_description['Fmodel_F0']
                device.fmodel_f1 = device_description['Fmodel_F1']
                device.fmodel_f2 = device_description['Fmodel_F2']
                device.icemodel_i1 = device_description['ICEmodel_I1']
                device.icemodel_i2 = device_description['ICEmodel_I2']

        except KeyError as e:
            return_error(f'JSON error - key {str(e)} did not find')

        devices.append(device)

    df_columns = list()
    df_columns.append('Time')
    for device_num, _ in enumerate(devices):
        for field in output_measurements_order2:
            df_columns.append('Device' + str(device_num) + '_' + field)

    measurements_buffer['data'] = pd.DataFrame(columns=df_columns)

    # находим все каналы, на которых есть решетки
    for device in devices:
        active_channels.add(int(device.channel))

    instrument_ip = instrument_description['IP_address']
    if not isinstance(instrument_ip, str):
        instrument_ip = instrument_ip[0]

    # проверяем готовность прибора
    with socket.socket() as s:
        s.settimeout(1)
        instrument_address = (instrument_ip, hyperion.COMMAND_PORT)
        try:
            s.connect(instrument_address)
        except socket.error:
            return_error('command port is not active on ip ' + instrument_ip)
            pass

    """
    # соединяемся с x55
    h1 = hyperion.Hyperion(instrument_ip)
    while not h1:
        try:
            h1 = hyperion.Hyperion(instrument_ip)
        except hyperion.HyperionError as e:
            return_error(e.__doc__)
            return None
    while not h1.is_ready:
        await asyncio.sleep(asyncio_pause_sec)
        pass
    logging.info('x55 is ready, sn', h1.serial_number)
    """
    h1 = hyperion.AsyncHyperion(instrument_ip, loop)

    ds = instrument_description['DetectionSettings']
    for channel in ds.keys():
        # пользовательские настройки
        my_ds = hyperion.HPeakDetectionSettings(channel, 'settings_name', 'settings_descr', ds[channel][0], 250, 1, 1000, 16001)
        # запись настроек в память прибора
        # ToDo добавить проверку наличия настроек с таким id - если их нет, то нужно Add, а не Update
        await h1._execute_command('#UpdateDetectionSetting', my_ds.pack())
        # применение настроек для текущего канала
        await h1.set_channel_detection_setting_id(channel, my_ds.setting_id)
        # проверим что записалось
        detection_settings_ids = await h1.get_channel_detection_setting_ids()
        ds = await h1.get_detection_setting(detection_settings_ids[channel-1])
        print(channel, ds.setting_id, ds.name, ds.description, ds.boxcar_length, ds.diff_filter_length, ds.lockout, ds.ntv_period, ds.threshold, ds.mode)

    logging.info(f'Instrument {await h1.get_instrument_name()} connected')
    # запускаем процедуру периодического получения спектра
    # await get_one_spectrum(h1)

    # await h1.set_active_full_spectrum_channel_numbers(active_channels)

    # запускаем стриминг пиков
    if not peak_stream:
        peak_stream = hyperion.HCommTCPPeaksStreamer(instrument_ip, loop, queue)
        await peak_stream.stream_data()


def return_error(e):
    """ функция принимает все ошибки программы, передает их на сервер"""
    logging.info("Error %s" % e)
    return None


async def get_wls_from_x55_coroutine():
    """ получение длин волн от x55 c исходной частотой (складирование в буффер в памяти) """
    global wavelengths_buffer, wls_buffer_for_saving, wls_buffer_for_disk

    last_timestamp = 0
    try:
        while True:

            try:
                this_function_name = sys._getframe().f_code.co_name
                if this_function_name in coroutine_heart_rate:
                    coroutine_heart_rate[this_function_name] += 1
                else:
                    coroutine_heart_rate[this_function_name] = 1

                peak_data = await queue.get()
                queue.task_done()
                if peak_data['data']:

                    cur_timestamp = round(peak_data['timestamp'])
                    if cur_timestamp != last_timestamp:
                        # print('wls -', cur_timestamp)
                        last_timestamp = cur_timestamp

                    peaks_by_channel = dict()
                    for channel in range(len(peak_data['data'].channel_slices)):
                        wls = []
                        for wl in peak_data['data'].channel_slices[channel]:
                            wls.append(wl)
                        peaks_by_channel[channel + 1] = wls

                    measurement_time = peak_data['timestamp']

                    # запись длин волн в буфер
                    if 0:
                        wls_buffer_for_saving['is_ready'] = False
                        try:
                            wls_buffer_for_saving['data'][measurement_time] = peaks_by_channel
                        finally:
                            wls_buffer_for_saving['is_ready'] = True

                    # запись длин волн в буфер 2
                    if wls_buffer_for_disk['is_ready']:
                        wls_buffer_for_disk['is_ready'] = False
                        t = [measurement_time]
                        try:
                            for key, value in peaks_by_channel.items():
                                t.extend(value)
                            wls_buffer_for_disk['data'][measurement_time] = t
                        finally:
                            wls_buffer_for_disk['is_ready'] = True

                    wavelengths_buffer['is_ready'] = False
                    try:
                        if measurement_time not in wavelengths_buffer.setdefault('data', dict()):
                            wavelengths_buffer['data'][measurement_time] = peaks_by_channel
                    except KeyError as e:
                        return_error(f'get_wls_from_x55_coroutine(): {e.__doc__}')
                    finally:
                        wavelengths_buffer['is_ready'] = True

                else:
                    # If the queue returns None, then the streamer has stopped.
                    break
            except Exception as e:
                logging.error(f'Some error during getting peaks - exception: {e.__doc__}')
                pass

        # если нет информации об инструменте, то не можем получать данные
        while not instrument_description:
            await asyncio.sleep(asyncio_pause_sec)
    finally:
        msg = 'get_wls_from_x55_coroutine is finishing'
        print(msg)
        logging.critical(msg)


async def wls_to_measurements_coroutine():
    """получение пересчет длин волн в измерения"""
    global wavelengths_buffer, measurements_buffer

    # приблизительная температура устройств, используется для поиска подходящего пика на спектре
    t_recommended = None

    try:
        while True:
            await asyncio.sleep(asyncio_pause_sec)

            this_function_name = sys._getframe().f_code.co_name
            if this_function_name in coroutine_heart_rate:
                coroutine_heart_rate[this_function_name] += 1
            else:
                coroutine_heart_rate[this_function_name] = 1

            # ждем появления данных в буфере
            while len(wavelengths_buffer['data']) < 2:
                await asyncio.sleep(asyncio_pause_sec)

            if not t_recommended:
                t_recommended = (devices[0].t_max + devices[0].t_min) / 2

            # ждем освобождения буфера
            while not wavelengths_buffer['is_ready']:
                await asyncio.sleep(asyncio_pause_sec)

            times_to_be_deleted = list()

            # блокируем буфер (чтобы надежно с ним работать в многопоточном доступе)
            wavelengths_buffer['is_ready'] = False
            try:
                for (measurement_time, peaks_by_channel) in wavelengths_buffer['data'].items():

                    # время усредненного блока, в которое попадает это измерение
                    averaged_block_time = measurement_time - measurement_time % (
                                1 / instrument_description['SampleRate'])
                    np.append([averaged_block_time], np.zeros(len(output_measurements_order2)))

                    devices_output3 = [measurement_time] + [np.nan] * len(output_measurements_order2) * len(devices)

                    raw_measurements_buffer_for_disk['data'][measurement_time] = [measurement_time]

                    # переводим пики в пикометры
                    for channel, wls in peaks_by_channel.items():
                        peaks_by_channel[channel] = [wl * 1000 for wl in wls]
                        pass

                    # шаг 1 - находим рекомендованную температуру
                    temperatures = list()
                    for device_num, device in enumerate(devices):
                        wls = device.find_yours_wls(peaks_by_channel[device.channel], delete_founded_peaks=False)
                        if wls:
                            temperatures.append(device.get_temperature(wls[0]))

                    # из списка температур выберем одну - которую будем рекомендовать далее
                    if len(temperatures) > 1:
                        t_recommended = statistics.median(temperatures)

                    # шаг 2 - находим пики с учетом рекомендованной температуры
                    for device_num, device in enumerate(devices):
                        # среди всех пиков ищем 3 подходящих для теукущего измерителя
                        wls = device.find_yours_wls(peaks_by_channel[device.channel], device.channel, t_recommended)

                        # если все три пика измерителя нашлись, то вычисляем тяжения и пр. Нет - вставляем пустышки
                        if wls:
                            device_output = device.get_tension_fav_ex(wls[1], wls[2], wls[0])

                            for field_num, filed in enumerate(output_measurements_order2):
                                devices_output3[1 + device_num * len(output_measurements_order2) + field_num] = \
                                device_output[filed]

                            raw_measurements_buffer_for_disk['data'][measurement_time].append(device_output['F1_N'])
                            raw_measurements_buffer_for_disk['data'][measurement_time].append(device_output['F2_N'])
                        else:
                            none_value = None
                            for field_num, filed in enumerate(output_measurements_order2):
                                devices_output3[
                                    1 + device_num * len(output_measurements_order2) + field_num] = none_value

                            raw_measurements_buffer_for_disk['data'][measurement_time].append(none_value)
                            raw_measurements_buffer_for_disk['data'][measurement_time].append(none_value)

                    if len(devices_output3) > 1:

                        while not measurements_buffer['is_ready']:
                            await asyncio.sleep(asyncio_pause_sec)
                        try:
                            measurements_buffer['is_ready'] = False

                            df = pd.DataFrame([tuple(devices_output3)], columns=measurements_buffer['data'].columns)
                            measurements_buffer['data'] = measurements_buffer['data'].append(df, ignore_index=True)

                            if measurement_time not in times_to_be_deleted:
                                times_to_be_deleted.append(measurement_time)

                        finally:
                            measurements_buffer['is_ready'] = True

                # измерения учтены, их можно удалять
                for time in sorted(times_to_be_deleted):
                    wavelengths_buffer['data'].pop(time)

            except Exception as e:
                pass

            finally:
                wavelengths_buffer['is_ready'] = True
    finally:
        msg = 'wls_to_measurements is finishing'
        print(msg)
        logging.debug(msg)


async def averaging_measurements_coroutine():
    """усреднение измерений"""
    global measurements_buffer, averaged_measurements_buffer_for_OSM

    cur_measurements = list()
    averaged_block_end_time = None
    try:
        while True:
            try:
                await asyncio.sleep(asyncio_pause_sec)

                this_function_name = sys._getframe().f_code.co_name
                if this_function_name in coroutine_heart_rate:
                    coroutine_heart_rate[this_function_name] += 1
                else:
                    coroutine_heart_rate[this_function_name] = 1

                # ждем освобождения буфера
                while not measurements_buffer['is_ready']:
                    await asyncio.sleep(asyncio_pause_sec)

                try:
                    # блокируем буфер (чтобы надежно с ним работать в многопоточном доступе)
                    measurements_buffer['is_ready'] = False

                    # ждем появления данных
                    if measurements_buffer['data'].size == 0:
                        await asyncio.sleep(asyncio_pause_sec)
                        continue

                    first_measurement_time = min(measurements_buffer['data'].Time)
                    last_measurement_time = max(measurements_buffer['data'].Time)

                    # время блока для первого и последнего измерения
                    first_measurement_block = (
                            first_measurement_time - first_measurement_time % data_averaging_interval_sec)
                    last_measurement_block = (
                            last_measurement_time - last_measurement_time % data_averaging_interval_sec)

                    # ждем появления достаточного количества данных
                    # если первое и последнее измерение находятся в разных блоках
                    if last_measurement_block <= first_measurement_block:
                        continue

                    # время начала усредненного блока, в которое попадает первое измерение
                    averaged_block_start_time = first_measurement_time - first_measurement_time % data_averaging_interval_sec
                    averaged_block_end_time = averaged_block_start_time + data_averaging_interval_sec

                    # выборка значений усредненного блока (по времени начала и конца блока)
                    block = measurements_buffer['data'].loc[
                        (measurements_buffer['data']['Time'] >= averaged_block_start_time) &
                        (measurements_buffer['data']['Time'] < averaged_block_end_time)]

                    # усреднение данных
                    cur_measurements = [averaged_block_end_time]

                    for device_num, device in enumerate(devices):

                        t_min = 0
                        t_max = 0
                        ice = 0

                        for field_num, field_name in enumerate(output_measurements_order2):
                            field_name = 'Device' + str(device_num) + '_' + field_name

                            if field_num == 0:
                                n = block[field_name].count()
                                if n == 0:
                                    pass
                                cur_measurements.append(n)

                            block_mean = 0
                            block_std = 0
                            block_max = 0
                            block_min = 0
                            try:
                                block_max = block[field_name].max()
                                block_min = block[field_name].min()
                                block_mean = block[field_name].mean()
                                block_std = block[field_name].std()
                            finally:
                                cur_measurements.append(block_mean)
                                cur_measurements.append(block_std)
                                if 'T_degC' in field_name:
                                    t_min = block_min
                                    t_max = block_max
                                if 'Ice_mm' in field_name:
                                    ice = block_mean

                        if instrument_description['version'] == "0.1" or instrument_description['version'] == "0.2":
                            # расчет границ нормального тяжения - при котором виртуальный гололед не более 1мм
                            # fok = f_extra(ice_threshold)
                            ice_threshold = 1
                            fok = 10 * (device.icemodel_i1 * ice_threshold + device.icemodel_i2 * (ice_threshold ** 2))
                            fmodel_max = 10 * (
                                    device.fmodel_f2 * (t_max ** 2) + device.fmodel_f1 * t_max + device.fmodel_f0)
                            fmodel_min = 10 * (
                                    device.fmodel_f2 * (t_min ** 2) + device.fmodel_f1 * t_min + device.fmodel_f0)
                            fok_min = fmodel_min - fok
                            fok_max = fmodel_max + fok

                            cur_measurements.append(fok_min)
                            cur_measurements.append(fok_max)

                    # обработанные данные убираем из блока
                    measurements_buffer['data'] = measurements_buffer['data'].loc[
                        (measurements_buffer['data']['Time'] >= averaged_block_end_time)]

                except KeyError:
                    pass
                finally:
                    measurements_buffer['is_ready'] = True

                print(cur_measurements)

                # запись выходных измерений в буфер для ОСМ и для записи на диск
                while not averaged_measurements_buffer_for_OSM['is_ready']:
                    await asyncio.sleep(asyncio_pause_sec)
                try:
                    averaged_measurements_buffer_for_OSM['is_ready'] = False
                    averaged_measurements_buffer_for_OSM['data'][averaged_block_end_time] = cur_measurements

                finally:
                    averaged_measurements_buffer_for_OSM['is_ready'] = True

                while not averaged_measurements_buffer_for_disk['is_ready']:
                    await asyncio.sleep(asyncio_pause_sec)
                try:
                    averaged_measurements_buffer_for_disk['is_ready'] = False
                    averaged_measurements_buffer_for_disk['data'][averaged_block_end_time] = cur_measurements
                finally:
                    averaged_measurements_buffer_for_disk['is_ready'] = True

            finally:
                pass
    finally:
        msg = 'function averaging_measurements is finished'
        print(msg)
        logging.critical(msg)
        loop.create_task(averaging_measurements_coroutine())


async def save_measurements_coroutine(buffer, file_type='avg'):
    """запись усредненных измерений на диск"""

    # строка с измерениями для сохранения на диск (и время этих измерений - чтобы не сохранять одно и тоже повторно)
    timestamp_msg, send_msg = None, ''

    if file_type == 'avg':
        file_prefix = ''
    elif file_type == 'raw':
        file_prefix = '_raw'
    elif file_type == 'wls':
        file_prefix = '_wls'
    else:
        raise Exception(ValueError, 'Value of file_type is unexpected')

    try:
        while True:
            await asyncio.sleep(asyncio_pause_sec)

            this_function_name = sys._getframe().f_code.co_name
            if this_function_name in coroutine_heart_rate:
                coroutine_heart_rate[this_function_name] += 1
            else:
                coroutine_heart_rate[this_function_name] = 1

            # ждем появления данных в буфере
            if len(buffer['data'].keys()) < 1:
                continue

            # ждем освобождения буфера
            if not buffer['is_ready']:
                continue

            try:
                # блокируем буфер (чтобы надежно с ним работать в многопоточном доступе)
                buffer['is_ready'] = False

                timestamp_msg = sorted(buffer['data'].keys(), reverse=False)[0]
                if file_type == 'wls':
                    send_msg = '\t'.join(['%.4f' % x for x in buffer['data'][timestamp_msg]])
                else:
                    send_msg = '\t'.join(['%.3f' % x for x in buffer['data'][timestamp_msg]])
            except Exception as e:
                logging.error(f'Some error during avg measurements sorting - exception: {e.__doc__}')
                logging.error(str(e))
                print(buffer['data'])
            finally:
                buffer['is_ready'] = True

            while send_msg != 'sent':
                await asyncio.sleep(asyncio_pause_sec)

                # send data block
                try:
                    data_arch_file_name = datetime.datetime.utcfromtimestamp(timestamp_msg).strftime(
                        f'%Y%m%d%H{file_prefix}.txt')

                    # add header if needed
                    if file_type == 'raw' and not Path(data_arch_file_name).is_file():
                        header = 'Timestamp, s\t'
                        for device in devices:
                            header += f'{device.name}_F1, N\t{device.name}_F2, N\t'
                        send_msg = header[:-1] + '\n' + send_msg

                    with open(data_arch_file_name, 'a') as f:
                        f.write(send_msg + '\n')
                except OSError:
                    logging.error('OS error during avg data saving')
                except Exception as e:
                    logging.error(
                        f'Some error during avg measurements saving - measurements: {send_msg}; exception: {e.__doc__}')
                else:
                    send_msg = 'sent'

            # записанные измерения можно удалять
            if not buffer['is_ready']:
                continue
            try:
                buffer['is_ready'] = False

                # удаление отправленного измерения
                if send_msg == 'sent' and timestamp_msg in buffer['data']:
                    buffer['data'].pop(timestamp_msg, None)

            finally:
                buffer['is_ready'] = True

    finally:
        send_msg = 'Function save_avg_measurements is finished'
        print(send_msg)
        logging.critical(send_msg)
        loop.create_task(save_measurements_coroutine(buffer, file_type))


async def send_avg_measurements_coroutine():
    """отправка усредненных измерений на сервер ОСМ"""
    global averaged_measurements_buffer_for_OSM, master_connection

    # строка с измерениями для отправки на OSM (и время этих измерений - чтобы не отправлять одно и тоже повторно)
    timestamp_msg, send_msg = None, ''

    # отправка измерений пакетами (True) или по одному (False)
    send_multi_packages = False
    last_send_time = 0
    # what_to_send = dict()
    try:
        while True:
            await asyncio.sleep(asyncio_pause_sec)

            this_function_name = sys._getframe().f_code.co_name
            if this_function_name in coroutine_heart_rate:
                coroutine_heart_rate[this_function_name] += 1
            else:
                coroutine_heart_rate[this_function_name] = 1

            # ждем соединения
            if not master_connection:
                continue

            # ждем появления данных в буфере
            if len(averaged_measurements_buffer_for_OSM['data'].keys()) < 1:
                continue

            # ждем освобождения буфера
            if not averaged_measurements_buffer_for_OSM['is_ready']:
                continue

            try:
                # блокируем буфер (чтобы надежно с ним работать в многопоточном доступе)
                averaged_measurements_buffer_for_OSM['is_ready'] = False

                if send_multi_packages:
                    for timestamp_msg in sorted(averaged_measurements_buffer_for_OSM['data'].keys(), reverse=True):
                        if timestamp_msg not in what_to_send and len(what_to_send) < 5:
                            what_to_send[timestamp_msg] = averaged_measurements_buffer_for_OSM['data'][timestamp_msg]

                    send_msg = '['
                    for timestamp_msg, measurements in what_to_send.items():
                        send_msg += '[' + ', '.join(
                            [str(x) for x in measurements]) + '], '
                    send_msg = send_msg[:-2] + ']'
                else:
                    timestamp_msg = sorted(averaged_measurements_buffer_for_OSM['data'].keys(), reverse=True)[0]
                    send_msg = '[' + ', '.join(
                        [str(x) for x in averaged_measurements_buffer_for_OSM['data'][timestamp_msg]]) + ']'

            finally:
                averaged_measurements_buffer_for_OSM['is_ready'] = True

            while send_msg != 'sent':
                await asyncio.sleep(asyncio_pause_sec)

                cur_time = datetime.datetime.now().timestamp()
                if master_connection and (cur_time - last_send_time) >= send_pause_sec:
                    # is client still alive? - прикрыто по причине нестыковки ping-pong в связке ОСМ-УПК
                    '''
                    try:
                        await master_connection.ping(data=str(int(datetime.datetime.now().timestamp())))
                    except websockets.exceptions.ConnectionClosed or ValueError:
                        continue
                    '''

                    # send data block
                    try:
                        await master_connection.send(send_msg)
                    except websockets.exceptions.ConnectionClosed:
                        logging.error('No connection while sending data - websockets.exceptions.ConnectionClosed. Zerroing master_connection.')
                        master_connection = None
                    except Exception as e:
                        logging.error(f'Some error during measurements sending to OSM - exception: {e.__doc__}')
                    else:
                        send_msg = 'sent'

                        # успешная отправка увеличивает счетчик
                        this_function_name = sys._getframe().f_code.co_name
                        counter_name = this_function_name + '_success_sent'
                        if counter_name in coroutine_heart_rate:
                            coroutine_heart_rate[counter_name] += 1
                        else:
                            coroutine_heart_rate[counter_name] = 1

                        last_send_time = datetime.datetime.now().timestamp()

            # отправленные измерения можно удалять
            if not averaged_measurements_buffer_for_OSM['is_ready']:
                continue
            try:
                averaged_measurements_buffer_for_OSM['is_ready'] = False

                if send_msg == 'sent':
                    if send_multi_packages:
                        # удаление отправленных измерений
                        for timestamp_msg in what_to_send:
                            if timestamp_msg in averaged_measurements_buffer_for_OSM['data']:
                                averaged_measurements_buffer_for_OSM['data'].pop(timestamp_msg, None)
                        what_to_send.clear()
                    else:
                        # удаление отправленного измерения
                        if timestamp_msg in averaged_measurements_buffer_for_OSM['data']:
                            averaged_measurements_buffer_for_OSM['data'].pop(timestamp_msg, None)

            finally:
                averaged_measurements_buffer_for_OSM['is_ready'] = True

    finally:
        send_msg = 'Function send_avg_measurements is finished'
        print(send_msg)
        logging.critical(send_msg)

        # restart current coroutine
        loop.create_task(send_avg_measurements_coroutine())


async def save_wls():
    """ Функция записывает накопленные сырые измерения в файл
    :return:
    """
    while True:
        return


async def get_one_spectrum(hyperion_x55_async):
    """
    Функция получает единичный спектр, используя класс hyperion.AsyncHyperion
    :param hyperion_x55_async: инициализированный класс hyperion.AsyncHyperion
    :return:
    """
    while True:
        await asyncio.sleep(asyncio_pause_sec)

        # калибровка спектра (из попугаев в dBm)
        await hyperion_x55_async.get_power_cal()

        # спектр и пики с прибора
        spectrum = await hyperion_x55_async.get_spectra()
        peaks = await hyperion_x55_async.get_peaks()
        raw_peaks = list()

        spectrum_step_pm = (spectrum.wavelengths[1] - spectrum.wavelengths[0]) * 1000

        # печать спектра
        if 0:
            for i in range(len(spectrum.wavelengths)):
                wl = spectrum.wavelengths[i]
                power = spectrum.data[channel][i]
                print(i, wl, power)

        for channel in spectrum.channel_map:
            peak_indexes, peaks_properties = find_peaks(spectrum.data[channel],
                                                        distance=peak_distance_pm / spectrum_step_pm,
                                                        prominence=peak_height_dbm,
                                                        width=peak_width_pm / spectrum_step_pm)

            # пики по сырому спектру
            for i in peak_indexes:
                wl = spectrum.wavelengths[i]
                power = spectrum.data[channel][i]
                raw_peaks.append(wl)

        print(datetime.datetime.utcnow(), spectrum.header.timestamp_int + spectrum.header.timestamp_frac * 1E-9,
              'spectrum', spectrum.header.serial_number, raw_peaks)

        await asyncio.sleep(one_spectrum_interval_sec)


'''
async def clock_sync():
    clock_sync_interval_sec = 3600
    while h1.get_is_ready():
        await asyncio.sleep(asyncio_pause_sec)
    last_sync_time = 0
    while True:
        await asyncio.sleep(asyncio_pause_sec)
        cur_time = datetime.datetime.now().timestamp()
        if h1.is_ready and cur_time - last_sync_time > clock_sync_interval_sec:
            try:
                await h1.set_instrument_utc_date_time(datetime.datetime.utcnow())
            except Exception as e:
                logging.debug(f'Some error during h1.set_instrument_utc_date_time - exception: {e.__doc__}')
            finally:
                last_sync_time = cur_time
'''


async def save_spectrum():
    spectrum_interval_sec = 600

    while not h1:
        await asyncio.sleep(asyncio_pause_sec)
    while not await h1.get_is_ready():
        await asyncio.sleep(asyncio_pause_sec)

    last_spectrum_time = 0
    try:
        while True:
            await asyncio.sleep(asyncio_pause_sec)
            cur_time = datetime.datetime.now().timestamp()

            if await h1.get_is_ready() and cur_time - last_spectrum_time > spectrum_interval_sec:
                try:
                    # калибровка спектра (из попугаев в dBm)
                    await h1.get_power_cal()

                    spectra_data = await h1.get_spectra()
                except Exception as e:
                    logging.error(f'Some error in h1.get_spectra() - exception: {e.__doc__}')

                try:
                    timestamp = spectra_data.header.timestamp_frac * 1e-9 + spectra_data.header.timestamp_int
                    str_to_save = ''

                    for channel in spectra_data.channel_map:
                        str_to_save = f'{str_to_save}{timestamp}\t{channel}\t' + '\t'.join(
                            ['%.2f' % pwr for pwr in spectra_data.data[channel]][::4]) + '\n'

                    data_arch_file_name = datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y%m%d%H_spectra.txt')
                    with open(data_arch_file_name, 'a') as f:
                        f.write(str_to_save)

                    logging.info(f'Got spectrum at {timestamp} {len(str_to_save)} bytes')
                    last_spectrum_time = cur_time
                except Exception as e:
                    logging.error(f'Some error in h1.save_spectrum() - exception: {e.__doc__}')
    finally:
        send_msg = 'Function save_spectrum is finished'
        print(send_msg)
        logging.critical(send_msg)

        # restart current coroutine
        loop.create_task(save_spectrum())


async def heart_rate():
    heart_rate_timeout_sec = 10
    delimiter = ' '
    # await asyncio.sleep(heart_rate_timeout_sec)

    try:
        out_str = f'heart_rate_order: connection {delimiter.join([str(x) for x in coroutine_heart_rate.keys()])}' + delimiter
        buffers_names = ['wavelengths_buffer', 'measurements_buffer', 'averaged_measurements_buffer_for_OSM',
                         'averaged_measurements_buffer_for_disk', 'wls_buffer_for_saving']
        out_str += delimiter.join(buffers_names) + delimiter
        print(out_str)
        logging.info(out_str)

        last_check_time = 0
        while True:
            await asyncio.sleep(asyncio_pause_sec)

            cur_time = datetime.datetime.now().timestamp()
            if cur_time - last_check_time > heart_rate_timeout_sec:

                last_check_time = cur_time
                out_str = 'heart_rate: '
                if master_connection:
                    out_str += '1 '
                else:
                    out_str += '0 '

                for key in coroutine_heart_rate:
                    out_str += str(coroutine_heart_rate[key]) + delimiter
                    coroutine_heart_rate[key] = 0

                out_str = out_str.rstrip() + delimiter + \
                          str(len(wavelengths_buffer['data'])) + delimiter + \
                          str(len(measurements_buffer['data'])) + delimiter + \
                          str(len(averaged_measurements_buffer_for_OSM['data'])) + delimiter + \
                          str(len(averaged_measurements_buffer_for_disk['data'])) + delimiter + \
                          str(len(wls_buffer_for_saving['data'])) + delimiter

                print(out_str)
                logging.info(out_str)
    finally:
        send_msg = 'Function heart_rate is finished'
        print(send_msg)
        logging.critical(send_msg)

        # restart current coroutine
        loop.create_task(heart_rate())

def getFileProperties(fname):
    """
    Read all properties of the given file return them as a dictionary.
    https://stackoverflow.com/questions/580924/how-to-access-a-files-properties-on-windows
    """
    propNames = ('Comments', 'InternalName', 'ProductName',
        'CompanyName', 'LegalCopyright', 'ProductVersion',
        'FileDescription', 'LegalTrademarks', 'PrivateBuild',
        'FileVersion', 'OriginalFilename', 'SpecialBuild')

    props = {'FixedFileInfo': None, 'StringFileInfo': None, 'FileVersion': None}

    try:
        # backslash as parm returns dictionary of numeric info corresponding to VS_FIXEDFILEINFO struc
        fixedInfo = win32api.GetFileVersionInfo(fname, '\\')
        props['FixedFileInfo'] = fixedInfo
        props['FileVersion'] = "%d.%d.%d.%d" % (fixedInfo['FileVersionMS'] / 65536,
                fixedInfo['FileVersionMS'] % 65536, fixedInfo['FileVersionLS'] / 65536,
                fixedInfo['FileVersionLS'] % 65536)

        # \VarFileInfo\Translation returns list of available (language, codepage)
        # pairs that can be used to retreive string info. We are using only the first pair.
        lang, codepage = win32api.GetFileVersionInfo(fname, '\\VarFileInfo\\Translation')[0]

        # any other must be of the form \StringfileInfo\%04X%04X\parm_name, middle
        # two are language/codepage pair returned from above

        strInfo = {}
        for propName in propNames:
            strInfoPath = u'\\StringFileInfo\\%04X%04X\\%s' % (lang, codepage, propName)
            ## print str_info
            strInfo[propName] = win32api.GetFileVersionInfo(fname, strInfoPath)

        props['StringFileInfo'] = strInfo
    except:
        pass

    return props

if __name__ == "__main__":
    log_file_name = datetime.datetime.now().strftime('UPK_server_%Y%m%d%H%M%S.log')
    logging.basicConfig(format=u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s',
                        level=logging.DEBUG, filename=log_file_name)

    logging.info(u'Program starts v.' + program_version)
    logging.info(getFileProperties(sys.argv[0]))

    address, port = None, None
    try:
        address = sys.argv[1]
        port = int(sys.argv[2])
    except Exception as e:
        logging.critical(
            f'Restart program with two arguments (address and port, space is delimiter) - exception: {e.__doc__}')
        print('Restart program with two arguments (address and port, space is delimiter)')
        exit(0)

    # связь с сервером, получение описания прибора
    loop.run_until_complete(websockets.serve(connection_handler, address, port, ping_interval=None, ping_timeout=30))
    logging.info('Server {} has been started'.format((address, port)))

    # метрики работы функций
    loop.create_task(heart_rate())

    # получение длин волн от x55 c исходной частотой (складирование в буффер в памяти)
    loop.create_task(get_wls_from_x55_coroutine())

    # получение пересчет длин волн в измерения
    loop.create_task(wls_to_measurements_coroutine())

    # усреднение измерений
    loop.create_task(averaging_measurements_coroutine())

    # отправка усредненных измерений на сервер
    loop.create_task(send_avg_measurements_coroutine())

    # запись усредненных измерений на диск
    loop.create_task(save_measurements_coroutine(averaged_measurements_buffer_for_disk, file_type='avg'))

    # запись неусредненных измерений F1, F2 на диск
    loop.create_task(save_measurements_coroutine(raw_measurements_buffer_for_disk, file_type='raw'))

    # запись длин волн на диск
    loop.create_task(save_measurements_coroutine(wls_buffer_for_disk, file_type='wls'))

    # x55 clock syncronization
    # loop.create_task(clock_sync())

    # saving spectrum
    # loop.create_task(save_spectrum())

    # если есть задание на диске, то загрузим его и начнем работать до получения нового задания
    if Path(instrument_description_filename).is_file():
        # file exists
        logging.info('Found instrument description file')
        try:
            with open(instrument_description_filename, 'r') as f:
                instrument_description = json.load(f)
        except Exception as e:
            logging.debug(f'Some error during instrument decsription file reading; exception: {e.__doc__}')
        else:
            logging.info('Loaded instrument description ' + json.dumps(instrument_description))
        loop.create_task(instrument_init())

    loop.run_forever()
