"""
UPK_supervisor
Скрипт проверяет появление данных в указанной папке
Если скорость поступления данных ниже порога, то выполняется перезапуск службы

Также есть второй триггер, который перезапускает службу (без перезагрузки ИТО) с заданным интевалом

ini-файл
    ;*********************************
    [main]
    ;*********************************

    ; шаблон имени файла для подсчета размера папки
    files_template = *.txt

    ; имя файла с описанием оборудования
    instrument_description_filename = instrument_description.json

    ; продолжительность перезагрузки прибора
    ITO_rebooting_duration_sec = 40

    ; пауза при перезапуске службы
    win_service_restart_pause = 10



    ;*********************************
    [trigger1]
    ;*********************************

    ; минимальная скорость прироста размера папки, при которой не будет перезапускаться служба
    dir_size_speed_threshold_mb_per_h = 8

    ; имя службы для перезапуска
    service_name = OAISKGN_UPK

    ; интервал проверки
    dir_check_interval_sec = 60

    ; количество срабатываний триггера до реальных действий
    num_of_triggers_before_action = 5



    ;*********************************
    [trigger2]
    ;*********************************

    ;интервал безусловной перезагрузки службы
    win_service_restart_interval_sec = 3600

"""

import datetime
import glob
import os
import time
import logging
import subprocess
import hyperion
import sys
import json
from pathlib import Path
import socket
import random
import win32api
import configparser

program_version = '24122020'

# Глобальные переменные
files_template = '*.txt'  # шаблон имени файла для подсчета размера папки
instrument_description_filename = 'instrument_description.json'  # имя файла с описанием оборудования
ITO_rebooting_duration_sec = 40  # время перезагрузки прибора
win_service_restart_pause = 10  # пауза при перезапуске службы


def get_dir_size_bytes():
    total_size = 0
    for file_name in glob.glob(files_template):
        file_size = os.path.getsize(file_name)
        total_size += file_size
    return total_size


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


def action_when_trigger_released(ITO_reboot=False):

    try:
        logging.info(f'Stopping service {service_name}...')
        # stop the service
        args = ['sc', 'stop', service_name]
        result1 = subprocess.run(args)
        logging.info(f'Stop service {service_name} return code {result1.returncode}')

        logging.info(f"Pause for {win_service_restart_pause}sec")
        time.sleep(win_service_restart_pause)
    except Exception as e:
        logging.error(f'An exception happened: {e.__doc__}')

    # проверка готовности прибора (должен отвечать порт, по которому идут команды)
    with socket.socket() as s:
        s.settimeout(1)
        instrument_address = (instrument_ip, hyperion.COMMAND_PORT)
        try:
            s.connect(instrument_address)  # подключаемся к порту команд
        except socket.error:
            logging.error(f'command port is not active {instrument_ip}:{hyperion.COMMAND_PORT}')
        else:
            logging.info(f"Hyperion command port test passed {instrument_ip}:{hyperion.COMMAND_PORT}")

            if ITO_reboot:

                try:
                    h1 = hyperion.Hyperion(instrument_ip)
                except Exception as e:
                    logging.debug(f'Some error during ITO init - exception: {e.__doc__}')
                else:

                    try:
                        logging.info(f'Current ITO time {h1.instrument_utc_date_time.strftime("%d.%m.%Y %H:%M:%S")}')

                        utcnow = datetime.datetime.utcnow()
                        logging.info(f'Setting ITO time to UPK-UTC {utcnow.strftime("%d.%m.%Y %H:%M:%S")}')
                        h1.instrument_utc_date_time = utcnow

                        logging.info(f'Current ITO time {h1.instrument_utc_date_time.strftime("%d.%m.%Y %H:%M:%S")}')

                    except Exception as e:
                        logging.debug(f'Some error during h1.instrument_utc_date_time - exception: {e.__doc__}')

                    try:
                        logging.info(f'Rebooting ITO...')
                        h1.reboot()
                    except Exception as e:
                        logging.error(f'An exception happened: {e.__doc__}')

                    logging.info(f"Pause for {ITO_rebooting_duration_sec}sec")
                    time.sleep(ITO_rebooting_duration_sec)
                    logging.info(f"Instrument {h1.instrument_name}, ip {instrument_ip} rebooted")

    try:
        # start the service
        logging.info(f'Starting service {service_name}...')
        args = ['sc', 'start', service_name]
        result2 = subprocess.run(args)
        logging.info(f'Start service {service_name} return code {result2.returncode}')

    except Exception as e:
        logging.error(f'An exception happened: {e.__doc__}')


if __name__ == "__main__":
    log_file_name = datetime.datetime.now().strftime('UPK_supervisor_%Y%m%d%H%M%S.log')
    logging.basicConfig(format=u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s',
                        level=logging.DEBUG, filename=log_file_name)

    logging.info(u'Program starts v.' + program_version)
    logging.info(f'EXE-file {sys.argv[0]}')
    logging.info(getFileProperties(sys.argv[0]))

    # default ini-values
    dir_size_speed_threshold_mb_per_h = 8  # минимальная скорость прироста размера папки, при которой не будет перезапускаться служба
    service_name = "OAISKGN_UPK"  # имя службы для перезапуска
    dir_check_interval_sec = 60  # интервал проверки
    num_of_triggers_before_action = 5  # количество срабатываний триггера до реальных действий
    win_service_restart_interval_sec = 3600  # интервал безусловной перезагрузки службы

    try:
        filename, file_extension = os.path.splitext(sys.argv[0])
        ini_file_name = f"{filename}.ini"
        if not Path(ini_file_name).is_file():
            raise FileExistsError(f'no file {ini_file_name}')

        config = configparser.ConfigParser()

        config.read(ini_file_name)

        files_template = config['main']['files_template']
        instrument_description_filename = config['main']['instrument_description_filename']
        ITO_rebooting_duration_sec = float(config['main']['ITO_rebooting_duration_sec'])
        win_service_restart_pause = float(config['main']['win_service_restart_pause'])

        dir_size_speed_threshold_mb_per_h = float(config['trigger1']['dir_size_speed_threshold_mb_per_h'])
        service_name = config['trigger1']['service_name']
        dir_check_interval_sec = float(config['trigger1']['dir_check_interval_sec'])
        num_of_triggers_before_action = float(config['trigger1']['num_of_triggers_before_action'])

        win_service_restart_interval_sec = float(config['trigger2']['win_service_restart_interval_sec'])

    except Exception as e:
        logging.info(f'Error during ini-file reading: {str(e)}')
        sys.exit(0)

    # оставим в логе входные параметры
    logging.info(f'dir_size_speed_threshold_mb_per_h={dir_size_speed_threshold_mb_per_h}, '
                 f'service_name={service_name}, dir_check_interval_sec={dir_check_interval_sec}, '
                 f'num_of_triggers_before_action={num_of_triggers_before_action}, '
                 f'win_service_restart_interval_sec={win_service_restart_interval_sec}')

    # случайная добавка к интервалу перезагрузки - чтобы не было в одно и то же время
    win_service_restart_interval_sec += random.randint(-int(win_service_restart_interval_sec / 8),
                                                       int(win_service_restart_interval_sec / 8))

    last_dir_check_time = datetime.datetime.now().timestamp()
    last_unconditional_reboot_time = datetime.datetime.now().timestamp()
    cur_time = 0
    cur_dir_size = 0
    last_dir_size = 0
    cur_num_of_triggers = 0

    ITO_reboot_next_time = False

    logging.info('Looking for instrument description file...')
    instrument_ip = None
    while not instrument_ip:
        # если есть задание на диске, то загрузим его и начнем работать до получения нового задания
        if Path(instrument_description_filename).is_file():
            try:
                with open(instrument_description_filename, 'r') as f:
                    instrument_description = json.load(f)
            except Exception as e:
                logging.debug(f'Some error during instrument description file reading; exception: {e.__doc__}')
            else:
                logging.info('Loaded instrument description ' + json.dumps(instrument_description))

            instrument_ip = instrument_description['IP_address']
        else:
            logging.info(f'No file {instrument_description_filename}, pause for {dir_check_interval_sec} sec..')
            time.sleep(dir_check_interval_sec)

    # print("cur_time\tlast_check_time\ttime_diff_sec\tdir_size_diff_byte\tcur_speed_mb_per_hour")
    while True:

        # безуслованя перезагрузка по таймеру
        if 1:
            cur_time = datetime.datetime.now().timestamp()
            if (cur_time - last_unconditional_reboot_time) >= win_service_restart_interval_sec > 0:
                logging.info('Time-trigger released')

                last_unconditional_reboot_time = cur_time

                # случайная добавка к интервалу перезагрузки - чтобы не было в одно и то же время
                win_service_restart_interval_sec += random.randint(-int(win_service_restart_interval_sec / 8),
                                                                   int(win_service_restart_interval_sec / 8))

                action_when_trigger_released()

        # перезагрузка по скорости наполнения папки с данными
        try:
            cur_time = datetime.datetime.now().timestamp()
            cur_dir_size = get_dir_size_bytes()
            if (cur_time - last_dir_check_time) >= dir_check_interval_sec:

                time_diff_sec = cur_time - last_dir_check_time
                dir_size_diff_byte = cur_dir_size - last_dir_size

                if time_diff_sec <= 0:
                    raise NameError('time_diff_sec should be more than zero')

                cur_speed_mb_per_h = 3600 / (1024 * 1024) * dir_size_diff_byte / time_diff_sec

                # print(cur_time, last_dir_check_time, time_diff_sec, dir_size_diff_byte, cur_speed_mb_per_h, sep='\t')

                if 0.0 <= cur_speed_mb_per_h < dir_size_speed_threshold_mb_per_h:
                    # print('Speed %.1fMb/h is too slow' % cur_speed_mb_per_h)
                    logging.info('Speed %.1fMb/h is too low' % cur_speed_mb_per_h)

                    cur_num_of_triggers += 1
                    if cur_num_of_triggers >= num_of_triggers_before_action:
                        # print('Action!')
                        logging.info('Action!')
                        cur_num_of_triggers = 0

                        action_when_trigger_released(ITO_reboot_next_time)
                        ITO_reboot_next_time = not ITO_reboot_next_time
                else:
                    logging.info('Speed %.1fMb/h is ok' % cur_speed_mb_per_h)

            sleeping_time = dir_check_interval_sec - (cur_time - last_dir_check_time)
            if sleeping_time < 0 or sleeping_time > dir_check_interval_sec:
                sleeping_time = dir_check_interval_sec
            time.sleep(sleeping_time)

        except Exception as e:
            logging.error(f'An exception happened: {e.__doc__}')

        finally:
            last_dir_check_time = cur_time
            last_dir_size = cur_dir_size
