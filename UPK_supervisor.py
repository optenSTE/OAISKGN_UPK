"""
Скрипт проверяет появление данных в указанной папке
Если скорость поступления данных ниже порога, то выполняется перезапуск службы
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

# Константы
files_template = '*.txt'  # шаблон имени файла для подсчета размера папки
instrument_description_filename = 'instrument_description.json'  # имя файла с описанием оборудования
ITO_rebooting_duration_sec = 10

# Глобальные переменные
ITO_reboot_next_time = False


def get_dir_size_bytes():
    total_size = 0
    for file_name in glob.glob(files_template):
        file_size = os.path.getsize(file_name)
        total_size += file_size
    return total_size


def actions_when_slow_data():
    global ITO_reboot_next_time

    try:
        logging.info(f'Stopping service {service_name}...')
        # stop the service
        args = ['sc', 'stop', service_name]
        result1 = subprocess.run(args)
        logging.info(f'Stop service {service_name} return code {result1.returncode}')

        logging.info(f"Pause for {ITO_rebooting_duration_sec}sec")
        time.sleep(ITO_rebooting_duration_sec)
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

            if ITO_reboot_next_time:
                ITO_reboot_next_time = False

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
            else:
                ITO_reboot_next_time = True

    try:
        # start the service
        logging.info(f'Starting service {service_name}...')
        args = ['sc', 'start', service_name]
        result2 = subprocess.run(args)
        logging.info(f'Start service {service_name} return code {result2.returncode}')

        logging.info(f"Pause for {ITO_rebooting_duration_sec}sec")
        time.sleep(ITO_rebooting_duration_sec)
    except Exception as e:
        logging.error(f'An exception happened: {e.__doc__}')


if __name__ == "__main__":
    log_file_name = datetime.datetime.now().strftime('UPK_supervisor_%Y%m%d%H%M%S.log')
    logging.basicConfig(format=u'%(filename)s[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s',
                        level=logging.DEBUG, filename=log_file_name)

    logging.info(u'Program starts')

    program_params = None
    try:
        program_params = sys.argv
        if len(program_params) != 5:
            raise NameError('Wrong program parameters')
    except Exception as e:
        message = 'Wrong program parameters\n' \
                  'OAISKGN_UPK_supervisor <dir_size_speed_threshold_mb_per_h> <service_name> <check_interval_sec> <num_of_triggers_before_action>\n' \
                  'Restart as shown below\n' \
                  'OAISKGN_UPK_supervisor 1 OAISKGN_UPK 60 5'
        print(message)
        logging.error(message)
        exit(0)

    dir_size_speed_threshold_mb_per_h = float(program_params[1])  # минимальная скорость прироста размера папки, при которой не будет перезапускаться служба
    service_name = program_params[2]  # имя службы для перезапуска
    check_interval_sec = float(program_params[3])  # интервал проверки
    num_of_triggers_before_action = int(program_params[4])  # количество срабатываний триггера до реальных действий

    last_check_time = 0
    cur_time = 0
    cur_dir_size = 0
    last_dir_size = 0
    cur_num_of_triggers = 0

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
            logging.info(f'No file {instrument_description_filename}, pause for {check_interval_sec} sec..')
            time.sleep(check_interval_sec)

    print("cur_time\tlast_check_time\ttime_diff_sec\tdir_size_diff_byte\tcur_speed_mb_per_hour")
    while True:
        try:
            cur_time = datetime.datetime.now().timestamp()
            cur_dir_size = get_dir_size_bytes()
            if (last_check_time > 0) and (cur_time - last_check_time >= check_interval_sec):

                time_diff_sec = cur_time - last_check_time
                dir_size_diff_byte = cur_dir_size - last_dir_size

                if time_diff_sec <= 0:
                    raise NameError('time_diff_sec should be more than zero')

                cur_speed_mb_per_h = 3600 / (1024 * 1024) * dir_size_diff_byte / time_diff_sec

                print(cur_time, last_check_time, time_diff_sec, dir_size_diff_byte, cur_speed_mb_per_h, sep='\t')

                if 0.0 <= cur_speed_mb_per_h < dir_size_speed_threshold_mb_per_h:
                    print('Speed %.1fMb/h is too low' % cur_speed_mb_per_h)
                    logging.info('Speed %.1fMb/h is too low' % cur_speed_mb_per_h)

                    cur_num_of_triggers += 1
                    if cur_num_of_triggers >= num_of_triggers_before_action:
                        print('Action!')
                        logging.info('Action!')
                        cur_num_of_triggers = 0
                        actions_when_slow_data()
                else:
                    logging.info('Speed %.1fMb/h is ok' % cur_speed_mb_per_h)

            sleeping_time = check_interval_sec - (cur_time - last_check_time)
            if sleeping_time < 0 or sleeping_time > check_interval_sec:
                sleeping_time = check_interval_sec
            time.sleep(sleeping_time)

        except Exception as e:
            logging.error(f'An exception happened: {e.__doc__}')

        finally:
            last_check_time = cur_time
            last_dir_size = cur_dir_size
