# -*- coding: utf-8 -*-
# v12012021
import math
import copy


# MicroOptics FBG-sensor class (either strain and temperature)
class FBG:

    def __init__(self):
        self.id = 0
        self.type = 'os3110'
        self.name = 'NONAME'
        self.t0 = 0.0
        self.wl0 = 0.0
        self.p_min = 0.0
        self.p_max = 65535
        self.wl_min = 1420000
        self.wl_max = 1660000
        self.fg = 0.89  # 0.89 только os3110
        self.ctet = 0.0  # КТЛР подложки os3110
        self.st = 1.87754658264289E-5  # коэффициент для расчета температуры os4100

    def __str__(self):
        print_str = 'Sensor %s Name=%s ID=%s WL0=%.4f T0=%.2f' % (self.type, self.name, self.id, self.wl0, self.t0)
        return print_str

    def get_temperature(self, wl):
        return self.t0 + (wl - self.wl0) / (self.wl0 * self.st)

    def is_power_ok(self, power):
        if self.p_min <= power <= self.p_max:
            return True
        return False


class ODTiT:
    def __init__(self, channel=0):
        self.channel = channel
        self.id = 0  # идентификатор устройства
        self.name = 'NONAME'  # наименование устройства, используется для вывода
        self.channel = 1  # номер порта MicronOptics si255, к которому подключено устройство
        self.sample_rate = 0  # ожидаемое значение частоты поступления данных, Гц

        self.e = 0.0  # модуль Юнга устройства
        self.ctes = 0.0  # КТЛР тела ОДТиТ
        self.size = (0.0, 0.0)  # размер рабочей области (толщина, высота), мм
        self.bend_sens = 0.01  # чувствительность к поперечным нагрузкам, ustr/daN

        self.span_len = 0.0  # длина пролета, м
        self.span_rope_diameter = 0.0  # диаметр троса, м
        self.span_rope_density = 0.0  # погонная масса провода, кг/м
        self.span_rope_EJ = 0.0  # модуль упругости * момент инерции провода

        self.f_min = 0.0  # окно возможных тяжений провода, даН
        self.f_max = 400.0
        self.f_reserve = 1000.0  # запас по тяжению (расширяет диапазон f_min...f_max в обе стороны), даН

        # полином вычисления ожидаемого тяжения Fожид(T) = f2*T^2 + f1*T + f0, где Т-температура
        self.fmodel_f0 = 0
        self.fmodel_f1 = 0
        self.fmodel_f2 = 0

        # полином вычисления стенки гололеда по превышению текущего тяжения над ожидаемым Fextra(Ice) = i2*Ice^2+i1*Ice
        self.icemodel_i1 = 0
        self.icemodel_i2 = 0

        # используются только для проверки принадлежности измерений температурной решетке
        self.t_min = -60.0  # минимальная эксплутационная температура, С
        self.t_max = 60.0  # максимальная эксплутационная температура, С

        self.time_of_flight = 0  # Задержка распространения света в волокне от Прибора до первой решетки Измерителя и обратно
        self.distance = 0  # оптическое расстояние до устройства, м
        '''
        sol_poly_k0, _k1, _k2 - коэффициенты полинома компенсации длины волны за задержку распространения света
        wl_shift[nm] = wl_compensated[nm] - wl[nm] =  dist[m] * ( k2*wl[nm]^2 + k1*wl[nm] + k0)
            dist - расстояние, м
            wl - измеренная длина волны, пм
            wl_comp - длина волны с учетом компенсации
        УПК Газовая - sol_poly=(-8.19436E-04, 1.06632E-06, -3.29350E-10)
        '''
        self.sol_poly_k0 = 0
        self.sol_poly_k1 = 0
        self.sol_poly_k2 = 0

        self.sensors = []  # три решетки - температурная, натяжная левая, натяжная правая
        for i in range(3):
            self.sensors.append(FBG())

    def __str__(self):
        print_str = 'ODTiT device: %s\t%s\t%s\t%s' % (self.name, self.sensors[1].__str__(), self.sensors[2].__str__(), self.sensors[0].__str__())
        return print_str

    def find_yours_wls(self, wls_pm, channel=None, t_recommended=None, delete_founded_peaks=True, apply_sol = False):
        """Function checks is this wavelength belongs of
            this ODTiT device (any of optical sensor)
        :param wls_pm: list(), список длин волн пиков, пм - после выполнения функции из него будут удалены найденные пики
        :param channel: int(), номер канала, в настоящее время не используется, сохранени для совместимости
        :param t_recommended: int(), ориентировочная температура - в случае обнаружения нескольких пиков у данныго ОДТиТ будет использован ближайший к этой температуре
                                    по умолчанию None - в случае нескольких пиков будет ввозвращено False
        :return: list()or Bool, wavelengths belongs of this ODTiT device or False
        """

        ret_value = [None, None, None]

        wls_local = copy.deepcopy(wls_pm)
        if apply_sol:
            wls_local = self.__apply_sol(wls_local)

        cur_t = None
        for sensor_num, sensor in enumerate(self.sensors):
            # для каждой решетки описываются параметры выхова функции self._get_wl_from_value для вычисления wl_min, wl_max, wl_recommended
            if sensor_num == 0:
                if t_recommended:
                    param = [[sensor_num, self.t_min], [sensor_num, self.t_max], [sensor_num, t_recommended]]
                else:
                    param = [[sensor_num, self.t_min], [sensor_num, self.t_max], [sensor_num, (self.t_min + self.t_max)/2]]
            else:
                param = [[sensor_num, cur_t, self.f_min - self.f_reserve], [sensor_num, cur_t, self.f_max + self.f_reserve], [sensor_num, cur_t, (self.f_min + self.f_max)/2]]  # для натяжной нет рекомендованной длины волны

            wls = (self._get_wl_from_value(*param[0]), self._get_wl_from_value(*param[1]))
            wl_min = min(wls)
            wl_max = max(wls)
            wl_recommended = self._get_wl_from_value(*param[2])

            candidates = []
            for wl in wls_local:
                if wl_min < wl < wl_max:
                    candidates.append(wl)

            # ничего не найдено то продолжать нельзя
            if len(candidates) == 0:
                break

            # найдено несколько пиков - выбираем ближайший по recommended_t
            min_index = 0
            if len(candidates) > 1:

                # если нет рекомендованной температуры, то невозможно выбрать правильный вариант
                if not t_recommended:
                    break

                diffs = [abs(wl-wl_recommended) for wl in candidates]
                min_index = diffs.index(min(diffs))

            # остался только один вариант - удаляем его из массива пиков
            cur_sensor_wl = candidates[min_index]
            ret_value[sensor_num] = cur_sensor_wl
            wls_local.remove(cur_sensor_wl)  # удаляем пик из локальной базы, чтобы не мешался далее

            if sensor_num == 0:
                cur_t = self.get_temperature(cur_sensor_wl)

        if None in ret_value:
            return False

        # найдены все 3 пика датчика, можно их удалять из исходного списка
        if delete_founded_peaks:
            for wl in ret_value:
                wls_pm.remove(wl)

        return ret_value

    def is_wl_of_temperature_sensor(self, wl, channel=0, apply_sol=False):
        """Function checks is this wavelength belongs of
            this ODTiT device (temperature optical sensor)
        :param wl: wavelength, pm
        :param channel: channel num for MOI si255
        :return: is the wavelength belongs of this ODTiT device
        """

        # apply SoL compensation
        if apply_sol:
            wl = self.__apply_sol(wl)

        wl_max = self.sensors[0].wl0 * (1 + (self.t_max - self.sensors[0].t0) * self.sensors[0].st)
        wl_min = self.sensors[0].wl0 * (1 + (self.t_min - self.sensors[0].t0) * self.sensors[0].st)

        ret_value = False
        if min(wl_min, wl_max) <= wl <= max(wl_min, wl_max):
            ret_value = True

        if 0 < self.channel != channel:
            ret_value = False

        return ret_value

    def is_wl_of_strain_sensor(self, wl, t, sensor_num, channel=0, apply_sol=False):
        """For strain sensors os3110 checks is given WL belongs of this ODTiT device
        :param wl: measured strain sensor's wavelength, pm
        :param t: ODTiT device temperature, degC (by os4100 sensor)
        :param sensor_num: 1 or 2 - first or second sensor into ODTiT device
        :param channel: channel num for strain sensor, only for MIO instruments
        :return: is the wavelength belongs of this ODTiT device
        """

        # apply SoL compensation
        if apply_sol:
            wl = self.__apply_sol(wl)

        if 1 < sensor_num > 2:
            raise IndentationError('Sensor_num should be 1 or 2')

        # WL = WL_0 * (1 + ((f1 * 10 / (E * S) - (T - Ts1_0) * (CTET - CTES) / 1000000) * FG + (T - Tt_0) * ST));

        wl_min = self.sensors[sensor_num].wl0 * (1 + (((self.f_min - self.f_reserve) * 10 / (self.e * self.size[0] * self.size[1] * 1E-6) - (t - self.sensors[sensor_num].t0) * (
                    self.sensors[sensor_num].ctet - self.ctes) / 1E+6) * self.sensors[sensor_num].fg + (t - self.sensors[0].t0) * self.sensors[0].st))
        wl_max = self.sensors[sensor_num].wl0 * (1 + (((self.f_max + self.f_reserve) * 10 / (self.e * self.size[0] * self.size[1] * 1E-6) - (t - self.sensors[sensor_num].t0) * (
                    self.sensors[sensor_num].ctet - self.ctes) / 1E+6) * self.sensors[sensor_num].fg + (t - self.sensors[0].t0) * self.sensors[0].st))

        ret_value = False
        if min(wl_min, wl_max) <= wl <= max(wl_min, wl_max):
            ret_value = True

        if 0 < self.channel != channel:
            ret_value = False

        return ret_value

    def get_temperature(self, wl_temperature_sensor, apply_sol=False):

        # apply SoL compensation
        if apply_sol:
            wl_temperature_sensor = self.__apply_sol(wl_temperature_sensor)

        return self.sensors[0].get_temperature(wl_temperature_sensor)

    def get_tension_fav(self, wl_tension_sensor_1, wl_tension_sensor_2, wl_temperature_sensor, apply_sol=False):
        return self.get_tension_fav_ex(wl_tension_sensor_1, wl_tension_sensor_2, wl_temperature_sensor, apply_sol)[0]

    def _get_wl_from_value(self, sensor_num, temperature, force=0, apply_sol=False):
        '''
        :param sensor_num: int(), номер сенсора 0 - температурный, 1 и 2 - натяжной
        :param temperature: float(), температура
        :param force: float(), сила [даН] (для натяжных решеток)
        :return: float(), длина волны, соответствующая заданным условиям
        '''

        if sensor_num == 0:
            ret_value = self.sensors[0].wl0 * (1 + (temperature - self.sensors[0].t0) * self.sensors[0].st)
        else:
            # WL = WL_0 * (1 + ((f1 * 10 / (E * S) - (T - Ts1_0) * (CTET - CTES) / 1000000) * FG + (T - Tt_0) * ST));
            ret_value = self.sensors[sensor_num].wl0 * (1 + (((force - self.f_reserve) * 10 / (self.e * self.size[0] * self.size[1] * 1E-6) - (temperature - self.sensors[sensor_num].t0) * (
                    self.sensors[sensor_num].ctet - self.ctes) / 1E+6) * self.sensors[sensor_num].fg + (temperature - self.sensors[0].t0) * self.sensors[0].st))

        if apply_sol:
            self.__apply_sol(ret_value, True)
        return ret_value

    def get_tension_fav_ex(self, wl_tension_sensor_1, wl_tension_sensor_2,
                           wl_temperature_sensor, return_nan=False, apply_sol=False):

        ret_value = dict()

        # apply SoL compensation
        if apply_sol:
            wl_tension_sensor_1, wl_tension_sensor_2, wl_temperature_sensor = [self.__apply_sol(wl) for wl in [wl_tension_sensor_1, wl_tension_sensor_2, wl_temperature_sensor]]

        ret_value.setdefault('T_degC', None)
        ret_value.setdefault('eps1_ustr', None)
        ret_value.setdefault('eps2_ustr', None)
        ret_value.setdefault('F1_N', None)
        ret_value.setdefault('F2_N', None)
        ret_value.setdefault('Fav_N', None)
        ret_value.setdefault('Fbend_N', None)
        ret_value.setdefault('Ice_mm', None)

        if not return_nan:
            temperature_value = self.get_temperature(wl_temperature_sensor)

            eps1 = 1E+06 * ((wl_tension_sensor_1 - self.sensors[1].wl0) / self.sensors[1].wl0 - (wl_temperature_sensor - self.sensors[0].wl0) / self.sensors[0].wl0) / self.sensors[
                1].fg + (temperature_value - self.sensors[0].t0) * (self.sensors[1].ctet - self.ctes)
            eps2 = 1E+06 * ((wl_tension_sensor_2 - self.sensors[2].wl0) / self.sensors[2].wl0 - (wl_temperature_sensor - self.sensors[0].wl0) / self.sensors[0].wl0) / self.sensors[
                2].fg + (temperature_value - self.sensors[0].t0) * (self.sensors[2].ctet - self.ctes)

            f1 = (eps1 * self.e * self.size[0] * self.size[1]) / (1E+6 * 1E+6)
            f2 = (eps2 * self.e * self.size[0] * self.size[1]) / (1E+6 * 1E+6)

            f_av = (f1 + f2) / 2

            f_model = 10*(self.fmodel_f0 + self.fmodel_f1*temperature_value + self.fmodel_f2*temperature_value**2)
            f_extra = f_av - f_model

            ice_mm = None
            if self.icemodel_i2 != 0:
                under_sqrt_seq = 4*self.icemodel_i2*f_extra/10.0 + self.icemodel_i1**2
                if under_sqrt_seq > 0:
                    ice_mm = (math.sqrt(under_sqrt_seq) - self.icemodel_i1)/(2*self.icemodel_i2)

            if not -30.0 < temperature_value < 5.0:
                ice_mm = 0.0

            ret_value['T_degC'] = temperature_value
            ret_value['eps1_ustr'] = eps1
            ret_value['eps2_ustr'] = eps2
            ret_value['F1_N'] = f1
            ret_value['F2_N'] = f2
            ret_value['Fav_N'] = (f1 + f2) / 2
            ret_value['Fbend_N'] = (eps1 - eps2) / (2 * self.bend_sens)
            ret_value['Ice_mm'] = ice_mm

        return ret_value

    def __apply_sol(self, wls, back_apply=False):
        """
        Compensate SoL shift to raw wl
        :param wls: raw wls [nm], list
        :param back_apply: back compensation (wl_out > wl_in)
        :return: compensated wls [nm], list
        """
        ret_value = list()
        for wl in wls:
            wl_shift = self.distance * (self.sol_poly_k2 * wl * wl + self.sol_poly_k1 * wl + self.sol_poly_k0)
            if back_apply:
                ret_value.append(wl_shift - wl)
            else:
                ret_value.append(wl + wl_shift)

        # if list contains one one wl, then return float
        if len(ret_value) == 1:
            ret_value = ret_value[0]

        return ret_value
