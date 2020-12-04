chcp 1251 >NUL

REM ����� websocket-������� ��� (��������� ����� � ���� �����������, ���� 7681)
set websocket_server="192.168.174.128 7681"

REM ������� ����� � ������� OAISKGN_UPK
set data_folder="C:\OAISKGN_UPK\data"





REM ***********************************************************
REM ��������� ������ ���
REM ***********************************************************
nssm install OAISKGN_UPK "OAISKGN_UPK.exe"

REM ��������� ������� <IP_����� ����������> <����>
nssm set OAISKGN_UPK AppParameters %websocket_server%

nssm set OAISKGN_UPK AppDirectory %data_folder%
nssm set OAISKGN_UPK DisplayName "�������. �� ���"
nssm set OAISKGN_UPK Description "��������� ��� (���������� ������� ����������), ��������� ���������, ������ �������� �����������, ������� � ��������, ���������� �������� � �������� �� ������ ���."
nssm set OAISKGN_UPK Start SERVICE_AUTO_START
nssm start OAISKGN_UPK



REM ***********************************************************
REM ��������� �����������
REM ***********************************************************
nssm install OAISKGN_UPK_SUPERVISOR "OAISKGN_UPK_supervisor.exe"
REM ��������� ������� <dir_size_speed_threshold_mb_per_h> <service_name> <check_interval_sec> <num_of_triggers_before_action>
nssm set OAISKGN_UPK_SUPERVISOR AppParameters "1 OAISKGN_UPK 60 5"
nssm set OAISKGN_UPK_SUPERVISOR AppDirectory %data_folder%
nssm set OAISKGN_UPK_SUPERVISOR DisplayName "�������. ���������� �� ���"
nssm set OAISKGN_UPK_SUPERVISOR Description "������������ ������� ������ �������. �� ���"
nssm set OAISKGN_UPK_SUPERVISOR Start SERVICE_AUTO_START
nssm start OAISKGN_UPK_SUPERVISOR