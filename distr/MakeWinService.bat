chcp 1251 >NUL
nssm install OAISKGN_UPK "C:\OAISKGN_UPK\distr\OAISKGN_UPK.exe"
nssm set OAISKGN_UPK AppParameters "192.168.1.92 7681"
nssm set OAISKGN_UPK AppDirectory "C:\OAISKGN_UPK\data"
nssm set OAISKGN_UPK DisplayName "�������. �� ���"
nssm set OAISKGN_UPK Description "��������� ��� (���������� ������� ����������), ��������� ���������, ������ �������� �����������, ������� � ��������, ���������� �������� � �������� �� ������ ���."
nssm set OAISKGN_UPK Start SERVICE_AUTO_START
nssm start OAISKGN_UPK