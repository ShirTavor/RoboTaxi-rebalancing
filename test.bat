@echo off
@rem path "C:\Program Files\IBM\ILOG\CPLEX_Studio201\opl\bin\x64_win64";%PATH%
@rem path "C:\ProgramData\Anaconda3";%PATH%
@rem set WaitTime=5
@rem ping localhost -n %WaitTime% > nul

python AV16.py train.sorted 1200 720 30 5700 6 900 30 8 TR
python AV16.py train.sorted 1200 720 30 6000 6 900 30 8 TR
python AV16.py train.sorted 1200 720 30 6300 6 900 30 8 TR
python AV16.py train.sorted 1200 720 30 6600 6 900 30 8 TR

pause