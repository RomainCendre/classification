set CURRENT=%~dp0
set PYTHONPATH=%CURRENT%
%userprofile%\.conda\envs\Skin\python.exe %CURRENT%annotate_tool\main.py
pause