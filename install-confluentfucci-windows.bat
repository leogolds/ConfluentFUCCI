:: adapted from https://howchoo.com/chocolatey/use-chocolatey-and-batch-file-to-automatically-install-pc-programs


echo OFF

NET SESSION >nul 2>&1
IF %ERRORLEVEL% EQU 0 (
   echo.
) ELSE (
   echo.-------------------------------------------------------------
   echo ERROR: YOU ARE NOT RUNNING THIS WITH ADMINISTRATOR PRIVILEGES.
   echo. -------------------------------------------------------------
   echo. If you're seeing this, it means you don't have admin privileges!
   echo.
   echo. You will need to restart this program with Administrator privileges by right-clicking and select "Run As Administrator"
   echo.
   echo Press any key to leave this program. Make sure to Run As Administrator next time!
   pause

   EXIT /B 1
)

powershell -NoProfile -InputFormat None -ExecutionPolicy Bypass -Command "[System.Net.ServicePointManager]::SecurityProtocol = 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))" && SET "PATH=%PATH%;%ALLUSERSPROFILE%\chocolatey\bin"
choco feature enable -n=allowGlobalConfirmation

echo Chocolatey is ready to begin installing packages!


choco install python39 docker-desktop
echo.
echo python and docker installed
call RefreshEnv.cmd

python --version
cd %userprofile%
python -m venv venv
%userprofile%\venv\Scripts\pip install confluentfucci

echo %userprofile%\venv\Scripts\python -m confluentfucci.gui >> %userprofile%\Desktop\ConfluentFUCCI.bat


echo.
echo Your installation is complete. Please use the ConfluentFUCCI.bat on your desktop to start the GUI
pause