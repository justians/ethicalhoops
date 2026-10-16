' Launches EthicalHoops silently in the background with no console window or dock icon.
Set WshShell = CreateObject("WScript.Shell")
strDir = Replace(WScript.ScriptFullName, WScript.ScriptName, "")
strPythonw = strDir & "venv\Scripts\pythonw.exe"
strScript = strDir & "ethicalhoops.py"

WshShell.Run """" & strPythonw & """ """ & strScript & """", 0, False

