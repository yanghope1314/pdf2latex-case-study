@echo off
REM ============================================================
REM  ASCII-ONLY launcher.  Do NOT put Chinese text in this file.
REM
REM  Why: CMD reads .bat in the system ANSI codepage (GBK on
REM  Chinese Windows). Chinese paths inside would need the file
REM  saved as GBK, which then clashes with Python's UTF-8 output.
REM  Keeping this file ASCII and delegating all Chinese paths to
REM  launcher.py avoids the whole mess.
REM
REM  This file must also be PORTABLE: it looks for Python on PATH
REM  instead of hard-coding one machine's conda path. If you need
REM  a specific interpreter, set PDF2LATEX_PYTHON (see launcher.py).
REM ============================================================
chcp 65001 >nul
title PDF to LaTeX
cd /d "%~dp0"

REM Try "python" first, then the "py" launcher.
where python >nul 2>nul
if not errorlevel 1 (
    python launcher.py
    goto :done
)

where py >nul 2>nul
if not errorlevel 1 (
    py launcher.py
    goto :done
)

echo.
echo   [ERROR] Python not found on PATH.
echo.
echo   Install Python 3.10 or newer, then make sure typing
echo   "python" in a terminal works.  If Python is installed in a
echo   conda env, activate it first, or set the environment
echo   variable PDF2LATEX_PYTHON to the full path of python.exe.
echo.

:done
echo.
pause
