@echo off
title ONNX 轻量工作台启动器
color 0A

:: ========================================
::   ONNX 轻量工作台启动脚本
:: ========================================
echo.
echo  ========================================
echo   ONNX 轻量工作台启动器
echo  ========================================
echo.

:: ---------- 路径配置 ----------
set "WORK_DIR=D:\onnxrunttime\backend"
set "PYTHON_EXE=D:\onnxrunttime\.venv\Scripts\python.exe"
set "APP_SCRIPT=app.py"

:: ---------- 检查工作目录 ----------
if not exist "%WORK_DIR%" (
    echo  [错误] 工作目录不存在: %WORK_DIR%
    pause
    exit /b 1
)

:: ---------- 检查 Python 解释器 ----------
if not exist "%PYTHON_EXE%" (
    echo  [错误] Python 解释器未找到: %PYTHON_EXE%
    pause
    exit /b 1
)

:: ---------- 切换目录并启动 ----------
cd /d "%WORK_DIR%"
echo  [信息] 当前目录: %CD%
echo  [信息] 正在启动应用，请稍候...
echo.

"%PYTHON_EXE%" "%APP_SCRIPT%"

:: ---------- 处理退出状态 ----------
if errorlevel 1 (
    echo.
    echo  [错误] 应用异常退出，错误代码: %errorlevel%
) else (
    echo.
    echo  [信息] 应用已正常关闭
)

echo.
echo  按任意键退出...
pause > nul
exit /b 0