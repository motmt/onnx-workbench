@echo off
chcp 65001 >nul
title ONNX 轻量化工作台
cd /d D:\onnxrunttime\backend
echo ============================================
echo   ONNX 轻量化工作台 启动中...
echo ============================================
echo.
echo 项目目录: D:\onnxrunttime
echo 访问地址: http://127.0.0.1:5000
echo.
echo 按 Ctrl+C 可停止服务
echo --------------------------------------------
echo.
start "" http://127.0.0.1:5000
D:\onnxrunttime\.venv\Scripts\python.exe app.py
pause
