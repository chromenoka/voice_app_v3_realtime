@echo off
chcp 65001 >nul
title Git 撤销回退
echo ⚠️ 警告：即将在本地强行撤销最近一次修改，回退到上一个完好版本！
echo.
pause
H:\Git\cmd\git.exe reset --hard HEAD~1
echo.
echo ✅ 已成功撤销并恢复至上一个稳定版本！
timeout /t 3
