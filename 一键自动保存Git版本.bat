@echo off
chcp 65001 >nul
title Git 自动保存快照
echo 正在自动打包保存当前项目快照...
H:\Git\cmd\git.exe add .
H:\Git\cmd\git.exe commit -m "AutoSave: %date% %time%"
H:\Git\cmd\git.exe push origin master
echo.
echo ✅ 已成功自动保存快照并同步推送至 GitHub！
timeout /t 3
