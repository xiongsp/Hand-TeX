:: Perform a Windows build.
:: Be sure to switch venv first!

.\venv\Scripts\pip install pyinstaller
.\venv\Scripts\pyinstaller.exe handtex/main.py --paths 'venv/Lib/site-packages' ^
    --onedir --noconfirm --clean --workpath=build --distpath=dist_exe --windowed ^
    --name="Hand TeX" --icon=handtex/data/custom_icons/logo.ico ^
    --copy-metadata numpy ^
    --copy-metadata packaging ^
    --copy-metadata pyyaml ^
    --collect-datas handtex

@REM Copy "docs\What is _internal.txt" "dist_exe/PanelCleaner\What is _internal.txt"
xcopy "handtex\data" "dist_exe\Hand TeX\_internal\handtex\data" /E /I /Y
cd "dist_exe\Hand TeX\_internal\handtex\data"
for /d /r . %%d in (__pycache__) do @rmdir /s /q "%%d"
