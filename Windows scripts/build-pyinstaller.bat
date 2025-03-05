:: Perform a Windows build.
:: Be sure to switch venv first!

.\venv\Scripts\pip install pyinstaller
.\venv\Scripts\pyinstaller.exe handtex/main.py --paths 'venv/Lib/site-packages' ^
    --onedir --noconfirm --clean --workpath=build --distpath=dist_exe --windowed ^
    --name="HandTeX" --icon=handtex/data/custom_icons/logo.ico ^
    --copy-metadata numpy ^
    --copy-metadata packaging ^
    --collect-datas handtex

xcopy "handtex\data" "dist_exe\HandTeX\_internal\handtex\data" /E /I /Y
cd "dist_exe\HandTeX\_internal\handtex\data"
for /d /r . %%d in (__pycache__) do @rmdir /s /q "%%d"
Copy "docs\What is _internal.txt" "dist_exe\HandTeX\What is _internal.txt"
