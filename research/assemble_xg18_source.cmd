@echo off
copy /b research\xg18-frozen-source.zip.part00+research\xg18-frozen-source.zip.part01+research\xg18-frozen-source.zip.part02+research\xg18-frozen-source.zip.part03 research\xg18-frozen-source.zip
if errorlevel 1 exit /b 1
certutil -hashfile research\xg18-frozen-source.zip SHA256
