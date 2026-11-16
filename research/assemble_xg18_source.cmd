@echo off
copy /b research\xg18-frozen-source.zip.part00+research\xg18-frozen-source.zip.part01+research\xg18-frozen-source.zip.part02+research\xg18-frozen-source.zip.part03+research\xg18-frozen-source.zip.part04+research\xg18-frozen-source.zip.part05+research\xg18-frozen-source.zip.part06+research\xg18-frozen-source.zip.part06b+research\xg18-frozen-source.zip.part07+research\xg18-frozen-source.zip.part08+research\xg18-frozen-source.zip.part09 research\xg18-frozen-source.zip
if errorlevel 1 exit /b 1
certutil -hashfile research\xg18-frozen-source.zip SHA256
