@echo off
setlocal
set "SRC=%~dp0AIImagePose"
set "MOD=%~dp0AIImagePose.mod"
set "DEST=%USERPROFILE%\Documents\maya\modules"
if not exist "%DEST%" mkdir "%DEST%"
if exist "%DEST%\AIImagePose" rmdir /S /Q "%DEST%\AIImagePose"
xcopy "%SRC%" "%DEST%\AIImagePose" /E /I /Y >nul
copy /Y "%MOD%" "%DEST%\AIImagePose.mod" >nul
echo.
echo AI Image Pose installed successfully.
echo Restart Maya, then open the AI Image Pose menu from the Maya top menu bar.
echo.
pause
