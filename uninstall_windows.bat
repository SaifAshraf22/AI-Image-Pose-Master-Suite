@echo off
setlocal
set "DEST=%USERPROFILE%\Documents\maya\modules"
if exist "%DEST%\AIImagePose" rmdir /S /Q "%DEST%\AIImagePose"
if exist "%DEST%\AIImagePose.mod" del /Q "%DEST%\AIImagePose.mod"
echo AI Image Pose uninstalled.
pause
