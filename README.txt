AI Image Pose Maya Toolkit v1

This package turns the project into a Maya-ready tool, so the user does not need to open the Python shell and paste code.

What it includes:
1) Procedural Model mode
   - Stable procedural robot/mannequin workflow.
   - Best fallback for demos and delivery.
   - Does not require an external character rig.

2) Character Model mode
   - Advanced FBX/Mixamo-style character workflow.
   - Browse Image, Browse Character, Generate Pose.
   - Uses existing skeleton/IK retargeting workflow.

Installation:
1) Extract this ZIP anywhere.
2) Double-click install_windows.bat.
3) Restart Maya.
4) In the top Maya menu bar, click AI Image Pose > Open AI Image Pose.

Manual installation:
1) Copy AIImagePose folder to:
   C:\Users\<you>\Documents\maya\modules\AIImagePose
2) Copy AIImagePose.mod to:
   C:\Users\<you>\Documents\maya\modules\AIImagePose.mod
3) Restart Maya.

Important dependency:
The pose extractor still needs the Python virtual environment that was already used in the project.
By default it tries:
- AI_IMAGE_POSE_PYTHON environment variable
- .venv next to the installed scripts
- E:\ITi\Secitions\25.AI\maya_pose_ai\.venv\Scripts\python.exe
- E:\ITi\Secitions\25.AI\FinalProject\.venv\Scripts\python.exe

Optional environment variables:
AI_IMAGE_POSE_ROOT   = path to the main project folder
AI_IMAGE_POSE_PYTHON = full path to python.exe inside the venv

Files included:
- Procedural Model script
- Character Model script
- Launcher/menu script
- userSetup.py startup hook
- extract_pose_image.py
- Character texture maps

Recommended usage:
- Use Procedural Model when you need the safest result.
- Use Character Model when you have a real Mixamo-style FBX with skin.
