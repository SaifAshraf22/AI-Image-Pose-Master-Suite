#**AI Pose Master Suite for Autodesk Maya**
Automate 3D character posing in Autodesk Maya directly from 2D reference images.

<img width="1408" height="768" alt="Gemini_Generated_Image_iea7t8iea7t8iea7" src="https://github.com/user-attachments/assets/c211c352-8364-4543-82b6-f8abc4acba7f" />

AI Pose Master Suite is an innovative pipeline tool designed for Autodesk Maya Technical Directors (TDs) and Animators. It bridges the gap between 2D conceptual reference and 3D production by leveraging cutting-edge AI pose estimation to instantly generate accurate 3D poses within Maya.

🚀 The Core Idea
Manually matching a 3D rig to a complex 2D reference image is a time-consuming and tedious process for animators. This tool automates the technical heavy lifting, allowing artists to generate a highly accurate "first pass" pose in seconds, leaving them free to focus on the creative polish and nuances of the performance.

🌟 Key Features
One-Click Posing: Instantly transform a 2D image into a 3D pose inside Maya.

MediaPipe Integration: Utilizes Google's robust MediaPipe Pose model for accurate 3D landmark extraction (33 key points) including Z-depth estimation.

Integrated Maya UI: Built with PySide/Qt, providing a seamless, artist-friendly workflow directly within Maya.

Hybrid Pipeline: Engineered as a combination of an external Python processing script (for AI handling) and a native Maya tool (for data execution).

Smart Scaling & Alignment: Implements advanced logic to handle body scaling, depth normalization, and spatial alignment to match the scene.

Foot Grounding: Features an automated system to ensure character feet remain planted on the ground plane.

⚙️ Operation Modes
The suite offers two distinct operational workflows to suit different production needs:

Mode 1: Procedural Robot Generator (Anatomy Accuracy)
Ideal for anatomical studies, blocking, or creating initial reference guides.

Mechanism: Generates a new, dynamically built anatomy-accurate robot mesh.

Benefit: The generated rig's proportions adapt 100% to the extracted AI data, ensuring a perfect anatomical match to the reference image, regardless of standard character limitations.

Mode 2: FBX Character Retargeting (Production Workflow)
Designed for transferring poses to production-ready characters (e.g., Mixamo, Custom FBX rigs).

Mechanism: Uses advanced retargeting logic to transfer pose data to an existing rig with fixed bone lengths.

Benefit: Allows animators to use fixed-proportion characters while automatically handling complex math for grounding and scaling to achieve a natural visual approximation.

🛠️ Technology Stack
Core Language: Python 3 (both standalone and Maya-internal).

AI Engine: Google MediaPipe Pose.

DCC API: Autodesk Maya maya.cmds & Python API.

User Interface: PySide / Qt.

Data Exchange: JSON.

📋 Prerequisites & Installation
Autodesk Maya (Tested on 2022 and later).

Standalone Python 3.7+ installed on the machine.

Required Python libraries (external): mediapipe, opencv-python.

(You should add detailed installation steps here, such as cloning the repo, installing libraries, and where to place the scripts in the Maya path)

🖥️ Usage
Launch the tool inside Maya from the custom shelf or command.

Use the UI to load your 2D reference image.

Click 'Extract Pose' to run the AI analysis (external process).

Choose your desired execution mode (Robot or Retarget).

Maya executes the pose on the selected/generated character.

Author: Saif Ashraf Elsawy (CG Technical Director | Software Engineer)
