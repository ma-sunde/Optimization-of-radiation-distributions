# ⚙️ GUI for UE Scenarios Design
This GUI allows the input of variables that will be used to design Unreal Engine scenarios. The Unreal Engine executable gets the assigned variables as arguments. These will be set in the "unreal_engine/Execution_Command_with_Args.txt" file.
# 🛠 The configurable variables

- **Max_Distance**: The maximum distance between the camera and the detection object.

- **Min_Distance**: The minimum distance between the camera and the detection object.

- **Step_Distance**: The step size for incrementing the distance between the camera and the detection object.

- **Right_Light_Distribution**: The file path to the right light distribution (.png-File).

- **Left_Light_Distribution**: The file path to the left light distribution (.png-File).

- **Assets_Folder**: The folder containing the detection objects (respective to the UE-Project-Folder hierarchy).

- **Substring_Detektionsobjekte**: The substring to use when searching for detection objects within the specified assets folder.

- **Output resolution Render**: The resolution of the rendered image.
    
## 🖼️ Screenshot

[![GUI-Screenshot.png](https://i.postimg.cc/y6c8QzRG/GUI-Screenshot.png)](https://postimg.cc/BtZ3bkM5)