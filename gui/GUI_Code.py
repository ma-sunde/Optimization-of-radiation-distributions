#PyQt Modules
from PyQt5 import QtWidgets
from GUI_Light_UE import Ui_MainWindow
import sys
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QDialog, QApplication, QFileDialog, QMessageBox
import os

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setFixedSize(1100, 830)
        #self.setWindowIcon(QtGui.QIcon('Logo.png'))
        
        # Declare the filName variables in instance variables of the class 
        # so that they can be accessed from any method within the class        
        self.fileName_Right_Light = None
        self.fileName_Left_Light = None

        # Connect the buttons to the functions
        self.Browse_btn_1.clicked.connect(self.Browse_Right_Light)
        self.Browse_btn_2.clicked.connect(self.Browse_Left_Light)
        self.SetSettings_btn.clicked.connect(self.Set_Settings)

        # Apply a drop shadow effect to the buttons
        self.Browse_btn_1.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(blurRadius=25, xOffset=3, yOffset=3))
        self.Browse_btn_2.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(blurRadius=25, xOffset=3, yOffset=3))
        self.SetSettings_btn.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(blurRadius=25, xOffset=3, yOffset=3))
        self.Start_RL_btn.setGraphicsEffect(QtWidgets.QGraphicsDropShadowEffect(blurRadius=25, xOffset=3, yOffset=3))


    def Browse_Right_Light(self):
        # The file dialog will start in the current directory ('.').
        fileName_Right_Light = QFileDialog.getOpenFileName(self, "Open Image", ".", "Image Files (*.png)")
        self.Right_Light.setText(fileName_Right_Light[0])
        self.fileName_Right_Light=fileName_Right_Light[0]
        # print(fileName_Right_Light)
    
    def Browse_Left_Light(self):
        # The file dialog starts in the last directory ('').
        fileName_Left_Light = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png)")
        self.Left_Light.setText(fileName_Left_Light[0])
        self.fileName_Left_Light=fileName_Left_Light[0]
        # print(fileName_Left_Light)


    def Set_Settings(self):
        # Get the values of the distances (min, max, step)
        Min_Distance_Value=self.Min_Distance.text()
        Max_Distance_Value=self.Max_Distance.text()
        Step_Distance_Value=self.Step_Distance.text()
        print(Min_Distance_Value, Max_Distance_Value, Step_Distance_Value)

        # Get the values of the detection parameters
        Assets_Folder_Value=self.Assets_Folder.text()
        Substring_Detection_Value=self.Substring_Detection.text()
        print(Assets_Folder_Value, Substring_Detection_Value)

        # Get the values of the resolution
        Resolution_X_Value=self.Resolution_X.text()
        Resolution_Y_Value=self.Resolution_Y.text()
        print(Resolution_X_Value, Resolution_Y_Value)

        # Get the fileNames
        print(self.fileName_Left_Light, self.fileName_Right_Light)

        # Make the execution command using the given parameters
        # The f-string f"Min_Distance={min_distance}" is equivalent to the string "Min_Distance=" + str(min_distance).
        command =   (f"--args "
                    f"Min_Distance={Min_Distance_Value} "
                    f"Max_Distance={Max_Distance_Value} "
                    f"Step_Distance={Step_Distance_Value} "
                    f"FileName_Left_Light=file:///{self.fileName_Left_Light} "
                    f"FileName_Right_Light=file:///{self.fileName_Left_Light} "
                    f"Assets_Folder={Assets_Folder_Value} "
                    f"Substring_Detection={Substring_Detection_Value} "
                    f"Resolution_X={Resolution_X_Value} "
                    f"Resolution_Y={Resolution_Y_Value}")
        print(command)

        # Write command to text file called "command.txt"
        with open(r"C:/Users/hamla/Desktop/light_distribution_optimization/unreal_engine/Execution_Command_with_Args.txt", "w") as f:
            f.write(command)

        #Show Message Box when Task ist completed
        msg = QMessageBox()
        msg.setStyleSheet("QPushButton{background-color: #000000; color:#fff}}");
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Success !")
        msg.setText("The new settings have been set !")
        x=msg.exec()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())