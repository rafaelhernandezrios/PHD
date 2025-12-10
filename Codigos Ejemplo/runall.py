import os
import time

ruta_codigo1 = "4_LSL_3channel_Bandpower.py"
ruta_codigo2 = "csvsaver.py"
ruta_codigo3 = "csvsaverRAW.py"


os.system(f"start cmd /c python {ruta_codigo1}")
print("AURAFilteredEEG")
time.sleep(5)
os.system(f"start cmd /c python {ruta_codigo2}")
os.system(f"start cmd /c python {ruta_codigo3}")
print("CSVsaver running....")
