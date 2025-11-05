import PySimpleGUI as sg
import ImageProcessLib as IPL
import DrawLib as DrawLib
import StandardDimentions as Stnds  # Import StandardDimentions
import os.path
import numpy as np
from datetime import date
import subprocess
import csv
import logging
import glob
import re

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def is_valid_filename(filename):
    # Forbidden characters in Windows filenames
    forbidden_chars = '<>:"/\\|?*'
    # Check for forbidden characters
    if any(char in filename for char in forbidden_chars):
        return False
    # Check if the filename ends with a space or period (invalid in Windows)
    if filename.endswith(' ') or filename.endswith('.'):
        return False
    return True

def run_smbclient(smb_server_ip, username, password, share, file_name, output_path):
    """Retrieve a file from an SMB share using smbclient."""
    try:
        cmd = [
            "smbclient",
            f"//{smb_server_ip}/{share}",
            "-U", f"{username}%{password}",
            "-c", f"get {file_name} {output_path}",
            "-d", "1"  # Minimal debug output
        ]
        logging.debug(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        logging.debug(f"smbclient output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"smbclient failed: {e}")
        logging.error(f"stderr: {e.stderr.strip()}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error in smbclient: {e}")
        return False

def get_next_eslah_filename(folder, file_type):
    """Determine the next filename like F1-ESLH-10.txt based on existing files."""
    pattern = os.path.join(folder, f"{file_type}-ESLH-*.txt")
    existing_files = glob.glob(pattern)
    max_number = 0
    for file in existing_files:
        match = re.search(r'-ESLH-(\d+)\.txt$', file)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)
    return os.path.join(folder, f"{file_type}-ESLH-{max_number + 1}.txt")

def Create_Eslah(folder, file_type, read_count):
    """Read last read_count columns from all_raw_results.csv, compute row averages,
    subtract from Diameters in StandardDimentions/{file_type}/{file_type}.txt,
    and save results to StandardDimentions/{file_type}/{file_type}-ESLH-{n}.txt."""
    try:
        # SMB configuration
        SMB_SERVER_IP = "192.168.1.100"
        SMB_USERNAME = "DTA-image"
        SMB_PASSWORD = "6783"
        SMB_SHARE = "ImageProcess"
        FILE_NAME = "all_raw_results.csv"
        OUTPUT_PATH = os.path.join(folder, FILE_NAME)

        # Retrieve CSV from SMB share
        if not run_smbclient(SMB_SERVER_IP, SMB_USERNAME, SMB_PASSWORD, SMB_SHARE, FILE_NAME, OUTPUT_PATH):
            sg.popup_error("Failed to retrieve all_raw_results.csv from SMB share")
            return

        # Read CSV and get last read_count columns
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            data = list(reader)
            if not data:
                sg.popup_error("CSV file is empty")
                return
            # Extract last read_count columns (excluding header)
            header = data[0][-read_count:] if read_count <= len(data[0]) else data[0]
            rows = [row[-read_count:] for row in data[1:] if row]  # Skip header
            if not rows:
                sg.popup_error("No data rows in CSV")
                return
            # Convert to float and compute row averages
            actual_dia = []
            for row in rows:
                try:
                    values = [float(x) for x in row if x]
                    if len(values) != read_count:
                        sg.popup_error(f"Row has {len(values)} values, expected {read_count}")
                        return
                    actual_dia.append(np.mean(values))
                except ValueError:
                    sg.popup_error("Invalid numeric data in CSV")
                    return

        # Read Diameters using read_info from StandardDimentions.py
        try:
            data = Stnds.read_info(file_type, base_dir=os.path.join(folder, "StandardDimentions"))
            diameters = data['arrays']['Diameters'][:, 1]  # Extract second column
        except FileNotFoundError as e:
            sg.popup_error(f"File not found: {e}")
            return
        except KeyError as e:
            sg.popup_error(f"Missing 'Diameters' section in {file_type}.txt")
            return
        except Exception as e:
            sg.popup_error(f"Error reading {file_type}.txt: {e}")
            return

        # Validate row count
        if len(diameters) != len(actual_dia):
            sg.popup_error(f"Mismatch: {len(actual_dia)} CSV rows vs {len(diameters)} diameters")
            return

        # Calculate differences: actual_dia - Diameters
        differences = [actual - desired for actual, desired in zip(actual_dia, diameters)]

        # Adjust first and last rows
        if len(differences) >= 2:
            differences[0] = (differences[0]+differences[1])/2  # First row = second row
            differences[-1] = differences[-2]  # Last row = second-to-last row

        # Save results to StandardDimentions/{file_type}/{file_type}-ESLH-{n}.txt
        output_dir = os.path.join(folder, "StandardDimentions", file_type)
        output_file = get_next_eslah_filename(output_dir, file_type)
        try:
            np.savetxt(output_file, differences, fmt='%.4f')  # No header
            #sg.popup_ok(f"Results saved to {output_file}")
        except Exception as e:
            sg.popup_error(f"Failed to save results: {e}")

    except Exception as e:
        logging.error(f"Error in Create_Eslah: {e}")
        sg.popup_error(f"Error in Create_Eslah: {e}")

def MainGUI():
    FileType = []
    CameraLayout1 = [
        [sg.In(size=(50, 1), enable_events=True, key="-FolderDir-"), sg.FolderBrowse()],
        [sg.Listbox(values=[], enable_events=True, key="-FolderList-", size=(55, 5))],
    ]
    CameraLayout2 = [
        [sg.Text("Name: "), sg.In(size=(39, 1), enable_events=True, key="-ImageName-"), sg.Button("Take Image")],
        [],
    ]
    CameraLayout = [[sg.Frame("Folder", CameraLayout1)], [sg.Frame("Image", CameraLayout2)]]
    ProcessLayout = [
        [
            sg.Radio('SX', group_id=1, key="-Radio_SX-"),
            sg.Radio('S1', group_id=1, key="-Radio_S1-"),
            sg.Radio('S2', group_id=1, key="-Radio_S2-"),
            sg.Radio('F1', group_id=1, key="-Radio_F1-"),
            sg.Radio('F2', group_id=1, key="-Radio_F2-", default=True),
            sg.Radio('F3', group_id=1, key="-Radio_F3-"),
        ],
        [sg.Button("Process Image")],
    ]
    DrawLayout = [
        [sg.Radio('Reolix', group_id=2, key="-Radio_Reolix-", default=True), sg.Radio('Triangle', group_id=2, key="-Radio_Triangle-")],
        [sg.Radio('Single Surface', group_id=3, key="-Radio_SingleGeo-"), sg.Radio('Full geometry', group_id=3, key="-Radio_FullGeo-", default=True)],
        [sg.Radio('Draw Helix line', group_id=4, key="-Radio_Helix-", default=True), sg.Radio('No Helix line', group_id=4, key="-Radio_NoHelix-")],
        [sg.Button("Draw 3D model")],
    ]
    EslahLayout = [
        [sg.Text("Read Count: "), sg.Input(default_text="2", size=(10, 1), enable_events=True, key="-ReadCount-")],
        [sg.Button("Create Eslah")],
    ]
    CNCLayout = [
        [sg.Text("step size: "), sg.In(default_text=0.2, size=(20, 1), enable_events=True, key="-stepSize-")],
        [sg.Text("maximum feed: "), sg.In(default_text=750, size=(20, 1), enable_events=True, key="-maxfeed-")],
        [sg.Radio('Simple', group_id=5, key="-Radio_SimpleCNC-", default=True), sg.Radio('Reolix', group_id=5, key="-Radio_ReolixCNC-"), sg.Radio('Metal', group_id=5, key="-Radio_MetalCNC-")],
        [sg.Button("Create CNC code")],
    ]
    layout = [
        [sg.Frame("Camera", CameraLayout, title_location=sg.TITLE_LOCATION_TOP)],
        [sg.Frame("Process", ProcessLayout, title_location=sg.TITLE_LOCATION_TOP)],
        [sg.Frame("3D Visualization", DrawLayout, title_location=sg.TITLE_LOCATION_TOP)],
        [sg.Frame("Eslah", EslahLayout, title_location=sg.TITLE_LOCATION_TOP)],
        [sg.Frame("CNC Code", CNCLayout, title_location=sg.TITLE_LOCATION_TOP)],
    ]
    window = sg.Window("Endo", layout, margins=(10, 10))
    
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event == "-FolderDir-":
            folder = values["-FolderDir-"]
            if os.path.isdir(folder):
                file_list = os.listdir(folder)
                window["-FolderList-"].update(file_list)
        if event == "Take Image":
            if values["-FolderDir-"] == "":
                sg.popup_ok("First, Select the save folder")
            elif not os.path.isdir(values["-FolderDir-"]):
                sg.popup_ok("That is not a correct DIR, Select folder again!")
            elif values["-ImageName-"] == "":
                sg.popup_ok("Enter a name for the new image")
            elif not is_valid_filename(values["-ImageName-"]):
                sg.popup_ok("That is not a valid name for a file")
            else:
                print(values["-ImageName-"])
                file_list = os.listdir(folder)
                window["-FolderList-"].update(file_list)
        if event == "Process Image":
            # Check which radio button is selected
            if values["-Radio_SX-"]:
                FileType = "SX"
            elif values["-Radio_S1-"]:
                FileType = "S1"
            elif values["-Radio_S2-"]:
                FileType = "S2"
            elif values["-Radio_F1-"]:
                FileType = "F1"
            elif values["-Radio_F2-"]:
                FileType = "F2"
            elif values["-Radio_F3-"]:
                FileType = "F3"
            else:
                FileType = []
                sg.popup_ok("Select a type")
            if FileType != []:
                if values["-FolderDir-"] == "":
                    sg.popup_ok("First, Select the save folder")
                elif not os.path.isdir(values["-FolderDir-"]):
                    sg.popup_ok("That is not a correct DIR, Select folder again!")
                else:
                    try:
                        Average_Error = IPL.MAIN(folder, FileType)
                        if sg.popup_yes_no("Do you want to save the average errors?", auto_close=True, auto_close_duration=10, keep_on_top=True) == "Yes":
                            filename = sg.tk.filedialog.asksaveasfilename(
                                defaultextension='txt',
                                filetypes=(("TXT File", "*.txt"),),
                                initialdir=folder,
                                parent=window.TKroot,
                                title="Save As"
                            )
                            try:
                                np.savetxt(filename, Average_Error, fmt='%.4f')
                                sg.popup_ok("File Saved")
                            except Exception as e:
                                print(f'Save failed: {e}')
                                sg.popup_error("Save failed!")
                    except Exception as e:
                        print(f"Something went wrong on process: {e}")
                        sg.popup_error("Something went wrong on process")
        if event == "Draw 3D model":
            # Check which radio button is selected
            if values["-Radio_SX-"]:
                FileType = "SX"
            elif values["-Radio_S1-"]:
                FileType = "S1"
            elif values["-Radio_S2-"]:
                FileType = "S2"
            elif values["-Radio_F1-"]:
                FileType = "F1"
            elif values["-Radio_F2-"]:
                FileType = "F2"
            elif values["-Radio_F3-"]:
                FileType = "F3"
            else:
                FileType = []
                sg.popup_ok("Select a type")
            if values["-Radio_Reolix-"]:
                Section = "reolix"
            elif values["-Radio_Triangle-"]:
                Section = "line"
            if values["-Radio_SingleGeo-"]:
                SingleSurface = True
            elif values["-Radio_FullGeo-"]:
                SingleSurface = False
            if values["-Radio_Helix-"]:
                drawHelix = True
            elif values["-Radio_NoHelix-"]:
                drawHelix = False
            if FileType != []:
                try:
                    DrawLib.Draw3D(FileType, Section, drawHelix, SingleSurface)
                except Exception as e:
                    print(f"Something went wrong on process: {e}")
                    sg.popup_error("Something went wrong on process")
        if event == "Create Eslah":
            # Check which radio button is selected
            if values["-Radio_SX-"]:
                FileType = "SX"
            elif values["-Radio_S1-"]:
                FileType = "S1"
            elif values["-Radio_S2-"]:
                FileType = "S2"
            elif values["-Radio_F1-"]:
                FileType = "F1"
            elif values["-Radio_F2-"]:
                FileType = "F2"
            elif values["-Radio_F3-"]:
                FileType = "F3"
            else:
                FileType = []
                sg.popup_ok("Select a type")
            if FileType != []:
                if values["-FolderDir-"] == "":
                    sg.popup_ok("First, Select the save folder")
                elif not os.path.isdir(values["-FolderDir-"]):
                    sg.popup_ok("That is not a correct DIR, Select folder again!")
                else:
                    try:
                        read_count = int(values["-ReadCount-"])
                        if read_count <= 0:
                            sg.popup_error("Read Count must be a positive integer")
                            continue
                        Create_Eslah(values["-FolderDir-"], FileType, read_count)
                    except ValueError:
                        sg.popup_error("Read Count must be a valid integer")
                    except Exception as e:
                        print(f"Error in Create Eslah: {e}")
                        sg.popup_error(f"Error in Create Eslah: {e}")
        if event == "Create CNC code":
            # Check which radio button is selected
            if values["-Radio_SX-"]:
                FileType = "SX"
            elif values["-Radio_S1-"]:
                FileType = "S1"
            elif values["-Radio_S2-"]:
                FileType = "S2"
            elif values["-Radio_F1-"]:
                FileType = "F1"
            elif values["-Radio_F2-"]:
                FileType = "F2"
            elif values["-Radio_F3-"]:
                FileType = "F3"
            else:
                FileType = []
                sg.popup_ok("Select a type")
            if values["-Radio_SimpleCNC-"]:
                IsReolix = False
            elif values["-Radio_ReolixCNC-"]:
                IsReolix = True
            elif values["-Radio_MetalCNC-"]:
                IsReolix = False
            if FileType != []:
                try:
                    savefilename = sg.tk.filedialog.asksaveasfilename(
                        defaultextension='.ngc',
                        filetypes=(("ngc File", "*.ngc"),),
                        parent=window.TKroot,
                        initialfile=FileType.lower(),
                        title="Save As"
                    )
                    print(f"Save file path: {savefilename}")
                    if not savefilename:
                        sg.popup_ok("No file selected. Operation canceled.")
                        continue
                    stepsize = float(values["-stepSize-"])
                    maxfeed = float(values["-maxfeed-"])
                    x_steps = 6
                    save_dir = os.path.dirname(savefilename)
                    if not os.path.exists(save_dir):
                        sg.popup_ok(f"Directory does not exist: {save_dir}")
                        continue
                    if DrawLib.create_CNC_code(FileType, stepsize, maxfeed, savefilename, IsReolix, x_steps):
                        sg.popup_ok("Code created successfully")
                except Exception as e:
                    print(f"Error in Create CNC code: {e}")
                    sg.popup_error(f"Error: {e}")
    window.close()

if __name__ == "__main__":
    MainGUI()