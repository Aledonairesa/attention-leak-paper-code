import subprocess
import sys
import os
import csv
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import requests
import psutil
import datetime
import time

##################################################################################################
## FUNCTIONS
##################################################################################################

def validate_name(name):
    """
    Valida el nom introduït, verificant que no contingui símbols o nombres
    i que el camp estigui omplert.
    """ 
    if not name:
        msg = "El nom és un valor requerit. Ompli amb un nom."
        return msg
    else:
        try:
            for letter in name:
                if letter in [
                    '\\', '/', ':', '*', '?', '\"', '<', '>', '|', '¿', ',', '.',
                    '@', '#', '$', '&', '%', '(', ')', '[', ']', '{', '}', '¡',
                    '!', '·', '+', ';', '=', 'º', '1', '2', '3', '4', '5',
                    '6', '7', '8', '9', '0'
                ]:
                    msg = "Nom invàlid. Eviti símbols o nombres\n al nom."
                    return msg
        except ValueError:
            msg = "Nom invàlid."
            return msg
    return ""

def validate_values(result_label, name_entry, interface_var, start_both_button, progress_bar):
    """
    Valida el valor del nom i, si és correcte, inicia la captura de dades.
    """
    result_label.config(text="Comprovant els valors")

    # Nom
    result_name = validate_name(name_entry.get())

    # Si el nom no és vàlid, deixem a l'usuari reescriure'l:
    if result_name != "":
        name_entry.delete(0, tk.END)
        result_label.config(text=result_name)
    else:
        # En cas que sigui tot correcte, deshabilitem el botó i iniciem la captura
        start_both_button.config(state=tk.DISABLED)
        # Iniciem la barra de progrés perquè sembli que hi ha activitat
        progress_bar.start(10)
        start_both_processes(result_label, name_entry, interface_var)

def stop_main_program(result_label):
    """
    Atura el procés de recollida de paquets (tshark).
    """ 
    processes = [p for p in psutil.process_iter(['pid', 'name']) if 'tshark.exe' in (p.info['name'] or '').lower()]
    for process in processes:
        try:
            p = psutil.Process(process.info['pid'])
            p.terminate()  
        except psutil.NoSuchProcess:
            pass
    result_label.config(text="Captura aturada.")

def stop_additional_program():
    """
    Atura el procés de registre de timestamps (timestamps.exe).
    """
    processos = [p for p in psutil.process_iter(['pid', 'name']) if 'timestamps.exe' in (p.info['name'] or '').lower()]
    for proc in processos:
        try:
            p = psutil.Process(proc.info['pid'])
            p.terminate()  
        except psutil.NoSuchProcess:
            pass
    
    # Wait until process is fully stopped
    while any((p.info['name'] or '').lower() == 'timestamps.exe' for p in psutil.process_iter(['name'])):
        time.sleep(0.5)  # Wait 0.5 seconds and check again

def rename_files(name, result_label):
    """
    Renombra traces.csv, timestamps.csv i capture.pcap afegint {name}_{timestamp} al final.
    """
    script_directory = os.path.dirname(sys.executable)
    files_path = os.path.join(script_directory, "Files")

    # Paths originals
    traces_file = os.path.join(files_path, "traces.csv")
    timestamps_file = os.path.join(files_path, "timestamps.csv")
    raw_file = os.path.join(files_path, "capture.pcap")

    # Fem servir data/hora actual per generar un timestamp
    current_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Nous noms de fitxer
    name_no_spaces = name.replace(" ", "")
    new_traces_name = f"traces_{name_no_spaces}_{current_time}.csv"
    new_timestamps_name = f"timestamps_{name_no_spaces}_{current_time}.csv"
    new_raw_name = f"capture_{name_no_spaces}_{current_time}.pcap"

    new_traces_path = os.path.join(files_path, new_traces_name)
    new_timestamps_path = os.path.join(files_path, new_timestamps_name)
    new_raw_path = os.path.join(files_path, new_raw_name)

    # Wait and retry renaming if necessary
    max_retries = 10
    retry_delay = 0.5  # 500ms
    for attempt in range(max_retries):
        try:
            if os.path.exists(traces_file):
                os.rename(traces_file, new_traces_path)
            if os.path.exists(timestamps_file):
                os.rename(timestamps_file, new_timestamps_path)
            if os.path.exists(raw_file):
                os.rename(raw_file, new_raw_path)
            break  # If renaming succeeds, exit loop
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)  # Wait and retry
            else:
                result_label.config(text="Error: No s'ha pogut renombrar els fitxers.")
                return

    result_label.config(text=f"Captura aturada. Ja pot tancar el programa.")

def start_main_program(result_label, name_entry, interface_var, count):
    """
    Inicia la captura de paquets amb tshark, guardant primer les dades en un fitxer pcap
    i després convertint-lo a CSV.
    """
    script_directory = os.path.dirname(sys.executable)
    files_path = os.path.join(script_directory, "Files")
    os.makedirs(files_path, exist_ok=True)

    # Fitxers per a l'exportació
    traces_file_path = os.path.join(files_path, "traces.csv")
    timestamps_file_path = os.path.join(files_path, "timestamps.csv")
    raw_pcap_file_path = os.path.join(files_path, "capture.pcap")

    # Crear un fitxer pcap buit abans de la captura per evitar errors
    with open(raw_pcap_file_path, 'wb') as pcap_file:
        pass

    interface_choice = interface_var.get()
    if interface_choice == "Ethernet":
        interface_value = "Ethernet"
    else:
        interface_value = "Wi-Fi"
        
    # Comanda per executar tshark: Captura raw en format pcap
    command = [
        "C:\\Archivos de programa\\Wireshark\\tshark.exe",
        "-i", interface_value,
        "-f", "tcp or udp",
        "-w", raw_pcap_file_path
    ]
    time_limit = 12 * 60 * 60  # 12 hores
    Timeout = False

    try:
        subprocess.call(command, timeout=time_limit, shell=True)
    except subprocess.TimeoutExpired:
        Timeout = True
        count += 1
        stop_main_program(result_label)
        stop_additional_program()
    
    # Un cop finalitzada la captura (o per timeout), convertim el pcap a CSV amb els camps sol·licitats:
    with open(traces_file_path, 'w', encoding='utf-8', newline='') as csv_file:
        subprocess.run([
            "C:\\Archivos de programa\\Wireshark\\tshark.exe",
            "-r", raw_pcap_file_path,
            "-T", "fields",
            "-e", "frame.number",
            "-e", "frame.time",
            "-e", "ip.src",
            "-e", "ip.dst",
            "-e", "ip.proto",
            "-e", "tcp.srcport",
            "-e", "tcp.dstport",
            "-e", "udp.srcport",
            "-e", "udp.dstport",
            "-e", "tcp.flags.str",
            "-e", "frame.len",
            "-e", "dns.qry.name",
            "-e", "dns.a",
            "-E", "header=y",
            "-E", "separator=,",
            "-E", "quote=d",
            "-E", "occurrence=f"
        ], stdout=csv_file)

    # Ja no s'envia cap correu, simplement s'informa per consola
    print("Captura finalitzada. Arxius generats. Ja pot tancar totes dues pestanyes.")

    # Si hem superat el timeout, reiniciem la captura
    if Timeout:
        start_both_processes(result_label, name_entry, interface_var, count)

def start_additional_program():
    """
    Inicia el procés de timestamps (Timestamps.exe), escrivint a timestamps.csv
    """
    script_directory = os.path.dirname(sys.executable)
    files_path = os.path.join(script_directory, "Files")

    # Crea timestamps.csv en blanc
    with open(os.path.join(files_path, 'timestamps.csv'), 'w') as file:
        pass
    
    # Executa Timestamps.exe
    program_path = os.path.join(script_directory, "Timestamps.exe")
    subprocess.run([program_path])

def start_both_processes(result_label, name_entry, interface_var, count=1):
    """
    Inicialitza la recollida de traces i el registre de timestamps en fils diferents.
    """
    script_directory = os.path.dirname(sys.executable)
    os.makedirs(os.path.join(script_directory, "Files"), exist_ok=True)

    main_thread = threading.Thread(
        target=start_main_program,
        args=(result_label, name_entry, interface_var, count)
    )
    main_thread.start()
    print('Captura iniciada.')
    result_label.config(text="Captura iniciada.")
    
    additional_thread = threading.Thread(target=start_additional_program)
    additional_thread.start()

##################################################################################################
## HELPER FUNCTIONS FOR THE UI
##################################################################################################

def show_about():
    messagebox.showinfo(
        "Quant a aquest programa",
        """Programa de recollida de dades. Versió 2.7.

Totes les dades registrades són guardades localment a la carpeta 'Files'.

Els arxius generats són dos CSV: 'traces' i 'timestamps', i ara també un fitxer pcap amb la captura bruta.

El vostre nom només s'utilitza per diferenciar els usuaris, afegint el nom al nom dels fitxers generats.

Per a més informació, contacta amb aledonairesa@gmail.com.

Gràcies per contribuir!
"""
    )

def start_window():
    # Crear finestra principal
    window = tk.Tk()
    window.title("Programa de recollida de dades")
    # Definir la mida per defecte (més gran)
    window.geometry("500x420")

    # Configurem el color de fons de la finestra (blau suau)
    window.configure(bg="#D0E8F2")

    # Fem servir ttk.Style per tenir un aspecte consistent i personalitzar-lo
    style = ttk.Style(window)
    style.theme_use('default')
    
    # Definim personalitzacions de color i font
    style.configure("TFrame", background="#D0E8F2")
    style.configure("TLabel", background="#D0E8F2", foreground="#2C3E50", font=("Segoe UI", 11, "bold"))
    style.configure("TButton", background="#A3CBEF", foreground="#2C3E50", font=("Segoe UI", 11, "bold"), padding=6)
    style.map("TButton", background=[("active", "#7FB3D5")])
    style.configure("TEntry", padding=5)

    # Títol
    title_label = ttk.Label(window, text="Programa de recollida de dades", font=("Segoe UI", 16), background="#D0E8F2", foreground="#2C3E50")
    title_label.pack(pady=(15, 5))

    # Estil dels botons de Wi-Fi/Ethernet
    style.configure("TRadiobutton", background="#D0E8F2", foreground="#2C3E50", font=("Segoe UI", 11))
    style.map("TRadiobutton", background=[("active", "#A3CBEF")])
    
    # Creem un frame contenedor pel contingut principal
    main_frame = ttk.Frame(window)
    main_frame.pack(fill="both", expand=True, padx=20, pady=20)

    # Afegim un menú superior per a "Ajuda" / "Sobre..."
    menubar = tk.Menu(window)
    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="Quant a...", command=show_about)
    menubar.add_cascade(label="Ajuda", menu=help_menu)
    window.config(menu=menubar)

    # Etiqueta de nom
    name_label = ttk.Label(main_frame, text="Nom i cognom:")
    name_label.pack(pady=(0, 5))

    # Entry de nom
    name_entry = ttk.Entry(main_frame, width=40)
    name_entry.pack(pady=(0, 15))

    # Etiqueta d'interfície
    interface_label = ttk.Label(main_frame, text="Seleccioni la interfície de xarxa:")
    interface_label.pack(pady=(0, 5))

    # Variable del tipus d'interfície i els seus botons
    interface_var = tk.StringVar()
    wifi_button = ttk.Radiobutton(main_frame, text="Wi-Fi", variable=interface_var, value="Wi-Fi")
    ethernet_button = ttk.Radiobutton(main_frame, text="Ethernet (cable)", variable=interface_var, value="Ethernet")
    wifi_button.pack()
    ethernet_button.pack()
    wifi_button.invoke()  # Per defecte, Wi-Fi

    # Barra de progrés (indeterminat)
    progress_bar = ttk.Progressbar(main_frame, orient='horizontal', mode='indeterminate', length=200)
    progress_bar.pack(pady=(15, 15))

    # Result label
    result_label = ttk.Label(main_frame, text="")
    result_label.pack(pady=(0, 15))

    # Botó d'iniciar
    start_both_button = ttk.Button(
        main_frame,
        text="Iniciar",
        command=lambda: validate_values(result_label, name_entry, interface_var, start_both_button, progress_bar)
    )
    start_both_button.pack(side="left", padx=(40, 10))

    # Botó d'aturar
    stop_button = ttk.Button(
        main_frame,
        text="Aturar",
        command=lambda: [
            stop_main_program(result_label),
            stop_additional_program(),
            rename_files(name_entry.get(), result_label),
            progress_bar.stop()
        ]
    )
    stop_button.pack(side="right", padx=(10, 40))

    window.mainloop()

#---------------------------------WIRESHARK INSTALLATION CHECK------------------------------------#

directories = ["C:\\Program Files\\Wireshark", "C:\\Program Files (x86)\\Wireshark"]
for directory in directories:
    if os.path.exists(directory):
        print("Wireshark ja està instal·lat.")
        break
else:
    print("Wireshark no està instal·lat. Instal·lant...")
    wireshark_url = "https://deic.uab.cat/~cborrego/focus/Wireshark-4.2.4-x64.exe"
    wireshark_exe = "Wireshark-4.2.4-x64.exe"
    response = requests.get(wireshark_url)
    with open(wireshark_exe, "wb") as file:
        file.write(response.content)
    subprocess.run([wireshark_exe], shell=True)
    print("Wireshark ja està instal·lat.")

#---------------------------------PROGRAM STARTING------------------------------------#
start_window()
