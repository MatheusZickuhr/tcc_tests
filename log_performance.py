from threading import Thread
import multiprocessing
import numpy
import psutil
import matplotlib.pyplot as plt
import os
import csv

LOG_INTERVAL = 1

TIME_UNIT_AXIS_LABEL = 'Tempo (s)'

CPU_AXIS_LABEL = 'Cpu usado (%)'
CPU_CHART_TITLE = 'Uso de cpu'

MEM_AXIS_LABEL = 'Memória usada (MiB)'
MEM_CHART_TITLE = 'Uso de memória'


def create_mpl_chart(y, y_label, title, folder_path):
    x = [i for i in range(len(y))]
    fig, ax = plt.subplots()
    ax.plot(x, y)

    ax.set(xlabel=TIME_UNIT_AXIS_LABEL, ylabel=y_label,
           title=title)
    ax.grid()

    fig.savefig(os.path.join(folder_path, f'{title.strip()}.png'))


def create_csv(mem_used_log, cpu_used_log, folder_path):
    file = open(os.path.join(folder_path, f'{CPU_CHART_TITLE} {MEM_CHART_TITLE}.csv'), 'w+')
    csv_writer = csv.writer(file)

    # header
    csv_writer.writerow((CPU_AXIS_LABEL, MEM_AXIS_LABEL))

    for cpu_used, mem_used in zip(cpu_used_log, mem_used_log):
        csv_writer.writerow((cpu_used, mem_used))

    file.close()


def log(pid, folder_path):
    psu_proc = psutil.Process(pid=pid)
    cpu_used_log = []
    mem_used_log = []
    while psu_proc.is_running():
        try:
            # value in bytes converted to megabytes
            used_mem = psu_proc.memory_info()[0] / 1024 / 1024
            used_cpu = psu_proc.cpu_percent(interval=LOG_INTERVAL)
            cpu_used_log.append(min(used_cpu / multiprocessing.cpu_count(), 100))
            mem_used_log.append(used_mem)
        except psutil.NoSuchProcess:
            break
    create_mpl_chart(y=cpu_used_log, y_label=CPU_AXIS_LABEL, title=CPU_CHART_TITLE, folder_path=folder_path)
    create_mpl_chart(y=mem_used_log, y_label=MEM_AXIS_LABEL, title=MEM_CHART_TITLE, folder_path=folder_path)
    create_csv(mem_used_log=mem_used_log, cpu_used_log=cpu_used_log, folder_path=folder_path)


def log_performance(folder_path):
    def wrapper(func):
        proc = multiprocessing.Process(target=func)
        proc.start()
        log_thread = Thread(target=log, args=(proc.pid, folder_path))
        log_thread.start()
        proc.join()
        log_thread.join()

    return wrapper
