from threading import Thread
import time
import os
import psutil


class SRULogger:
    """system resources usage log"""

    def __init__(self, file_path='log.txt', log_every_seconds=10):
        self.finished = False
        self.file_path = file_path
        self.log_every_seconds = log_every_seconds

        self.log_thread = Thread(target=self.keep_logging)
        self.log_thread.start()

    def finish(self):
        self.finished = True

    def keep_logging(self):
        logged_times = 1
        total_cpu_load = 0
        total_memory_usage = 0
        process = psutil.Process(os.getpid())
        start_time = time.time()
        while not self.finished:
            total_cpu_load += psutil.cpu_percent()
            total_memory_usage += process.memory_info().rss / 1024 / 1024  # megabytes
            with open(self.file_path, 'w+') as file:
                file.write(
                    f"""cpu_load_avg = {total_cpu_load / logged_times}
                    memory_usage_avg = {total_memory_usage / logged_times}
                    time (seconds) = {time.time() - start_time}""".replace(' ', '')
                )
            logged_times += 1
            time.sleep(self.log_every_seconds)
