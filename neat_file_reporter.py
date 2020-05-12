import csv
import time

from neat.reporting import BaseReporter


class NeatFileReporter(BaseReporter):

    def __init__(self, file_path):
        self.file_path = file_path
        self.start_generation_time = 0
        self.generation_time = 0
        self.logged_data = []
        self.current_generation = 0

    def start_generation(self, generation):
        self.start_generation_time = time.time()
        self.current_generation = generation + 1

    def end_generation(self, config, population, species_set):
        self.generation_time = time.time() - self.start_generation_time

    def post_evaluate(self, config, population, species, best_genome):
        self.logged_data.append({
            'generation': self.current_generation,
            'best_element_fitness': best_genome.fitness,
            'time_to_run_generation': self.generation_time,
        })

    def found_solution(self, config, generation, best):
        file = open(self.file_path, 'w+')
        csv_writer = csv.DictWriter(file, fieldnames=self.logged_data[0].keys())

        csv_writer.writeheader()

        for data in self.logged_data:
            csv_writer.writerow(data)

        file.close()

