import time

from src.jobs.job_manager import JobManager

manager = None


def initialize(config, func, start, end):
    global manager
    manager = JobManager(config, func, start, end)

def queue_all(population, config, priority=False):
    for individ in population:
        queue(individ, config, priority)

def queue(individ, config, priority=False):
    global manager
    manager.queue_job((individ, config), priority=priority)

def await_all_jobs_finish():
    global manager
    manager.await_all_jobs_finish()
    # Wait for RSYNC to finish
    time.sleep(24)
