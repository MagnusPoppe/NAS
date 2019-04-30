import pickle

from src.jobs.MP import run_jobs

if __name__ == '__channelexec__':
    args = channel.receive()
    population, server_id, config = pickle.loads(args)
    try:
        new_population = run_jobs(population, server_id, config=config)
        if not any(len(x.report) > 0 or len(x.fitness) > 0 for x in new_population):
            raise Exception("New population has not trained...")
        channel.send(pickle.dumps(new_population))
    except Exception as e:
        channel.send(pickle.dumps(e))
