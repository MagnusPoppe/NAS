from operator import attrgetter

from firebase.upload import upload_modules


def get_improvement(individ):
    if individ.predecessor:
        return individ.fitness - individ.predecessor.fitness
    else:
        return 0

def generation_finished(generation, population):

    print("--> Generation {} Leaderboards:".format(generation))

    for i, individ in enumerate(population):
        print("    {}. {}:  Accuracy: {} %, improvement {} %".format(
            i+1, population[i].ID, population[i].fitness, get_improvement(population[i])
        ))

    upload_modules(population)