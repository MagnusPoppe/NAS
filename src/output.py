from operator import attrgetter

from firebase.upload import upload_population


def get_improvement(individ):
    if individ.predecessor:
        return individ.validation_fitness[-1] - individ.predecessor.validation_fitness[-1]
    else:
        return 0

def generation_finished(generation, population):

    print("--> Generation {} Leaderboards:".format(generation))

    for i, individ in enumerate(population):
        print("    {rank}. {name}: Accuracy: {acc} %, Validation Accuracy: {vacc} %, improvement {improved} %".format(
            rank=i+1,
            name=population[i].ID,
            acc=round(population[i].fitness[-1] * 100, 2),
            vacc=round(population[i].validation_fitness[-1] * 100, 2),
            improved=get_improvement(population[i])
        ))

    upload_population(population)