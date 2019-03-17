from operator import attrgetter

from firebase.upload import upload_population


def get_improvement(individ):
    if individ.predecessor:
        return individ.validation_fitness[-1] - individ.predecessor.validation_fitness[-1]
    else:
        return 0

def generation_finished(population):

    for i, individ in enumerate(population):
        print("\t{rank}. {name}: TRN={acc}%, VAL={vacc}% \t".format(
            rank=i+1,
            name=population[i].ID,
            acc=round(population[i].fitness[-1] * 100, 2),
            vacc=round(population[i].validation_fitness[-1] * 100, 2),
            improved=get_improvement(population[i])
        ), end=" ")

        print('F1 scores: ', end="")
        for key, report in individ.report[individ.epochs_trained].items():
            print(f'{key}={round(report["f1-score"] * 100)}', end="% ")
        print()

    upload_population(population)
