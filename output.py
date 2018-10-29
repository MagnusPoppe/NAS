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
            i, population[i].ID, population[i].fitness, get_improvement(population[i])
        ))

    upload_modules(population)


def output_stats(population, _time=None, plot_folder="./results"):
    import os, time
    from tensorflow import keras
    os.makedirs(plot_folder, exist_ok=True)

    print("--> Accuracy of the best architecture was {} % ({})".format(population[-1].fitness, population[-1].ID))
    print("--> Plots of different network architectures can be found under {}".format(plot_folder))

    if _time:
        print("--> Total elapsed time: {}".format(int(time.time() - _time)))

    def plot_model(individ, img_name):
        keras.utils.plot_model(individ.keras_operation, to_file='{}/{} ({}).png'.format(
            plot_folder, img_name, individ.ID)
                               )

    # Find biggest/smallest architecture:
    biggest = None
    smallest = None
    for individ in population:
        if not biggest or len(individ.children) > len(biggest.children):
            biggest = individ
        elif not smallest or len(individ.children) < len(biggest.children):
            smallest = individ

    plot_model(population[0], "lowest_accuracy")
    plot_model(population[-1], "highest_accuracy")
    if smallest: plot_model(smallest, "smallest_architecture")
    if biggest:  plot_model(biggest, "biggest_architecture")
