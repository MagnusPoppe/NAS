from firebase.upload import upload_population


def get_improvement(individ):
    if len(individ.report) > 1:
        keys = list(individ.report.keys())
        impr = individ.report[keys[-1]]['weighted avg']["f1-score"] \
               - individ.report[keys[-2]]['weighted avg']["f1-score"]
        return str(round(impr, 1))
    else:
        return "-"


def col(val: str, cols: int):
    return val + " " * (cols - len(val))


def generation_finished(population, prefix):
    print(prefix)
    rankings = []
    longest_string = 14
    for i, individ in enumerate(population):
        rank = f"    {i + 1}. {individ.ID}: "
        longest_string = max(longest_string, len(rank))
        rankings += [rank]

    header = col("    SPECIMEN", longest_string) \
             + col("ACC", 7) + col("VACC", 7) \
             + col("IMPR", 7) \
             + col("1", 7) \
             + col("2", 7) \
             + col("3", 7) \
             + col("4", 7) \
             + col("5", 7) \
             + col("6", 7) \
             + col("7", 7) \
             + col("8", 7) \
             + col("9", 7) \
             + col("10", 7) \
             + col("MIAVG", 7) \
             + col("MAVG", 7) \
             + col("WAVG", 7)

    print(header)

    for rank, individ in zip(rankings, population):
        rank += " " * (longest_string - len(rank))
        rank += col(str(round(individ.fitness[-1] * 100, 1)), 7)
        rank += col(str(round(individ.validation_fitness[-1] * 100, 1)), 7)
        rank += col(get_improvement(individ), 7)
        for report in individ.report[individ.epochs_trained].values():
            rank += f'{col(str(round(report["f1-score"] * 100, 1)), 7)}'
        print(rank)

    upload_population(population)
