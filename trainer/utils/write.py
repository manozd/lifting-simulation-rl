import os
import csv


def log_results(directory, filename, data_collect, loss_log):
    # Save the results to a file so we can graph it later.
    learn_path = os.path.join(directory, f"learn_{filename}.csv")
    with open(learn_path, "w") as data_dump:
        wr = csv.writer(data_dump)
        wr.writerows(data_collect)

    loss_path = os.path.join(directory, f"loss_{filename}.csv")
    with open(loss_path, "w") as lf:
        wr = csv.writer(lf)
        for loss_item in loss_log:
            wr.writerow(loss_item)


def params_to_filename(params):
    return (
        str(params["nn"][0])
        + "-"
        + str(params["nn"][1])
        + "-"
        + str(params["batchSize"])
        + "-"
        + str(params["buffer"])
    )
