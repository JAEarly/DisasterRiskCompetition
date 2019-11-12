from ast import literal_eval

from matplotlib import pyplot as plt


def parse_results(filepath):
    losses = []
    configs = []
    with open(filepath, "r") as file:
        for line in file.readlines():
            if not line.startswith("+") and not line.startswith("| Pos |"):
                line = line.replace(" ", "").replace("\n", "")
                splits = line.split("|")[1:-1]
                _, loss, _, config, _ = splits

                loss = float(loss)
                config = literal_eval(config)

                losses.append(loss)
                configs.append(config)
    return losses, configs


def collect_configs(losses, configs, fixed_parameters):
    collected_losses = []
    collected_configs = []
    for loss, config in zip(losses, configs):
        collect = True
        for key, value in fixed_parameters.items():
            if config[key] != value:
                collect = False
                break
        if collect:
            collected_losses.append(loss)
            collected_configs.append(config)
    return collected_losses, collected_configs


def collect_and_plot(
    losses,
    configs,
    fixed_parameters,
    x_param_name,
    x_param_order,
    axis,
    label=None,
    style="-o",
    title=None,
):
    y, x_configs = collect_configs(losses, configs, fixed_parameters)
    x = [c[x_param_name] for c in x_configs]

    for i, x_elem in enumerate(x):
        if x_elem is None or type(x_elem) is list:
            x[i] = str(x_elem)

    data_dict = dict(zip(x, y))
    new_x = []
    new_y = []
    for x_param in x_param_order:
        new_x.append(x_param)
        new_y.append(data_dict[x_param])

    x, y = new_x, new_y

    if label is None:
        label = str(fixed_parameters)
    axis.plot(x, y, style, label=label)

    if title is not None:
        axis.set_title(title)
    else:
        axis.set_title("None")


def plot_by(
    intra_param_name,
    intra_params,
    inter_param_name,
    inter_params,
    x_param_name,
    x_param_order,
    losses,
    configs,
):
    for inter_param in inter_params:
        _, axis = plt.subplots(nrows=1, ncols=1)
        for intra_param in intra_params:
            collect_and_plot(
                losses,
                configs,
                {intra_param_name: intra_param, inter_param_name: inter_param},
                x_param_name,
                x_param_order,
                axis,
                label=intra_param,
                title=inter_param,
            )
        axis.legend(loc="best")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _losses, _configs = parse_results(
        "./models/grid_search_alexnet_linearnn/results.txt"
    )

    epochs_range = [1, 5, 10, 15, 20]
    class_weight_methods = ["Unweighted", "SumBased", "MaxBased"]
    balance_methods = ["NoSample", "UnderSample", "AvgSample", "OverSample"]

    plot_by(
        "balance_method",
        balance_methods,
        "class_weight_method",
        class_weight_methods,
        "epochs",
        epochs_range,
        _losses,
        _configs,
    )
    # plot_by('epochs', epochs_range, 'balance_method', balance_methods, 'class_weight_method', class_weight_methods, _losses, _configs)
    # plot_by('class_weight_method', class_weight_methods, 'epochs', epochs_range, 'balance_method', balance_methods, _losses, _configs)
