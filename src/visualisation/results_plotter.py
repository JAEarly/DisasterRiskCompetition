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


def group_configs(losses, configs, match_on, match_values):
    grouped_losses = []
    grouped_configs = []
    for match_value in match_values:
        matching_losses = []
        matching_configs = []
        for loss, config in zip(losses, configs):
            if config[match_on] == match_value:
                matching_losses.append(loss)
                matching_configs.append(config)
        grouped_losses.append(matching_losses)
        grouped_configs.append(matching_configs)
    return grouped_losses, grouped_configs


def plot_against_fixed(
    losses,
    configs,
    fixed_params,
    legend_param_name,
    legend_param_values,
    x_axis_param_name,
):
    _, axis = plt.subplots(nrows=1, ncols=1)

    losses, configs = collect_configs(losses, configs, fixed_params)
    grouped_losses, grouped_configs = group_configs(
        losses, configs, legend_param_name, legend_param_values
    )

    for y, x_configs in zip(grouped_losses, grouped_configs):
        x = [c[x_axis_param_name] for c in x_configs]
        x, y = zip(*sorted(zip(x, y)))
        label = str(x_configs[0][legend_param_name]) + " " + legend_param_name
        axis.plot(x, y, "-o", label=label)
    axis.set_title(fixed_params)
    axis.set_xlabel(x_axis_param_name)
    axis.set_ylabel("Loss")
    axis.legend(loc="best")


if __name__ == "__main__":
    _losses, _configs = parse_results(
        "./models/grid_search_alexnet_linearnn_dropout_extended/results.txt"
    )

    epochs_range = [10, 15]
    class_weight_methods = ["Unweighted", "SumBased"]
    balance_methods = ["NoSample"]
    dropout_values = [0.1, 0.25, 0.4]

    _legend_param_name = "epochs"
    _legend_param_values = epochs_range
    _x_axis_param_name = "dropout"

    _fixed_params = {"class_weight_method": "Unweighted"}
    plot_against_fixed(
        _losses,
        _configs,
        _fixed_params,
        _legend_param_name,
        _legend_param_values,
        _x_axis_param_name,
    )

    _fixed_params = {"class_weight_method": "SumBased"}
    plot_against_fixed(
        _losses,
        _configs,
        _fixed_params,
        _legend_param_name,
        _legend_param_values,
        _x_axis_param_name,
    )

    plt.tight_layout()
    plt.show()
