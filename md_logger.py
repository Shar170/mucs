import os
import datetime
import matplotlib.pyplot as plt

# Функция для логирования начальных параметров

def log_initialization(log_folder, L, P, z1, z2, timestamp, filename="calculation_log.md"):
    return
    filepath = os.path.join(log_folder, filename)
    with open(filepath, "a") as f:
        f.write(f"# Initialization - {timestamp}\n")
        f.write(f"- **L**: {L}\n")
        f.write(f"- **P**: {P}\n")
        f.write(f"- **z1**: {z1}\n")
        f.write(f"- **z2**: {z2}\n")
        f.write("\n")

# Функция для создания графиков и логирования вывода данных после расчета

def log_output(log_folder, output, r_array, f_array, filename="calculation_log.md"):
    return
    filepath = os.path.join(log_folder, filename)
    with open(filepath, "a") as f:
        f.write("# Output Data\n")
        for stat in output:
            f.write(f"## Time: {stat['time']} minutes\n")
            f.write(f"- **Mean Diameter**: {stat['mean']} micrometers\n")
            f.write(f"- **Mass**: {stat['mass']}\n")
            f.write("\n")

    # Создание графиков распределения для каждого шага времени
    for t, stat in enumerate(output):
        save_plot(log_folder, r_array, f_array[t], t)

# Вспомогательная функция для сохранения графика

def save_plot(log_folder, x, y, step, xlabel="Radius", ylabel="Distribution", title_prefix="Distribution at step", filename_prefix="distribution_step"):
    plt.figure()
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title_prefix} {step}")
    plt.grid(True)
    plot_filename = f"{filename_prefix}_{step}.png"
    plot_path = os.path.join(log_folder, plot_filename)
    plt.savefig(plot_path)
    plt.close()

    # Логирование графика в markdown-файл
    filepath = os.path.join(log_folder, "calculation_log.md")
    with open(filepath, "a") as f:
        f.write(f"### Step {step} Graph\n")
        f.write(f"![Graph](./{plot_filename})\n")
        f.write("\n")