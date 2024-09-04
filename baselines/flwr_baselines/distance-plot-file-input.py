import matplotlib.pyplot as plt
import re

def extract_accuracy_data(file_path, search_string):
    accuracy_data = []
    with open(file_path, 'r') as file:
        for line in file:
            if search_string in line:
                match = re.search(r"accuracy': \[(.*?)\]", line)
                if match:
                    data_str = match.group(1)
                    data_list = eval(f"[{data_str}]")
                    accuracy_data.extend(data_list)
    return accuracy_data

def plot_accuracy_data(file_paths, labels, search_string):
    plt.figure(figsize=(15, 8))
    
    for file_path, label in zip(file_paths, labels):
        accuracy_data = extract_accuracy_data(file_path, search_string)
        x_values = [point[0] for point in accuracy_data]
        y_values = [point[1] for point in accuracy_data]
        plt.plot(x_values, y_values, marker='o', label=label)
    
    plt.title('Accuracy over Rounds')
    plt.xlabel('Round Number')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()
    plt.savefig("./save_figure1")

# File paths to the .out files
file_paths = [
    # './nolayer0.1-6.txt',
    # './nolayer0.1.txt',
    # './random0.1-6.txt',
    # './random0.1.txt',

    # './nolayer0.9-6.txt',
    # './nolayer0.9.txt',
    # './random0.9-6.txt',
    # './random0.9.txt',

    './nolayer0.5-6.txt',
    './nolayer0.5.txt',
    './random0.5-6.txt',
    './random0.5.txt',
]

# Custom labels for each plot
labels = [
    # '6 Clients distance selection with alph 0.1',
    # '8 Clients distance selection with alph 0.1',
    # '6 Clients random selection with alph 0.1',
    # '8 Clients random selection with alph 0.1',

    # '6 Clients distance selection with alph 0.9',
    # '8 Clients distance selection with alph 0.9',
    # '6 Clients random selection with alph 0.9',
    # '8 Clients random selection with alph 0.9',

    '6 Clients distance selection with alph 0.5',
    '8 Clients distance selection with alph 0.5',
    '6 Clients random selection with alph 0.5',
    '8 Clients random selection with alph 0.5',
]

# String to search for in the .out files
search_string = "INFO flwr 2024"

# Plot the accuracy data
plot_accuracy_data(file_paths, labels, search_string)