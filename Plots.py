# Databricks notebook source
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# COMMAND ----------

def get_accuracy_files(folder_path):
    accuracy_files = []
    for file_name in os.listdir(folder_path):
        if file_name.startswith("accuracy_file"):
            accuracy_files.append(file_name)
    return accuracy_files

# COMMAND ----------

def read_file(file_path):
    main_task_accuracy = []
    time_total_per_epoch = []
    time_over_aggregation_per_round = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('main_task_accuracy='):
                main_task_accuracy = eval(line[len('main_task_accuracy='):])
            elif line.startswith('time total per epoch='):
                time_total_per_epoch = eval(line[len('time total per epoch='):])
            elif line.startswith('time over aggregation per round='):
                time_over_aggregation_per_round = eval(line[len('time over aggregation per round='):])
                
    return [main_task_accuracy, time_total_per_epoch, time_over_aggregation_per_round]

# COMMAND ----------

folder = "save/2205 - badclients/"
fed = [ read_file("save/1505 - times/accuracy_file_cifar_resnet_avg_1715688466_no_malicious.txt"), 
        read_file("save/1505 - times/accuracy_file_cifar_VGG_avg_1715674442_no_malicious.txt"), 
        read_file("save/1505 - times/accuracy_file_fashion_mnist_cnn_avg_1715702708_no_malicious.txt"), 
        read_file("save/1505 - times/accuracy_file_fashion_mnist_rlr_mnist_avg_1715701354_no_malicious.txt"), 
        read_file("save/1505 - times/accuracy_file_mnist_cnn_avg_1715708232_no_malicious.txt"), 
        read_file("save/1505 - times/accuracy_file_mnist_rlr_mnist_avg_1715704094_no_malicious.txt")]
swarm_rand = [read_file(folder + "accuracy_file_cifar_resnet_avg_1716270315_no_malicious.txt"),
              read_file(folder + "accuracy_file_cifar_VGG_avg_1716241164_no_malicious.txt"), 
              read_file("save/2205 - badclients/accuracy_file_fashion_mnist_cnn_avg_1716315098_no_malicious.txt"), 
              read_file("save/2205 - badclients/accuracy_file_fashion_mnist_rlr_mnist_avg_1716310874_no_malicious.txt"), 
              read_file("save/2205 - badclients/accuracy_file_mnist_cnn_avg_1716331644_no_malicious.txt"), 
              read_file("save/2205 - badclients/accuracy_file_mnist_rlr_mnist_avg_1716317921_no_malicious.txt")]
swarm_smart = [read_file(folder + "accuracy_file_cifar_resnet_avg_1716269630_no_malicious.txt"), 
               read_file(folder + "accuracy_file_cifar_VGG_avg_1716241392_no_malicious.txt"), 
               read_file("save/2205 - badclients/accuracy_file_fashion_mnist_cnn_avg_1716314465_no_malicious.txt"),  
               read_file("save/2205 - badclients/accuracy_file_fashion_mnist_rlr_mnist_avg_1716311547_no_malicious.txt"), 
               read_file("save/2205 - badclients/accuracy_file_mnist_cnn_avg_1716332295_no_malicious.txt"),  
               read_file("save/2205 - badclients/accuracy_file_mnist_rlr_mnist_avg_1716318591_no_malicious.txt")]

# COMMAND ----------

folder = "save/0406 - badclients/"
names = get_accuracy_files(folder)
fed_names = [s for s in names if "swarm" not in s]
swarm_rand_names = [s for s in names if "swarm_no" in s]
swarm_smart_names = [s for s in names if "swarm_smart" in s]

fed = [read_file(folder + s) for s in fed_names]
swarm_rand = [read_file(folder + s) for s in swarm_rand_names]
swarm_smart = [read_file(folder + s) for s in swarm_smart_names]

# COMMAND ----------

def plot_six(title, index, fed, swarm_rand, swarm_smart, shape):
    x = np.linspace(0, shape, shape)
    fig, axs = plt.subplots(6, 1, figsize=(10, 15))
    x= x

    # Plot data on each subplot
    for i, ax in enumerate(axs):
        ax.plot(x, fed[i][index], label='federated')
        ax.plot(x, swarm_rand[i][index], label='standard swarm learning')
        ax.plot(x, swarm_smart[i][index], label='reputation swarm learning')
        ax.legend()

    # Show the plot
    # plt.title(title)
    plt.show()

def plot_time(title, index, fed, swarm_rand, swarm_smart, shape):
    x = np.linspace(0, shape, shape)
    fig, axs = plt.subplots(6, 1, figsize=(10, 15))

    # Plot data on each subplot
    for i, ax in enumerate(axs):
        print(f'experiment: {i}')
        print(f'fed: {np.mean(fed[i][index])}')
        print(f'swarm random: {np.mean(swarm_rand[i][index])}')
        print(f'swarm smart: {np.mean(swarm_smart[i][index])}')
        smooth_rand = pd.Series(swarm_rand[i][index]).rolling(window=30).mean()
        smooth_smart = pd.Series(swarm_smart[i][index]).rolling(window=30).mean()
        ax.plot(x, smooth_rand, label='standard swarm learning')
        ax.plot(x, smooth_smart, label='reputation swarm learning')
        ax.legend()

    # Show the plot
    plt.title(title)
    plt.show()

# COMMAND ----------

plot_six("Accuracy", 0, fed, swarm_rand, swarm_smart, 101)

# COMMAND ----------

plot_time("Round time", 1, fed, swarm_rand, swarm_smart, 100)

# COMMAND ----------

plot_time("Aggregation time", 2, fed, swarm_rand, swarm_smart, 100)

# COMMAND ----------

# MAGIC %md
# MAGIC # Clients Experiment

# COMMAND ----------

folder = "save/0506clientsvariation/"
names = get_accuracy_files(folder)
fed_names = [s for s in names if "swarm" not in s]
swarm_rand_names = [s for s in names if "swarm_no" in s]
swarm_smart_names = [s for s in names if "swarm_smart" in s]

fed = [read_file(folder + s) for s in fed_names]
swarm_rand = [read_file(folder + s) for s in swarm_rand_names]
swarm_smart = [read_file(folder + s) for s in swarm_smart_names]


# COMMAND ----------

plot_six("Multiple clients", 0, fed, swarm_rand, swarm_smart, 101)
