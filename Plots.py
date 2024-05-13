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

# folder = "save/cifar-allbad/"
folder = "save/fcn-nonebad/"
names = get_accuracy_files(folder)
fed_names = [s for s in names if "swarm" not in s]
swarm_rand_names = [s for s in names if "swarm_no" in s]
swarm_smart_names = [s for s in names if "swarm_smart" in s]

fed_names.sort()
swarm_smart_names.sort()
swarm_rand_names.sort()

fed = [read_file(folder + s) for s in fed_names]
swarm_rand = [read_file(folder + s) for s in swarm_rand_names]
swarm_smart = [read_file(folder + s) for s in swarm_smart_names]

# COMMAND ----------

def plot_six(title, index, fed, swarm_rand, swarm_smart, shape):
    x = np.linspace(0, shape, shape)
    fig, axs = plt.subplots(6, 1, figsize=(10, 30))
    x= x
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    # Plot data on each subplot
    for i, ax in enumerate(axs):
        print("FL: ", round(fed[i][index][100], 2), ", SL: ", round(swarm_rand[i][index][100], 2), ", SL-Rep: ", round(swarm_smart[i][index][100],2))
        print("-----")
        ax.plot(x[:len(fed[i][index])], fed[i][index], label='federated')
        ax.plot(x[:len(swarm_rand[i][index])], swarm_rand[i][index], label='standard swarm learning')
        ax.plot(x[:len(swarm_smart[i][index])], swarm_smart[i][index], label='reputation swarm learning')
        ax.legend()
        ax.set_xlabel('Epochs') 
        ax.set_ylabel('Accuracy') 

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
        ax.set_xlabel('Epochs') 
        ax.set_ylabel('Miliseconds') 

    # Show the plot
    plt.title(title)
    plt.show()

# COMMAND ----------

plot_six("Accuracy", 0, fed, swarm_rand, swarm_smart, 201)

# COMMAND ----------

plot_time("Round time", 1, fed, swarm_rand, swarm_smart, 100)

# COMMAND ----------

plot_time("Aggregation time", 2, fed, swarm_rand, swarm_smart, 100)

# COMMAND ----------

# MAGIC %md
# MAGIC # Clients Experiment

# COMMAND ----------

# fed, swarm_rand, swarm_smart
x = [1,2,3,4,5,6]
x_labels = [3,5,10,20,50,100]
means = [[],[],[]]
y_err_lower = [[],[],[]]
y_err_upper = [[],[],[]]
for i in range(6):
    tmp = np.mean(fed[i][0][80:])
    means[0].append(tmp)
    y_err_lower[0].append(np.min(fed[i][0][90:]) - tmp)
    y_err_upper[0].append(np.max(fed[i][0][90:]) - tmp)
    tmp = np.mean(swarm_rand[i][0][80:])
    means[1].append(np.mean(swarm_rand[i][0][80:]))
    y_err_lower[1].append(np.min(swarm_rand[i][0][90:]) - tmp)
    y_err_upper[1].append(np.max(swarm_rand[i][0][90:]) - tmp)
    tmp = np.mean(swarm_smart[i][0][80:])
    means[2].append(np.mean(swarm_smart[i][0][80:]))
    y_err_lower[2].append(np.min(swarm_smart[i][0][90:]) - tmp)
    y_err_upper[2].append(np.max(swarm_smart[i][0][90:]) - tmp)


plt.plot(x, means[0], label='Federated')
plt.plot(x, means[1], label='Swarm')
plt.plot(x, means[2], label='Smart swarm')
plt.errorbar(x, means[0], yerr=[np.abs(y_err_lower[0]), y_err_upper[0]], fmt='ob', ecolor='blue')
plt.errorbar([i+0.05 for i in x], means[1], yerr=[np.abs(y_err_lower[1]), y_err_upper[1]], fmt='o', color='orange', ecolor='orange')
plt.errorbar([i-0.05 for i in x], means[2], yerr=[np.abs(y_err_lower[2]), y_err_upper[2]], fmt='og', ecolor='green')
# Plot data on each subplot
plt.legend()
plt.xticks(x, x_labels)
plt.xlabel('Number of Clients') 
plt.ylabel('Average accuracy') 

# Show the plot
# plt.title(title)
plt.show()

# COMMAND ----------

folder = "save/1106-clinetvariations/"
names = get_accuracy_files(folder)
fed_names = [s for s in names if "swarm" not in s]
swarm_rand_names = [s for s in names if "swarm_no" in s]
swarm_smart_names = [s for s in names if "swarm_smart" in s]

fed_names.sort()
swarm_smart_names.sort()
swarm_rand_names.sort()

fed = [read_file(folder + s) for s in fed_names]
swarm_rand = [read_file(folder + s) for s in swarm_rand_names]
swarm_smart = [read_file(folder + s) for s in swarm_smart_names]


# COMMAND ----------

plot_six("Multiple clients", 0, fed, swarm_rand, swarm_smart, 101)

# COMMAND ----------

plot_time("Aggregation time", 2, fed, swarm_rand, swarm_smart, 100)

# COMMAND ----------

plot_time("Round time", 1, fed, swarm_rand, swarm_smart, 100)
