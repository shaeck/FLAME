# Databricks notebook source
# MAGIC %sh
# MAGIC pip install scikit-image opencv-python torchmetrics

# COMMAND ----------

# MAGIC %sh
# MAGIC echo "test"

# COMMAND ----------

# MAGIC %sh
# MAGIC python main_fed.py --dataset cifar --model VGG --num_users 10 --all_clients --federated
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset cifar --model resnet --num_users 10 --all_clients --federated
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset fashion_mnist --model rlr_mnist --num_users 10 --all_clients --federated
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset fashion_mnist --model cnn --num_users 10 --all_clients --federated
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset mnist --model rlr_mnist --num_users 10 --all_clients --federated
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset mnist --model cnn --num_users 10 --all_clients --federated
# MAGIC echo "-----------------------------------------"

# COMMAND ----------

# MAGIC %sh
# MAGIC # python main_fed.py --dataset cifar --model VGG --num_users 10 --all_clients --swarm --random
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset cifar --model resnet --num_users 10 --all_clients --swarm --random
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset fashion_mnist --model rlr_mnist --num_users 10 --all_clients --swarm --random
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset fashion_mnist --model cnn --num_users 10 --all_clients --swarm --random
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset mnist --model rlr_mnist --num_users 10 --all_clients --swarm --random
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset mnist --model cnn --num_users 10 --all_clients --swarm --random
# MAGIC echo "-----------------------------------------"
