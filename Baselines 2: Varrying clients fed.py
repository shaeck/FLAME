# Databricks notebook source
# MAGIC %sh
# MAGIC pip install scikit-image opencv-python torchmetrics

# COMMAND ----------

# MAGIC %sh
# MAGIC # python main_fed.py --dataset fashion_mnist --model cnn --num_users 3 --all_clients --federated
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset fashion_mnist --model cnn --num_users 5 --all_clients --federated
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset fashion_mnist --model cnn --num_users 10 --all_clients --federated
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset fashion_mnist --model cnn --num_users 20 --all_clients --federated
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset fashion_mnist --model cnn --num_users 50 --all_clients --federated
# MAGIC echo "-----------------------------------------"
