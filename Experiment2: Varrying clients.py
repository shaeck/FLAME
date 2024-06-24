# Databricks notebook source


# COMMAND ----------

# MAGIC %sh
# MAGIC python main_fed.py --dataset fashion_mnist --model cnn --num_users 300 --all_clients --swarm --smart
# MAGIC echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset fashion_mnist --model cnn --num_users 5 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset fashion_mnist --model cnn --num_users 10 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset fashion_mnist --model cnn --num_users 20 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset fashion_mnist --model cnn --num_users 50 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
