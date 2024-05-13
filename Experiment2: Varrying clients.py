# Databricks notebook source


# COMMAND ----------

# MAGIC %sh
# MAGIC pip install scikit-image opencv-python 
# MAGIC cd /Workspace/Users/samuel.haeck@ing.com/FLAME
# MAGIC # python main_fed.py --dataset cifar --model VGG --num_users 3 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset cifar --model VGG --num_users 5 --all_clients --swarm --smart
# MAGIC echo "-----------------------------------------"
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset cifar --model VGG --num_users 10 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset cifar --model VGG --num_users 20 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset cifar --model VGG --num_users 50 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset cifar --model VGG --num_users 100 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
