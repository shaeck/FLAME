# Databricks notebook source
# MAGIC %sh
# MAGIC cd /Workspace/Users/samuel.haeck@ing.com/FLAME/

# COMMAND ----------

# MAGIC %sh
# MAGIC cd /Workspace/Users/samuel.haeck@ing.com/FLAME
# MAGIC pip install scikit-image opencv-python 
# MAGIC # python main_fed.py --dataset cifar --model VGG --num_users 3 --all_clients --swarm --random
# MAGIC # echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset cifar --model VGG --num_users 5 --all_clients --swarm --random
# MAGIC echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset cifar --model VGG --num_users 10 --all_clients --swarm --random
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset cifar --model VGG --num_users 20 --all_clients --swarm --random
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset cifar --model VGG --num_users 50 --all_clients --swarm --random
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset cifar --model VGG --num_users 100 --all_clients --swarm --random
# MAGIC # echo "-----------------------------------------"
