# Databricks notebook source
# MAGIC %sh
# MAGIC python main_fed.py --dataset cifar --model VGG --num_users 10 --all_clients --swarm --smart
# MAGIC echo "-----------------------------------------"
# MAGIC python main_fed.py --dataset cifar --model resnet --num_users 10 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset fashion_mnist --model rlr_mnist --num_users 10 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset fashion_mnist --model cnn --num_users 10 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset mnist --model rlr_mnist --num_users 10 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
# MAGIC # python main_fed.py --dataset mnist --model cnn --num_users 10 --all_clients --swarm --smart
# MAGIC # echo "-----------------------------------------"
