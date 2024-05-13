# SL-REP

Reputation strategy for SL, if there is any problem, please let me know.

## Requirement

Python=3.9 <br>
pytorch=1.10.1 <br>
scikit-learn=1.0.2 <br>
opencv-python=4.5.5.64 <br>
Scikit-Image=0.19.2 <br>
matplotlib=3.4.3 <br>
jupyterlab=3.3.2 <br>

Install instruction are recorded in install_requirements.sh

## Run

All combinations of datasets and models work. Depending on the configuration that you want to run some minor changes in the code have to made. For FCN, the data has to be flattened.

```
python main_fed.py --save name \
                   --init model_path \
                   --epochs 100 \
                   --num_users 100 \
                   --frac 0.1 \
                   --local_ep 3 \
                   --local_bs 50 \
                   --bs 64 \
                   --lr 0.01 \
                   --model CNN \
                   --dataset mnist \
                   --gpu 0 \
                   --server_dataset 200 \
                   --server_lr 1 \
                   --momentum 0.9 \
                   --split user \
                   --verbose \
                   --all_clients \
                   --federated \
                   --swarm \
                   --smart \
                   --random
```

Emample with with the CIFAR dataset:

```
python main_fed.py --dataset cifar --model VGG --num_users 10 --all_clients --swarm --smart
```

Results files are saved in './save' by default, including a figure and a accuracy record.
