# SP-DARTS: Soft Pruning Differentiable Architecture Search

## Description:
This project aims to solve the problems of Differentiable Architecture Search by introducing new search techniques based on soft pruning, namely SP-DARTS and the progressive version SRP-DARTS, as well as a criterion for determining the importance of operations to improve the accuracy and stability of the searched models.

## Usage:

### Search and Test Model:

    python3 train_search.py --dataset {cifar10/cifar100/imagenet16-120}              \ 
                            --method {SP-DARTS/SRP-DARTS}
