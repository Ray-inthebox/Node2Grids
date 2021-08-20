# coding:utf-8

from network import GraphNet
import argparse
import numpy as np

def configure():
    args = argparse.ArgumentParser()

    #mapping
    args.add_argument('--k', type=int, default=12)
    args.add_argument('--mapsize_a', type=int, default=11)
    args.add_argument('--mapsize_b', type=int, default=1)
    args.add_argument('--biasfactor', type=float, default=0.4)

    #data
    args.add_argument('--seed', type=int, default=100)
    args.add_argument('--dataset', type=str, default="pubmed")
    args.add_argument('--nodenum', type=int, default=19717)
    args.add_argument('--feanum', type=int, default=500)
    args.add_argument('--labelnum', type=int, default=3)
    args.add_argument('--trainstart', type=int, default=0)
    args.add_argument('--trainend', type=int, default=60)
    args.add_argument('--valstart', type=int, default=60)
    args.add_argument('--valend', type=int, default=560)
    args.add_argument('--teststart', type=int, default=18717)
    args.add_argument('--testend', type=int, default=19717)

    #network
    args.add_argument('--max_step', type=int, default=1000)
    args.add_argument('--learning_rate_base', type=float, default=0.008)
    args.add_argument('--batch_size', type=int, default=3)
    args.add_argument('--dropout', type=float, default=0.6)
    args.add_argument('--weight_decay', type=float, default=0.00015)
    args.add_argument('--attentionreg', type=float, default=0.07)
    args = args.parse_args(args=[])
    return args


def main():
    outcome = []
    conf = configure()
    for i in range(100):
        acc = GraphNet(conf).transductive_train()
        outcome.append(acc)
        print(outcome)
        Outcome = np.array(outcome)
        print("mean::", np.mean(Outcome))
        print("variance:", np.var(Outcome))
        print("std:", np.std(Outcome))
        print("max:", np.max(Outcome))
        print("min:", np.min(Outcome))

if __name__ == '__main__':
    main()
