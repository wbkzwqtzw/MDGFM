import subprocess
import sys
import argparse

def run_experiment(dataset, drop_percent, lr,downlr,epochs,shot_num):
    cmd = [
        sys.executable,
        'execute20.py',
        '--dataset', str(dataset),
        '--drop_percent', str(drop_percent),
        '--lr',str(lr),
        '--downstreamlr',str(downlr),
        '--epochs',str(epochs),
        '--shot_num',str(shot_num)
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment with specific parameters.')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name.')
    parser.add_argument('--drop_percent', type=float, default=0.5, help='Drop percentage.')
    parser.add_argument('--lr', type=float, default=0.002, help='pretrain lr')
    parser.add_argument('--downstreamlr', type=float, default=0.03, help='downstream lr')
    parser.add_argument('--epochs', type=int, default=60, help='epoch')
    parser.add_argument('--shot_num', type=int, default=1, help='shotnum')
    args = parser.parse_args()

    run_experiment(args.dataset, args.drop_percent, args.lr, args.downstreamlr, args.epochs, args.shot_num)