import subprocess
from termcolor import colored
from tqdm import tqdm
import argparse
import sys


def run_commands_in_parallel(commands):
    processes = [subprocess.Popen(cmd, shell=True) for cmd in commands]
    for p in processes:
        p.wait()


def split_list(input_list, K):
    return [input_list[i:i+K] for i in range(0, len(input_list), K)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="ec50_scaffold")
    parser.add_argument('--device', type=int, default=1)
    parser.add_argument('--num_cmds', type=int, default=1)
    parser.add_argument('--useAutoAug', action='store_true', default=False)
    parser.add_argument('--notUseRandomFeats', action='store_false', default=True)
    parser.add_argument('--erm', action='store_true', default=False)

    args = parser.parse_args()
    args = vars(args)

    dataset = args['dataset']
    device = args['device']
    inv_ws = [1e-1,1e-2,1e-3]
    reg_ws = [0.5,0.1]
    commitment_weights = [0.1,0.3,0.5]
    seeds = [1,2,3]
    cmds = []
    num_es = [1000,2000,4000]
    for e in num_es:
        for i in inv_ws:
            for r in reg_ws:
                for c in commitment_weights:
                    for seed in seeds:
                        cmd = f"python run.py --dataset  {dataset}  --num_e {e} --bs 32 --gamma 0.5 --inv_w {i} --reg_w {r} --gpu {device} --exp_name {dataset} --exp_id {seed} --epoch 200 --random_seed {seed}"
                        cmds.append(cmd)



    # Print or process the generated commands

    print(cmds[0])
    # sys.exit(0)

    commands_batches = split_list(cmds,args['num_cmds'])
    cnt = 0
    for commands in tqdm(commands_batches):
        cnt +=1
        print (colored(f"--------current progress: {cnt} out of {len(commands_batches)}---------", 'blue','on_white'))
        run_commands_in_parallel(commands)
    

