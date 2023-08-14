import re
import argparse
import time

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--logfile", type=str, required=True,
                        help="path to log file")
    parser.add_argument("--recent_n", type=int, default=10,
                        help="criterion on convergence: if value of recent n "
                             " checkpoints does not exceed the max value before")

    return parser.parse_args()

def main(args):

    bleu_values = []
    time_format = "%Y-%m-%d %H:%M:%S"
    with open(args.logfile, "r") as f:
        lineid = 0
        for line in f:
            time_str = " ".join(line.split()[:2]).split(".")[0]
            struct_time = time.strptime(time_str, time_format)
            step_time = time.mktime(struct_time)
            if lineid == 0:
                start_time = step_time
            step_time = step_time - start_time
            parse_step_and_bleu = line.split("step")[1].split(":")
            step = int(parse_step_and_bleu[0].strip())
            bleu = float(parse_step_and_bleu[1].strip())
            bleu_values.append((step_time, step, bleu))
            lineid += 1
            
    max_value = 0.0
    for i, ckp in enumerate(bleu_values):
        if i >= args.recent_n:
            max_value = max(max_value, bleu_values[i - args.recent_n][2])
        if max([c[2] for c in bleu_values[-args.recent_n:]]) < max_value:
            convergence_ckp = ckp
            break
    else:
        convergence_ckp = ckp

    print("Convergence Time :%s s, Step: %s , BLEU: %s" % convergence_ckp)

if __name__ == "__main__":
    args = parse_args()
    main(args)
