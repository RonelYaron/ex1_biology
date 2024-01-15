# import argparse
import numpy as np
from itertools import groupby
from enum import Enum
import pandas as pd


serialize_dict = {'A':0, 'C':1, 'G':2, 'T':3, '-':4}
score_table = None


class directions(Enum):
    START = 0
    LEFT = 1
    UP = 2
    LAU = 3


def overlap_seg(s, t):
    seg_array = np.empty((len(s)+1, len(t)+1), dtype=object)

    seg_array[0,0] = (0, directions.START)
    for i in range(len(s)):
        seg_array[i+1,0] = (0, directions.UP)
    for i in range(len(t)):
        seg_array[0,i+1] = ((i+1)*(-2), directions.LEFT)

    for j in range(len(s)):
        for i in range(len(t)):
            if j+1!=len(s):
                left_op = seg_array[j+1, i][0] + score_table[serialize_dict[s[j]], serialize_dict['-']]
                up_op = seg_array[j, i+1][0] + score_table[serialize_dict['-'], serialize_dict[t[i]]]
                lau_op = seg_array[j, i][0] + score_table[serialize_dict[s[j]], serialize_dict[t[i]]]
            else:
                left_op = seg_array[j+1, i][0]
                up_op = seg_array[j, i+1][0]
                lau_op = seg_array[j, i][0]

            m = max(left_op, up_op, lau_op)

            if m==left_op:
                direction = directions.LEFT
            elif m==up_op:
                direction = directions.UP
            else:
                direction = directions.LAU
            

            seg_array[j+1, i+1] = (m, direction, f"{s[j]},{t[i]}")

    curr_s_index, curr_t_index = len(s), len(t)

    # print(s, t)
    while curr_s_index!=0 or curr_t_index!=0:
        direction = seg_array[curr_s_index,curr_t_index][1]
        # print(f"s_ind:{curr_s_index}, t_ind:{curr_t_index}, direction:{direction}")
        if direction==directions.LAU:
            curr_s_index, curr_t_index = curr_s_index-1, curr_t_index-1
        elif direction==directions.LEFT:
            s.insert(curr_s_index, '-')
            curr_t_index -= 1
        elif direction==directions.UP:
            t.insert(curr_t_index, '-')
            curr_s_index -= 1
        else:
            pass

    return ''.join(s), ''.join(t), seg_array[-1, -1][0]


def local_seg(s, t):
    seg_array = np.empty((len(s)+1, len(t)+1), dtype=object)

    seg_array[0,0] = (0, directions.START)
    for i in range(len(s)):
        seg_array[i+1,0] = ((i+1)*(-2), directions.UP)
    for i in range(len(t)):
        seg_array[0,i+1] = ((i+1)*(-2), directions.LEFT)

    for j in range(len(s)):
        for i in range(len(t)):
            left_op = seg_array[j+1, i][0] + score_table[serialize_dict[s[j]], serialize_dict['-']]
            up_op = seg_array[j, i+1][0] + score_table[serialize_dict['-'], serialize_dict[t[i]]]
            lau_op = seg_array[j, i][0] + score_table[serialize_dict[s[j]], serialize_dict[t[i]]]
            start_op = 0

            m = max(left_op, up_op, lau_op, start_op)

            if m==lau_op:
                direction = directions.LAU
            elif m==left_op:
                direction = directions.LEFT
            elif m==up_op:
                direction = directions.UP
            else:
                direction = directions.START
            

            seg_array[j+1, i+1] = (m, direction, f"{s[j]},{t[i]}")
            # print(f"seg_array[{j+1}, {i+1}]={seg_array[j+1, i+1]}")

    curr_s_index, curr_t_index = len(s), len(t)

    # print(s, t)
    while curr_s_index!=0 or curr_t_index!=0:
        direction = seg_array[curr_s_index,curr_t_index][1]
        # print(f"s_ind:{curr_s_index}, t_ind:{curr_t_index}, direction:{direction}")
        if direction==directions.LAU:
            curr_s_index, curr_t_index = curr_s_index-1, curr_t_index-1
        elif direction==directions.LEFT:
            s.insert(curr_s_index, '-')
            curr_t_index -= 1
        elif direction==directions.UP:
            t.insert(curr_t_index, '-')
            curr_s_index -= 1
        else:
            t.insert(curr_t_index, ' ')
            s.insert(curr_s_index, ' ')
            curr_s_index -= 1
            curr_t_index -= 1

    return ''.join(s), ''.join(t), seg_array[-1, -1][0]


def global_seg(s, t):
    seg_array = np.empty((len(s)+1, len(t)+1), dtype=object)

    seg_array[0,0] = (0, directions.START)
    for i in range(len(s)):
        seg_array[i+1,0] = ((i+1)*(-2), directions.UP)
    for i in range(len(t)):
        seg_array[0,i+1] = ((i+1)*(-2), directions.LEFT)

    for j in range(len(s)):
        for i in range(len(t)):
            left_op = seg_array[j+1, i][0] + score_table[serialize_dict[s[j]], serialize_dict['-']]
            up_op = seg_array[j, i+1][0] + score_table[serialize_dict['-'], serialize_dict[t[i]]]
            lau_op = seg_array[j, i][0] + score_table[serialize_dict[s[j]], serialize_dict[t[i]]]

            m = max(left_op, up_op, lau_op)

            if m==lau_op:
                direction = directions.LAU
            elif m==left_op:
                direction = directions.LEFT
            else:
                direction = directions.UP
            

            seg_array[j+1, i+1] = (m, direction, f"{s[j]},{t[i]}")
            # print(f"seg_array[{j+1}, {i+1}]={seg_array[j+1, i+1]}")

    curr_s_index, curr_t_index = len(s), len(t)

    # print(s, t)
    while curr_s_index!=0 or curr_t_index!=0:
        direction = seg_array[curr_s_index,curr_t_index][1]
        # print(f"s_ind:{curr_s_index}, t_ind:{curr_t_index}, direction:{direction}")
        if direction==directions.LAU:
            curr_s_index, curr_t_index = curr_s_index-1, curr_t_index-1
        elif direction==directions.LEFT:
            s.insert(curr_s_index, '-')
            curr_t_index -= 1
        elif direction==directions.UP:
            t.insert(curr_t_index, '-')
            curr_s_index -= 1
        else:
            pass

    return ''.join(s), ''.join(t), seg_array[-1, -1][0]

def fastaread(fasta_name):
    """
    Read a fasta file. For each sequence in the file, yield the header and the actual sequence.
    In Ex1 you may assume the fasta files contain only one sequence.
    You may keep this function, edit it, or delete it and implement your own reader.
    """
    f = open(fasta_name)
    faiter = (x[1] for x in groupby(f, lambda line: line.startswith(">")))
    for header in faiter:
        header = next(header)[1:].strip()
        seq = "".join(s.strip() for s in next(faiter))
        yield header, seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_a', help='Path to first FASTA file (e.g. fastas/HomoSapiens-SHH.fasta)')
    parser.add_argument('seq_b', help='Path to second FASTA file')
    parser.add_argument('--align_type', help='Alignment type (e.g. local)', required=True)
    parser.add_argument('--score', help='Score matrix in.tsv format (default is score_matrix.tsv) ', default='score_matrix.tsv')
    command_args = parser.parse_args()

    s_gen, t_gen = fastaread(command_args.seq_a), fastaread(command_args.seq_b)

    if command_args.score != None:
        tsv = pd.read_table(command_args.score)
    else:
        tsv = pd.read_table('score_matrix.tsv')

    global score_table
    del tsv[tsv.columns[0]]
    score_table = np.array(tsv)

    s, t = [], []
    for i in s_gen:
        s += list(i[1])
    for j in t_gen:
        t += list(j[1])
    
    if command_args.align_type == 'global':
        res = global_seg(s, t)
        print(f'{res[0]}\n\n{res[1]}\n\nglobal:{res[2]}')
    elif command_args.align_type == 'local':
        res = local_seg(s, t)
        print(f'{res[0]}\n\n{res[1]}\n\nlocal:{res[2]}')
    elif command_args.align_type == 'overlap':
        res = overlap_seg(s, t)
        print(f'{res[0]}\n\n{res[1]}\n\noverlap:{res[2]}')
    elif command_args.align_type == 'global_lin':
        raise NotImplementedError
    # print the best alignment and score


if __name__ == '__main__':
    # print(global_seg(list("TAAGCTA"), list("AGTA")))
    main()
