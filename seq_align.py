import argparse
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


def print_format(s1, s2, score, align_type='global'):
    """
    prints
    :param s1:
    :param s2:
    :param score:
    :param align_type:
    :return:
    """
    s_blocks1 = [s1[n:n + 50] for n in range(0, len(s1), 50)]
    s_blocks2 = [s2[n:n + 50] for n in range(0, len(s2), 50)]
    for i in range(len(s_blocks1)):
        print(''.join(s_blocks1[i]))
        print(''.join(s_blocks2[i]))
        print()
    print(align_type + ":" + str(score))


def overlap_seg(s, t):
    seg_array = np.empty((len(s)+1, len(t)+1), dtype=object)

    seg_array[0,0] = (0, directions.START, 0)
    for i in range(len(s)):
        seg_array[i+1,0] = (0, directions.UP, 1)
    for i in range(len(t)):
        seg_array[0, i+1] = (0, directions.LEFT, 2)

    for j in range(len(s)):
        for i in range(len(t)):
            status = False

            status_left = seg_array[j+1, i][2]
            status_up = seg_array[j, i+1][2]
            status_lau = seg_array[j, i][2]

            if j+1==len(s) and status_left != 2:
                left_op = seg_array[j+1, i][0]
            else:
                left_op = seg_array[j+1, i][0] + score_table[serialize_dict[s[j]], serialize_dict['-']]

            if i+1==len(t) and status_up != 1:
                up_op = seg_array[j, i+1][0]
            else:
                up_op = seg_array[j, i+1][0] + score_table[serialize_dict['-'], serialize_dict[t[i]]]

            lau_op = seg_array[j, i][0] + score_table[serialize_dict[s[j]], serialize_dict[t[i]]]

            m = max(left_op, up_op, lau_op)

            if m==left_op:
                direction = directions.LEFT
                status = status_left
            elif m==up_op:
                direction = directions.UP
                status = status_up
            else:
                direction = directions.LAU
                status = status_lau
            
            # print(f"seg_array[{i+1},{j+1}] = {m} and direction is {direction}")
            seg_array[j+1, i+1] = (m, direction, status)

    # print(seg_array)

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

    return s, t, seg_array[-1, -1][0]


def local_seg(s, t):
    seg_array = np.empty((len(s)+1, len(t)+1), dtype=object)

    score_i, score_j = 0, 0
    score = 0

    seg_array[0,0] = (0, directions.START)
    for i in range(len(s)):
        seg_array[i+1,0] = ((i+1)*score_table[serialize_dict[s[i]], serialize_dict['-']], directions.UP)
    for i in range(len(t)):
        seg_array[0,i+1] = ((i+1)*score_table[serialize_dict['-'], serialize_dict[t[i]]], directions.LEFT)

    for j in range(len(s)):
        for i in range(len(t)):
            left_op = seg_array[j+1, i][0] + score_table[serialize_dict[s[j]], serialize_dict['-']]
            up_op = seg_array[j, i+1][0] + score_table[serialize_dict['-'], serialize_dict[t[i]]]
            lau_op = seg_array[j, i][0] + score_table[serialize_dict[s[j]], serialize_dict[t[i]]]

            start_left_op  = score_table[serialize_dict[s[j]], serialize_dict['-']]
            start_up_op = score_table[serialize_dict['-'], serialize_dict[t[i]]]
            start_lau_op = score_table[serialize_dict[s[j]], serialize_dict[t[i]]]

            m = max(left_op, up_op, lau_op, start_left_op, start_up_op, start_lau_op)
            if m > score:
                score, score_i, score_j = m, i, j

            if m==lau_op:
                direction = directions.LAU
            elif m==left_op:
                direction = directions.LEFT
            elif m==up_op:
                direction = directions.UP
            else:
                direction = directions.START
            

            seg_array[j+1, i+1] = (m, direction, f"{s[j]},{t[i]}")

    curr_s_index, curr_t_index = score_j + 1, score_i + 1
    ret_s, ret_t = [], []
    direction = directions.LAU

    while seg_array[curr_s_index, curr_t_index][1] != directions.START:
        if direction == directions.LAU:
            ret_s.insert(0, s[curr_s_index-1])
            ret_t.insert(0, t[curr_t_index-1])
            curr_s_index -= 1
            curr_t_index -= 1

        elif direction == directions.LEFT:
            ret_s.insert(0, '-')
            ret_t.insert(0, t[curr_t_index-1])
            curr_t_index -= 1

        else:
            ret_s.insert(0, s[curr_s_index-1])
            ret_t.insert(0, '-')
            curr_s_index -= 1

        direction = seg_array[curr_s_index, curr_t_index][1]

    return ret_s, ret_t, score


def global_seg(s, t):
    seg_array = np.empty((len(s)+1, len(t)+1), dtype=object)

    seg_array[0,0] = (0, directions.START)
    for i in range(len(s)):
        seg_array[i+1,0] = ((i+1)*score_table[serialize_dict[s[i]], serialize_dict['-']], directions.UP)
    for i in range(len(t)):
        seg_array[0,i+1] = ((i+1)*score_table[serialize_dict['-'], serialize_dict[t[i]]], directions.LEFT)

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

    return s, t, seg_array[-1, -1][0]

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
        print_format(res[0], res[1], res[2], 'global')
    elif command_args.align_type == 'local':
        res = local_seg(s, t)
        print_format(res[0], res[1], res[2], 'local')
    elif command_args.align_type == 'overlap':
        res = overlap_seg(s, t)
        print_format(res[0], res[1], res[2], 'overlap')
    elif command_args.align_type == 'global_lin':
        raise NotImplementedError
    # print the best alignment and score


if __name__ == '__main__':
    # tsv = pd.read_table('score_matrix.tsv')
    # del tsv[tsv.columns[0]]
    # score_table = np.array(tsv)

    # print(local_seg(list("TACCG"), list("TAG")))
    main()
