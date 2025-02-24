import os
import re
import numpy as np


def create_alignment_matrix(filename, num_sequences):
    # Initialize the matrix with zeros
    alignment_matrix = [[0] * num_sequences for _ in range(num_sequences)]

    # Read the alignment scores from the text file
    with open(filename, 'r') as file:
        for line in file:
            # Extract sequence numbers and alignment score using regular expressions
            match = re.search(r'\((\d+):(\d+)\)', line)
            if match:
                sequence1 = int(match.group(1))
                sequence2 = int(match.group(2))
                alignment_score = int(re.findall(r'\d+', line)[-1])

                # Populate the matrix with alignment scores
                alignment_matrix[sequence1 - 1][sequence2 - 1] = alignment_score
                alignment_matrix[sequence2 - 1][sequence1 - 1] = alignment_score

    # Fill the diagonal with 1s
    for i in range(num_sequences):
        alignment_matrix[i][i] = 100

    return alignment_matrix

os.chdir('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark/hiv_benchmarks/data/inhouse_data/clustering')
alignment = create_alignment_matrix('clustal_scores.txt', 869)
np.savetxt('alignment_matrix.csv', alignment, delimiter=',', fmt='%d')
