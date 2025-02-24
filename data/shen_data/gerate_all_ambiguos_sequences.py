import os
import time
import pandas as pd

os.chdir('/home/rocio/Documents/Doutorado_projeto/artigo_benchmark/hiv_benchmarks/data/shen_data')
data = pd.read_csv('ambiguous_protease_sequences.csv', header=None).iloc[1:].reset_index(drop=True)
data.to_csv('data_0.csv', header = False, index=False)

counter = 1

while True:
    start_time = time.time()  # Start time measurement
    old_file = f'data_{counter - 1}.csv'
    
    with open(old_file, 'r') as read_file:
        new_file = f'data_{counter}.csv'
        
        with open(new_file, 'w') as write_file:
            times_else_executed = 0

            for line in read_file:
                list_data = [element.replace('"', '') for element in line.strip().split(',')]

                part1 = list_data[:16]
                part2 = list_data[16:]

                first_element_nchar_gt1 = next(((idx, elem) for idx, elem in enumerate(part2) if len(elem) > 1), None)

                if first_element_nchar_gt1 is None:
                    new_line = ','.join(part1 + part2) + '\n'
                    write_file.write(new_line)
                else:
                    new_lists = [part2[:] for _ in range(len(first_element_nchar_gt1[1]))]
                    times_else_executed += 1

                    for i, sublist in enumerate(new_lists):
                        sublist[first_element_nchar_gt1[0]] = first_element_nchar_gt1[1][i]
                        new_line = ','.join(part1 + sublist) + '\n'
                        write_file.write(new_line)
        
        # Remove o arquivo antigo
        os.remove(old_file)

    counter += 1
    if times_else_executed == 0:
        break
    
    end_time = time.time()  # End time measurement
    print(f'Files processed: {counter - 1}. Elapsed time: {end_time - start_time:.2f} seconds')
