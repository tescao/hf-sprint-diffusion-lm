import os
import re

all_poems = []
dir_path = 'data/forms'
for style in os.listdir(dir_path):
    for f_name in os.listdir(os.path.join(dir_path, style)):
        fpath = os.path.join(dir_path, style, f_name)
        with open(fpath, 'r') as f:
            poem = f.readlines()

            poem_c = []
            for p in poem:
                p = p.strip().lower()
                if re.match("^[a-z.,?!:;' ]+$", p):
                    if not 'written' in p :
                        poem_c.append(p)


            if not poem_c:
                continue

            sample = ''
            len_sample = 200
            i = 0
            for i in range(len(poem_c)):
                if len(sample) + len(poem_c[i]) > len_sample:
                    if sample:
                        all_poems.append(sample)

                    sample = ''
                sample += poem_c[i]
                sample += ' '

print('Got poems', len(all_poems))

with open('data/poems.txt', 'w') as f:
    f.write('\n'.join(all_poems))

