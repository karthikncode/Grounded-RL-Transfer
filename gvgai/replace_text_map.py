""" Program to replace text for maps """

import sys
import random, collections

num_enemies = 2
num_friends = 2
min_dist_to_agent = 10  # Minimum starting distance from any object to the agent.
FRACTION = 1.0
SENTENCE_FRACTION = 1.0

# Read the game description. 
master_text_file = sys.argv[1]
text_file = sys.argv[2]

# Read the master text file.
text = collections.defaultdict(lambda:[])
for line in file(master_text_file).read().strip().split("\n"):
    obj_id = line.split()[0]
    text[obj_id].append(line)

# Read and replace.
lines = file(text_file).read().strip().split("\n")
g = file(text_file, 'w')

#get all ids first. 
ids = set()
for line in lines:
    ids.add(line.split()[0])

for obj_id in ids:
    for l in text[obj_id]:
        g.write(l+"\n")

g.close()


