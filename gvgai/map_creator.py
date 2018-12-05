""" Program to generate maps """

import sys
import random

num_enemies = 2
num_friends = 2
min_dist_to_agent = 10  # Minimum starting distance from any object to the agent.
FRACTION = 1.0
SENTENCE_FRACTION = 1.0

# Read the game description. 
game_file = sys.argv[1]
master_text_file = sys.argv[2]
x_dim = int(sys.argv[3])
y_dim = int(sys.argv[4])
map_number = int(sys.argv[5])

enemies = []
friends = []

f = file(game_file).read()
friend_list = f.split("friend >")[1].split("enemy >")[0].strip().split("\n")
enemy_list = f.split("enemy >")[1].split("LevelMapping")[0].strip().split("\n")
level_mapping = f.split("LevelMapping")[1].split("InteractionSet")[0].strip().split("\n")

friends = [line.split(">")[0].strip() for line in friend_list]
enemies = [line.split(">")[0].strip() for line in enemy_list]
obj_ids = {}
for line in friend_list+enemy_list:
    obj_ids[line.split(">")[0].strip()] = line.split("objectID=")[1].strip()

symbol_map = {}
for line in level_mapping:
    parts = line.split(">")
    symbol_map[parts[1].strip()] = parts[0].strip()

# Read the text file.
text = {}
for line in file(master_text_file).read().strip().split("\n"):
    obj_id = line.split()[0]
    text[obj_id] = line

# Add a wall spanning the provided points.
def addWall(m,x1,y1,x2,y2):
    for i in range(x1,x2+1):
        for j in range(y1, y2+1):
            m[i][j] = symbol_map["wall"]
    return m

# Create the outline of the map.
m = []
for i in range(y_dim):
    if i==0 or i==y_dim-1:
        m.append(list(symbol_map["wall"] * x_dim))
    else:
        m.append(list(symbol_map["wall"] + " " * (x_dim-2) + symbol_map["wall"]))

# Add some walls. 
# addWall(m, x_dim/4-1, y_dim/2, x_dim/4+1, y_dim/2)
# addWall(m, x_dim/2, y_dim/4-1, x_dim/2, y_dim/4+1)

# Now, add in objects. 
random.shuffle(friends)
random.shuffle(enemies)

# add agent. 
agent_x = x_dim/2
agent_y = y_dim/2
m[agent_x][agent_y] = "X"

# Create appropriate text file.
h = file(master_text_file+"."+str(map_number), "w")

def sampleText(text, fraction):
    # Sample sub-parts of text according to fraction. This is a naive way of generating noisy text. 
    new_words = []
    words = text.split()
    new_words += words[:2]  # object ID
    for w in words[2:]:
        if random.random() < fraction: 
            new_words.append(w)
    return " ".join(new_words)

# Get a random ordering of the objects (2 friends and 2 enemies).
objects = friends[:num_friends] + enemies[:num_enemies]
random.shuffle(objects)

# add objects by quadrant. 
obj_cnt = 0
for x_start in [1, x_dim/2]:
    for y_start in [1, y_dim/2]:
        x = random.randint(max(1, x_start), min(x_start+x_dim/2, x_dim-2))
        y = random.randint(max(1, y_start), min(y_start+y_dim/2, y_dim-2))
        while m[x][y] != " " or abs(x - agent_x) + abs(y - agent_y) < min_dist_to_agent:
            x = random.randint(max(1, x_start), min(x_start+x_dim/2, x_dim-2))
            y = random.randint(max(1, y_start), min(y_start+y_dim/2, y_dim-2))

        obj = objects[obj_cnt]
        if obj in friends:
            m[x][y] = symbol_map[obj]
            if (random.random() < SENTENCE_FRACTION):
                h.write(sampleText(text[obj_ids[obj]], FRACTION)+"\n")
        else:
            m[x][y] = symbol_map[obj]
            if (random.random() < SENTENCE_FRACTION):
                h.write(sampleText(text[obj_ids[obj]], FRACTION)+"\n")
        obj_cnt += 1

# Output map into folder as the game file. 
g = file(game_file.split(".txt")[0]+"_lvl"+str(map_number)+".txt", "w")
for line in m:
    g.write("".join(line) + "\n")    
g.close()


h.close()






