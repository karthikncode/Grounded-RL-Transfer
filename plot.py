''' plot R, Q, completion rate for multiple files at the same time'''

import sys, argparse
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
import glob
import pdb

N_max = int(sys.argv[1])
out = sys.argv[2]

sns.set(style='dark')

plt.gcf().subplots_adjust(bottom=0.15)

# labels = sys.argv[1].split(",")
file_patterns = []

# get the patterns first.
for i in range(3, len(sys.argv)):
    file_patterns.append(sys.argv[i])
    # f.append(map(float, file(sys.argv[i]).read().split('\n')[1:-1]))
# Now, read the files and populate Pandas dataframe.
values = []
timepoints = []
seeds = []
patterns = []

for i, pattern in enumerate(file_patterns):
    filenames = glob.glob(pattern+"/test_avgR.log")
    #filenames = glob.glob(pattern+"/test_avgQ.log")
    print len(filenames)
    for j, filename in enumerate(filenames):
        f = file(filename).read().strip()                
        data = map(float, f.split('\n')[1:-1])[:N_max]        
        values += data
        timepoints += range(len(data)+1)[1:]
        seeds += [j]*len(data)
        patterns += [pattern[:30]+"..."+pattern[-30:]] * len(data)

plot_values = pd.DataFrame({"values":values,
                            "timepoints":timepoints,
                            "seeds":seeds,
                            "patterns":patterns})

sns.tsplot(data=plot_values, time="timepoints", unit="seeds",
           condition="patterns", value="values")

# max_epochs = 300
# N = min(max_epochs, min(map(len, f)))

# colors = ['red', 'orange', 'b']
# markers = ['x', 6, '.']
# # linestyles = ['-', '--', '-.', ':']

# linestyles = ['-', '-','-']
# for i in range(len(f)):
#     plt.plot(f[i][:N], color=colors[i], label=labels[i], linestyle=linestyles[i], markersize=6, linewidth=1) #normal scale
#     # plt.plot(f[i][:N], color=colors[i], label=labels[i], linestyle=linestyles[i], markersize=6, linewidth=3) #normal scale
#     # plt.plot([-math.log(abs(x)) for x in f[i][:N]], color=colors[i], label=labels[i], linestyle=linestyles[i], markersize=6, linewidth=3) #log scale

# plt.xlabel('Epochs', fontsize=20)

# plt.ylabel('Reward', fontsize=25)

# plt.legend(loc=4, fontsize=15)
# labelSize=17
# plt.tick_params(axis='x', labelsize=labelSize)
# plt.tick_params(axis='y', labelsize=labelSize)


# x1,x2,y1,y2 = plt.axis()
# plt.axis((x1,x2,y1,y2)) #set y axis limit

plt.savefig('analysis/'+out+'.pdf')
