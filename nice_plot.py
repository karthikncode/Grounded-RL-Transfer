''' plot R, Q, completion rate for multiple files at the same time'''

import sys, argparse
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd
import glob
import pdb
import numpy as np
import copy

def movingaverage(data, window_size):
    new_data =[]
    for i in range(len(data)):
        start = max(0, i - window_size+1)
        new_data.append(sum(data[start:i+1])/float(i+1-start))

    return new_data

N_max = int(sys.argv[1])
out = sys.argv[2]

# sns.set(style='white')
sns.set(font_scale=1.6)
sns.set_style("whitegrid", {'axes.grid' : False})
# sns.set_context("paper", rc={"font.size":8,"axes.titlesize":8,"axes.labelsize":5})   

palette = ["#9b59b6", "#3498db", "#fac205", "#e74c3c", "#34495e", "#2ecc71"]
# palette = ["#3498db", "#fac205", "#34495e", "#2ecc71"]
sns.set_palette(palette)


plt.gcf().subplots_adjust(bottom=0.15)

labels = sys.argv[3].split(",")
file_patterns = []

# get the patterns first.
for i in range(4, len(sys.argv)):
    file_patterns.append(sys.argv[i])
    # f.append(map(float, file(sys.argv[i]).read().split('\n')[1:-1]))
# Now, read the files and populate Pandas dataframe.
values = []
timepoints = []
seeds = []
patterns = []
n_points_asymp = 20  # no. of points to calculate asymptotic values

for i, pattern in enumerate(file_patterns):
    filenames = glob.glob(pattern+"/test_avgR.log")
    #filenames = glob.glob(pattern+"/test_avgQ.log")
    print len(filenames)
    tot_area = []
    tot_opt = []
    tot_beg = []
    for j, filename in enumerate(filenames):
        f = file(filename).read().strip()                
        data = map(float, f.split('\n')[1:-1])[:N_max]  
        data_movavg = copy.copy(data)
        data_movavg = list(movingaverage(data_movavg, 5)) 
        values += data_movavg
        area = np.trapz(data)
        tot_area.append(float(area)/len(data))

        # Find the average of top 10 values for asymptotic performance.
        # sorted_data = sorted(map(float, f.split('\n')[1:-1]), reverse=True)
        # tot_opt += sum(sorted_data[:10])

        tot_opt.append(sum(data[-n_points_asymp:])/float(n_points_asymp))  # Take last n_points_asymp points.

        tot_beg.append(sum(data[:n_points_asymp])/float(n_points_asymp)) # Take first 20 points.

        # print "Num points: ", len(data)
        timepoints += [x*5 for x in range(len(data)+1)[1:]]
        seeds += [j]*len(data)
        patterns += [labels[i]] * len(data)
        #patterns += [pattern[:30]+"..."+pattern[-30:]] * len(data)
    try:
        print pattern, "Avg. reward (mean, std, max, min): ", np.mean(tot_area), np.std(tot_area), np.max(tot_area), np.min(tot_area)
        print "Asymptotic performance (mean, std, max, min): ", np.mean(tot_opt), np.std(tot_opt), np.max(tot_opt), np.min(tot_opt)
        print "Jumpstart performance (mean, std, max, min):", np.mean(tot_beg), np.std(tot_beg), np.max(tot_beg), np.min(tot_beg)
    except:
        pass

plot_values = pd.DataFrame({"reward":values,
                            "steps (thousands)":timepoints,
                            "seeds":seeds,
                            "Condition":patterns})

sns.tsplot(data=plot_values, time="steps (thousands)", unit="seeds",
           condition="Condition", value="reward", ci="sd")

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
