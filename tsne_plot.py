import numpy as np
import matplotlib.pyplot as plt, mpld3
import nltk
from nltk.corpus import stopwords
import sys
from sklearn.manifold import TSNE
import pdb

STOPWORDS = stopwords.words('english') #+ ['tries', 'north']


#read in tsne data
f = file(sys.argv[1], 'r').read().split('\n')

# plt.subplots_adjust(bottom = 0.1)



datax = []
datay = []
labels = []

fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
tags = {}
color_tags = []

for line in f[:-1]:
    label, x, y = line.split()
    if label in STOPWORDS: continue
    pos = nltk.pos_tag([label])
    if pos[0][1][0] is not 'N': continue
    x = float(x)
    y = float(y)
    datax.append(x)
    datay.append(y)
    labels.append(str(pos))
    if pos[0][1][0] not in tags:
        tags[pos[0][1][0]] = len(tags)
    color_tags.append(tags[pos[0][1][0]])

    # plt.annotate(label, xy = (x, y), xytext = (0, 0), textcoords = 'offset points')


# plt.scatter(datax, datay, color='white')
N = len(datax)
print N 
scatter = ax.scatter(datax,
                     datay,
                     # c=np.random.random(size=N),
                     c=color_tags,
                     s=100,
                     alpha=0.3,
                     cmap=plt.cm.jet)
ax.grid(color='white', linestyle='solid')

tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
mpld3.plugins.connect(fig, tooltip)



mpld3.show()
plt.savefig('tsne.pdf')
# plt.show()