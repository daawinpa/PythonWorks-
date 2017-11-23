import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


##########################################
plt.figure(figsize=(6,4));
np.random.seed(12)
for i in range(8):
    x = np.arange(1000)
    y = np.random.randn(1000).cumsum()
    plt.plot(x, y, label=str(i))
plt.legend();


######################################
import prettyplotlib as ppl
plt.figure(figsize=(6,4));
np.random.seed(12)
for i in range(8):
    x = np.arange(1000)
    y = np.random.randn(1000).cumsum()
    ppl.plot(x, y, label=str(i))
ppl.legend();



# Press COMMAND + ENTER to run a single line in the console
print('Welcome to Rodeo!')

# Press COMMAND + ENTER with text selected to run multiple lines
# For example, select the following lines
x = 7
x**2
# and remember to press COMMAND + ENTER

# You can also run code directly in the console below.

#####################################################################################
# Here is an example of using Rodeo:

# We'll use the popular package called Pandas
# Install it with pip
! pip install pandas

# Import it as 'pd'
import pandas as pd

# Create a dataframe
df=pd.DataFrame({"Animal":["dog","dolphin","chicken","ant","spider"],"Legs":[4,0,2,6,8]})
df.head()

#####################################################################################
# An example of making a plot:
! pip install ggplot

from ggplot import ggplot, aes, geom_bar

ggplot(df, aes(x="Animal", weight="Legs")) + geom_bar(fill='blue')

# Find this tutorial helpful?  Checkout the blue sidebar for more tutorials!

###############################################################

plt.figure(figsize=(4,3));
np.random.seed(12)
plt.pcolormesh(np.random.rand(16, 16));
plt.colorbar();

#################################################################

plt.figure(figsize=(4,3));
np.random.seed(12);
ppl.pcolormesh(np.random.rand(16, 16));



#################################################################

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline

n = 10  # Number of nodes in the graph.
# Each node is connected to the two next nodes,
# in a circular fashion.
adj = [(i, (i+1)%n) for i in range(n)]
adj += [(i, (i+2)%n) for i in range(n)]

g = nx.Graph(adj)
print(g.nodes())

print(g.edges())

print(nx.adjacency_matrix(g))

plt.figure(figsize=(4,4));
nx.draw_circular(g)

g.add_node(n, color='#fcff00')
# We add an edge from every existing 
# node to the new node.
for i in range(n):
    g.add_edge(i, n)
    
  plt.figure(figsize=(4,4));
# We define custom node positions on a circle
# except the last node which is at the center.
t = np.linspace(0., 2*np.pi, n)
pos = np.zeros((n+1, 2))
pos[:n,0] = np.cos(t)
pos[:n,1] = np.sin(t)
# A node's color is specified by its 'color'
# attribute, or a default color if this attribute
# doesn't exist.
color = [g.node[i].get('color', '#88b0f3')
         for i in range(n+1)]
# We now draw the graph with matplotlib.
nx.draw_networkx(g, pos=pos, node_color=color)
plt.axis('off');  


plt.figure(figsize=(4,4));
nx.draw_spectral(g, node_color=color)
plt.axis('off');



##########################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

x1 = np.random.randn(80)
x2 = np.random.randn(80)
x3 = x1 * x2
y1 = .5 + 2 * x1 - x2 + 2.5 * x3 + 3 * np.random.randn(80)
y2 = .5 + 2 * x1 - x2 + 2.5 * np.random.randn(80)
y3 = y2 + np.random.randn(80)

plt.figure(figsize=(4,3));
sns.violinplot([x1,x2, x3]);


plt.figure(figsize=(4,3));
sns.regplot(x2, y2);

df = pd.DataFrame(dict(x1=x1, x2=x2, x3=x3, 
                       y1=y1, y2=y2, y3=y3))
sns.corrplot(df);


########################################

import numpy as np
import bokeh.plotting as bkh
bkh.output_notebook()


x = np.linspace(0., 1., 100)
y = np.cumsum(np.random.randn(100))


#bkh.line(x, y, line_width=5)
#bkh.show()

p = bkh.figure()
p.line(x=x, y=y)
bkh.show(p)


from bokeh.sampledata.iris import flowers
colormap = {'setosa': 'red',
            'versicolor': 'green',
            'virginica': 'blue'}
flowers['color'] = flowers['species'].map(lambda x: colormap[x])

p = bkh.figure()
p.scatter(flowers["petal_length"], 
          flowers["petal_width"],
          color=flowers["color"], 
          fill_alpha=0.25, size=10,)
bkh.show(p)

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


from mpld3 import enable_notebook
enable_notebook()

X = np.random.normal(0, 1, (100, 3))
color = np.random.random(100)
size = 500 * np.random.random(100)
plt.figure(figsize=(6,4))
plt.scatter(X[:,0], X[:,1], c=color,
            s=size, alpha=0.5, linewidths=2)
plt.grid(color='lightgray', alpha=0.7)




############################################

fig, ax = plt.subplots(3, 3, figsize=(6, 6),
                       sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.3)
X[::2,2] += 3
for i in range(3):
    for j in range(3):
        ax[i,j].scatter(X[:,i], X[:,j], c=color,
            s=.1*size, alpha=0.5, linewidths=2)
        ax[i,j].grid(color='lightgray', alpha=0.7)



#################################################

import numpy as np
from vispy import app
from vispy import gloo


c = app.Canvas(keys='interactive')


vertex = """
attribute vec2 a_position;
void main (void)
{
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"""

fragment = """
void main()
{
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
"""
program = gloo.Program(vertex, fragment)


program['a_position'] = np.c_[
        np.linspace(-1.0, +1.0, 1000).astype(np.float32),
        np.random.uniform(-0.5, +0.5, 1000).astype(np.float32)]


@c.connect
def on_resize(event):
    gloo.set_viewport(0, 0, *event.size)


@c.connect
def on_draw(event):
    gloo.clear((1,1,1,1))
    program.draw('line_strip')
    
    c.show()
app.run();
