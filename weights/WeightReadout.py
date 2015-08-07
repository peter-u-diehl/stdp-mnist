import numpy as np
from pylab import *
import matplotlib.cm as cm

ending = ''
chosenCmap = cm.get_cmap('hot_r') #cm.get_cmap('gist_ncar')

readoutnames = []
readoutnames.append('XeAe' + ending)
# readoutnames.append('YeAe' + ending)

# readoutnames.append('AeAe' + ending)
 
# readoutnames.append('AiAe' + ending)

def computePopVector(popArray):
    size = len(popArray)
    complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in xrange(size)])
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos

def get_2d_input_weights():
    weight_matrix = XA_values
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
        
    for i in xrange(n_e_sqrt):
        for j in xrange(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights

def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = figure(figsize = (18, 18))
    im2 = imshow(weights, interpolation = "nearest", vmin = 0, cmap = chosenCmap) #my_cmap
    colorbar(im2)
    title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig

bright_grey = '#f4f4f4'    # 
red   = '#ff0000'  # 
green   = '#00ff00'  # 
black   = '#000000'    # 
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('own2',[bright_grey,black])

n_input = 784
n_e = 400

for name in readoutnames:
    readout = np.load(name + '.npy')
    if (name == 'XeAe' + ending):
        value_arr = np.nan * np.ones((n_input, n_e))
    else:
        value_arr = np.nan * np.ones((n_e, n_e))
    connection_parameters = readout
    #                 print connection_parameters
    for conn in connection_parameters: 
    #                     print conn
        # don't need to pass offset as arg, now we store the parent projection
        src, tgt, value = conn
        if np.isnan(value_arr[src, tgt]):
            value_arr[src, tgt] = value
        else:
            value_arr[src, tgt] += value
    if (name == 'YeAe' + ending):
        values = np.asarray(value_arr)#.transpose()
	for i in xrange(n_e):
            print values[i,i]
    else:
        values = np.asarray(value_arr)
        
    fi = figure()
#     if name == 'AeAe' + ending or  name == 'HeHe' + ending or  name == 'AeHe' + ending or  name == 'BeHe' + ending \
#         or  name == 'CeHe' + ending or  name == 'HeAe' + ending or  name == 'HeBe' + ending or  name == 'HeCe' + ending \
#         or  name == 'AiAe' + ending or  name == 'BiBe' + ending or  name == 'CiCe' + ending or  name == 'HiHe' + ending :
# #         if name == 'A_H_E_E' or  name == 'B_H_E_E' or  name == 'C_H_E_E':
# #             popVecs = np.zeros(n_e)
# #             tempValues = np.nan_to_num(values)
# #             for x in xrange(n_e):
# #                 popVecs[x] = computePopVector(tempValues[:nEH,x].transpose())
# #             argSortPopVecs = np.argsort(popVecs, axis = 0)
# #             tempValues = np.asarray([values[:,i] for i in argSortPopVecs])
# # #             print popVecs, argSortPopVecs, np.shape(tempValues), np.shape(values)
# #             im = imshow(tempValues[:n_e, :n_e], interpolation="nearest", cmap=cm.get_cmap(my_cmap))  # copper_r   autumn_r  Greys  my_cmap  gist_rainbow
# #         else:
#         im = imshow(values[:n_e, :n_e], interpolation="nearest", cmap=cm.get_cmap(my_cmap))  # copper_r   autumn_r  Greys  my_cmap  gist_rainbow
#             #     im = imshow(values, interpolation="nearest", cmap=cm.get_cmap('gist_rainbow'))  # copper_r   autumn_r  Greys  my_cmap  gist_rainbow
#     else:
#         im = imshow(values, interpolation="nearest", cmap=cm.get_cmap('gist_ncar'), aspect='auto')  # copper_r   autumn_r  Greys  my_cmap  gist_rainbow
#         # im = imshow(values, interpolation="nearest", cmap=cm.get_cmap('spectral'))  # copper_r   autumn_r  Greys  my_cmap  gist_rainbow
    im = imshow(values, interpolation="nearest", cmap = chosenCmap, aspect='auto')  # copper_r   autumn_r  Greys  my_cmap  gist_rainbow
    cbar = colorbar(im)
    xlabel('Target excitatory neuron number')
    ylabel('Source excitatory neuron number')
    title(name)
    savefig(str(fi.number))

    if name == 'XeAe' + ending:
        XA_values = np.copy(values)#.transpose()
    if name == 'YeBe' + ending:
        YB_values = np.copy(values)#.transpose()
    if name == 'ZeCe' + ending:
        ZC_values = np.copy(values)#.transpose()
    if name == 'AeAe' + ending:
        AA_values = np.copy(values)
    if name == 'BeBe' + ending:
        BB_values = np.copy(values)
    if name == 'CeCe' + ending:
        CC_values = np.copy(values)
    if name == 'AeHe' + ending:
        AH_values = np.copy(values)
    if name == 'BeHe' + ending:
        BH_values = np.copy(values)
    if name == 'CeHe' + ending:
        CH_values = np.copy(values)
    if name == 'HeAe' + ending:
        HA_values = np.copy(values)
    if name == 'HeBe' + ending:
        HB_values = np.copy(values)
    if name == 'HeCe' + ending:
        HC_values = np.copy(values)


# readout = np.loadtxt('H_A_E_E.txt')
# for i in nEH:
    
im, fi = plot_2d_input_weights()
savefig(str(fi.number))
# 
# 
# from mpl_toolkits.mplot3d import Axes3D
# point  = np.array([1, 2, 3])
# normal = np.array([1, 1, 2])
# 
# # a plane is a*x+b*y+c*z+d=0
# # [a,b,c] is the normal. Thus, we have to calculate
# # d and we're set
# d = -point.dot(normal)
# 
# # create x,y
# xx, yy = np.meshgrid(range(200), range(200))
# 
# # calculate corresponding z
# z = (1 * xx + 1 * yy) % 200
# 
# # plot the surface
# plt3d = plt.figure().gca(projection='3d')
# plt3d.plot_surface(xx, yy, z)

XA_sum = np.nansum(XA_values[0:n_input,0:n_e], axis = 0)/n_e
AA_sum = np.nansum(AA_values[0:n_e,0:n_e], axis = 0)/n_e

fi = figure()
plot(XA_sum, AA_sum, 'w.')
for label, x, y in zip(range(200), XA_sum, AA_sum):
    plt.annotate(label, 
                xy = (x, y), xytext = (-0, 0),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                color = 'k')
xlabel('summed input from X to A for A neurons')
ylabel('summed input from A to A for A neurons')
savefig(str(fi.number))



print 'done'

show()
