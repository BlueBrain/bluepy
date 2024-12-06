"""
This example computes numbers of inh and excitatory synaptic input synapses
for inh and exc neurons in all layers and some derivative metrics like ratio
of exc/inh synapses as a function of layer.

It plots the result in 3 figures.

Author: Eilif Muller
"""

import os
import numpy as np
import bluepy
import matplotlib.pyplot as plt

# For CellDB queries
import sqlalchemy as sqla
from bluepy.targets.mvddb import Neuron, MType, to_gid_list

circuit = ("/bgscratch/bbp/release/circuit_year/circuits_2012/"
           "23.07.12/SomatosensoryCxS1-v4.lowerCellDensity.r151/"
           "O1/merged_circuit/CircuitConfig")

cell_type_list = []
for layer in range(1, 7):
    cell_type_list += ["L%dEXC" % layer]
    cell_type_list += ["L%dINH" % layer]


def get_synapse_numbers(bc_path, hyper_column):

    c = bluepy.Circuit(bc_path)

    # exc_types = ['L%dEXC' % layer for layer in range(1,7)]
    # inh_types = ['L%dINH' % layer for layer in range(1,7)]

    cell_types = {}
    for layer in range(1, 7):
        cell_types['L%dEXC' % layer] = \
            c.mvddb.select(Neuron.hyperColumn == hyper_column,
                           Neuron.layer == layer,
                           MType.synapse_class == 'EXC')
        cell_types['L%dINH' % layer] = \
            c.mvddb.select(Neuron.hyperColumn == hyper_column,
                           Neuron.layer == layer,
                           MType.synapse_class == 'INH')

    # place to store the connection counts for each type
    type_dict = dict([(cell_type, []) for cell_type in cell_types])
    e_type_dict = dict([(cell_type, []) for cell_type in cell_types])
    i_type_dict = dict([(cell_type, []) for cell_type in cell_types])

    for cell_type in cell_types:
        print cell_type

        for n in cell_types[cell_type]:
            conns = c.get_presynaptic_data(n.gid)
            numm_conns = len(conns)  # len(h5['a%d'%n.gid])

            # numm_conns_e = len(nonzero(h5['a%d'%n.gid][:,13]>=100)[0])
            numm_conns_e = len(np.nonzero(conns[:, 13] >= 100)[0])
            numm_conns_i = numm_conns - numm_conns_e
            type_dict[cell_type].append(numm_conns)
            e_type_dict[cell_type].append(numm_conns_e)
            i_type_dict[cell_type].append(numm_conns_i)

    data = [type_dict[cell_type] for cell_type in cell_type_list]
    data_e = [e_type_dict[cell_type] for cell_type in cell_type_list]
    data_i = [i_type_dict[cell_type] for cell_type in cell_type_list]

    return data, data_e, data_i


def main():
    data1, data1_e, data1_i = get_synapse_numbers(circuit, 2)

    # what is the percent increase in number of connections?
    means1 = [np.mean(d) for d in data1]
    ratio = [np.mean(data1_e[i])/np.mean(data1[i]) for i in range(len(data1))]
    #means2 = [mean(d) for d in data2]
    #fN = np.array(means2)/np.array(means1)

    # interleave data1 and data2
    data = []
    totals = []
    totals_labels = []
    labels = []

    for i in xrange(len(data1)):
        data.append(data1_e[i])
        data.append(data1_i[i])
        totals.append(data1_e[i]+data1_i[i])
        labels.append(cell_type_list[i])
        labels.append('inh')
        totals_labels.append(cell_type_list[i])

    # Do some plotting
    plt.figure()
    b = plt.boxplot(data)
    l = plt.ylabel('absolute # of incoming synapses (exc&inh)')
    plt.xticks(1 + np.arange(len(labels)), labels, rotation=90)
    # green for x2.5
    plt.setp(b['boxes'][1::2], color='m')
    plt.setp(b['whiskers'][3::4], color='m')
    plt.setp(b['whiskers'][2::4], color='m')
    plt.setp(b['fliers'], color='c')
    #title('80k+repair_fix density vs production circuit')

    plt.figure()

    plt.plot(ratio[::2], label="excitatory cells")
    plt.plot(ratio[1::2], label="inhibitory cells")
    plt.xticks(np.arange(len(cell_type_list[::2])),
               [x[:2] for x in cell_type_list[::2]],
               rotation=90)
    plt.ylabel('% total input synapses are exc.')
    #title('mean: 80k+repair_fix density vs production circuit')

    plt.figure()
    b = plt.boxplot(totals)
    l = plt.ylabel('absolute # of total incoming synapses')
    plt.xticks(1 + np.arange(len(totals_labels)), totals_labels, rotation=90)
    # green for x2.5
    plt.setp(b['boxes'][1::2], color='m')
    plt.setp(b['whiskers'][3::4], color='m')
    plt.setp(b['whiskers'][2::4], color='m')
    plt.setp(b['fliers'], color='c')


if __name__ == "__main__":
    main()
