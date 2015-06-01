#!/usr/bin/env python
#-*- coding: utf-8 -*-

from optparse import OptionParser


import plotly.plotly as py
from plotly.graph_objs import *

from util import *


def plot_3d_scatter(x, y, z, filename):
    trace1 = Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=Marker(
            color='rgb(127, 127, 127)',
            size=12,
            symbol='circle',
            line=Line(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.9
        )
    )

    data = Data([trace1])
    layout = Layout(
        margin=Margin(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )

    fig = Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename=filename)

def plot_scatter(x, y, filename):
    trace1 = Scatter(
        x=x,
        y=y,
        mode='markers',
    )
    # trace2 = Scatter(
    #     x=[1, 2, 3, 4],
    #     y=[16, 5, 11, 9]
    # )
    data = Data([trace1])
    plot_url = py.plot(data, filename=filename)

def plot_hist(x, num_bins=50, x_label="", y_label="", title="", filename="hist.png", format="png"):
    # # the histogram of the data
    # n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
    # # add a 'best fit' line
    # sigma = 0.5
    # mu = sum(x) / (len(x) + 0.0)
    # y = mlab.normpdf(bins, mu, sigma)
    # # plt.plot(bins, y, 'r--')
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    # plt.title(title)
    #
    # # Tweak spacing to prevent clipping of ylabel
    # # plt.subplots_adjust(left=0.15)
    # plt.savefig(filename, format="png", dpi=1000)
    # plt.show()

    data = Data([
        Histogram(
            x=x
        )
    ])
    plot_url = py.plot(data, filename=filename)

def main():
    usage = "usage prog [options] arg"
    parser = OptionParser(usage=usage)
    parser.add_option("-i", "--input", dest="input",
        help="the input file")
    parser.add_option("-n", "--name", dest="name",
        help="the plog name")
    parser.add_option("-t", '--type', dest="type",
        help="the type")

    (options, remainder) = parser.parse_args()

    task = options.type
    if task == 'scatter':
        x = []
        y = []
        with open(options.input, 'r') as r:
            for line in r:
                m, n = line.strip().split(',')
                x.append(float(m))
                y.append(float(n))
        plot_scatter(x, y, "scatter-ratio")

        sys.exit(0)
    elif task == '3d':
        x = []
        y = []
        z = []
        with open(options.input, 'r') as r:
            for line in r:
                print line
                m, n, g = line.strip().split(',')
                x.append(float(m))
                y.append(float(n))
                z.append(float(g))
        plot_3d_scatter(x, y, z, "size_ratio_scatter")
        sys.exit(0)
    elif task == 'hist':

        hists = []

        hist_num = {}
        m = 0
        with open(options.input, 'r') as r:
            for line in r:
                v = float(line.strip())
                if v not in hist_num:
                    hist_num[v] = 1.0
                else:
                    hist_num[v] += 1.0


                hists.append(v)
                m += 1

        orders = order_dict(hist_num)[::-1]
        for k in orders:
            n = hist_num[k]
            print '%d %d (%.3f)' % (k, n, (n / m * 100))

        # plot_hist(hists, 20, "#number of enrollments", "#courses", "Histogram of course vs #enrollments", "course_hist.png")
        plot_hist(hists, 18, "#number of enrollments", "#users", "Histogram of user vs #enrollments", filename = options.name)

if __name__ == '__main__':
    main()
