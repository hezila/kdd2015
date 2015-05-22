#!/usr/bin/env python
#-*- coding: utf-8 -*-

from optparse import OptionParser


import plotly.plotly as py
from plotly.graph_objs import *

from util import *

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

    (options, remainder) = parser.parse_args()

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
