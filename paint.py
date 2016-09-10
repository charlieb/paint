import numpy as np
from itertools import product
from random import random
from math import sqrt
import svgwrite as svg
from numba import jit, float64, int64
import sys
import getopt

def make_grid(x,y):
    nlinks = 10
    gnd_max = 1.0
    gnd_min = 0.1
    len_mult_max = 1.5
    len_mult_min = 1.1
    grid = np.zeros(x*y,
            dtype=[('pos', 'float64', 2),
                ('gnd_pos', 'float64', 2),
                ('gnd_len', 'float64'),
                ('gnd_max_len', 'float64'),
                ('links', 'int64', nlinks),
                ('link_lens', 'float64', nlinks),
                ('link_max_lens', 'float64', nlinks)])
    for i, g in enumerate(grid):
        g['pos'] = [i%x, i//x]
        g['gnd_pos'] = [i%x, i//x]
        g['gnd_len'] = 0.
        g['gnd_max_len'] = gnd_min + random() * (gnd_max - gnd_min)
        g['links'] = [-1]*nlinks
        g['link_lens'] = [0]*nlinks
        g['link_max_lens'] = [0]*nlinks

    for i, g in enumerate(grid):
        targets = []
        if (i+1)%x != 0:
            targets.append(i+1)
        if i+x < x*y:
            targets.append(i+x)
        if (i+x+1)%x != 0 and i+x+1 < x*y:
            targets.append(i+x+1)
        
        for it, target in enumerate(targets):
            g['links'][it] = target
            tdist2 = (g['pos'] - grid[target]['pos'])**2
            g['link_lens'][it] = sqrt(tdist2[0] + tdist2[1])
            g['link_max_lens'][it] = g['link_lens'][it] * (len_mult_min + random() * (len_mult_max - len_mult_min))

    return grid

def expand_grid(grid):
    for g in grid:
        g['gnd_pos'] *= 1.1

#@jit(nopython=True)
def iterate(grid, t=0.1, constraint_iterations=10):
    lcs = 0.50 # link constraint strength
    gcs = 0.75 # ground constraint strength
    lef = 0.50 # link expansion factor
    gef = 0.25 # ground link expansion factor

    # Apply constraints
    for _ in range(constraint_iterations):
        for g in grid:
            for lnk, lnk_len in zip(g['links'], g['link_lens']):
                if lnk == -1: continue

                d = g['pos'] - grid[lnk]['pos']
                dmag = sqrt(d[0]**2 + d[1]**2)
                if dmag > 0.:
                    mv = lcs * 0.5 * (dmag - lnk_len) * d / dmag
                    g['pos'] -= mv
                    grid[lnk]['pos'] += mv

            # Ground linkage
            if g['gnd_len'] > -1: # not broken
                d = g['pos'] - g['gnd_pos']
                dmag = sqrt(d[0]**2 + d[1]**2)
                if dmag > 0.:
                    mv = gcs * (dmag - g['gnd_len']) * d / dmag
                    g['pos'] -= mv

    # calculate the strain and adjust link length for all the links
    for g in grid:
        for i, (lnk, lnk_len, lnk_max_len) in enumerate(zip(g['links'], g['link_lens'], g['link_max_lens'])):
            if lnk == -1: continue

            d = g['pos'] - grid[lnk]['pos']
            dmag = sqrt(d[0]**2 + d[1]**2)
            lnk_len += lef * (dmag - lnk_len)

            if lnk_len > lnk_max_len:
                # remove link
                g['links'][i] = -1

        if g['gnd_len'] > -1: # not broken
            d = g['pos'] - g['gnd_pos']
            dmag = sqrt(d[0]**2 + d[1]**2)
            g['gnd_len'] += gef * (dmag - g['gnd_len'])

            if g['gnd_len'] > g['gnd_max_len']:
                # remove link
                g['gnd_len'] = -1

def draw(grid, frame):
    dwg = svg.Drawing('test%05d.svg'%frame)
    minx = miny =  9999999
    maxx = maxy = -9999999
    for g in grid:
        minx = g['pos'][0] if g['pos'][0] < minx else minx
        maxx = g['pos'][0] if g['pos'][0] > maxx else maxx
        miny = g['pos'][1] if g['pos'][1] < miny else miny
        maxy = g['pos'][1] if g['pos'][1] > maxy else maxy

        for lnk in [k for k in g['links'] if k != -1]:
            line = svg.shapes.Line((g['pos'][0], g['pos'][1]),
                                    (grid[lnk]['pos'][0], grid[lnk]['pos'][1]),
                                    stroke='black',
                                    stroke_width=0.1)
            dwg.add(line)

    dwg.viewbox(minx=minx-2, miny=miny-2, 
                width=maxx-minx+2, height=maxy-miny+2)

    dwg.save()

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "s:l:", ('save=', 'load='))
    except getopt.GetoptError:
        print('Arg fail')

    print(opts, args)

    save_fn = 'default_save.npy'
    load_fn = ''
    for opt,arg in opts:
        if opt in ('-s', '--save'):
            save_fn = arg
        elif opt in ('-l', '--load'):
            load_fn = arg

    if load_fn != '':
        grid = np.load(load_fn)
        draw(grid, 0)
    else:
        x,y = 10,10
        grid = make_grid(x,y)
        for i in range(10):
            expand_grid(grid)
            iterate(grid)
            draw(grid, i)
            print('.', end='')

        np.save(save_fn, grid)


if __name__ == '__main__':
    main()
