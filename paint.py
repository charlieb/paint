import numpy as np
from itertools import product
from random import random
from math import sqrt
import svgwrite as svg
from numba import jit, int64, float64
import sys
import getopt

nlinks = 10
struct_dtype = np.dtype([
    ('pos', 'float64', 2),
    ('gnd_pos', 'float64', 2),
    ('gnd_len', 'float64'),
    ('gnd_max_len', 'float64'),
    ('links', 'int64', nlinks),
    ('link_lens', 'float64', nlinks),
    ('link_max_lens', 'float64', nlinks)])

#@jit(nopython=True)
def shuffle(grid):
    for i in range(grid.shape[0]-2):
        j = int(random() * (grid.shape[0] - i))
        #grid[i], grid[j] = grid[j], grid[i]
        #tmp = np.copy(grid[i])
        #np.copyto(grid[i], grid[j])
        #np.copyto(grid[j], tmp)
        tmp = np.zeros(1, dtype=struct_dtype)
        print(tmp, '\n', grid[i], '\n', grid[j])
        np.copyto(tmp, grid[i])
        np.copyto(grid[i], grid[j])
        np.copyto(grid[j], tmp)
        #grid[i] = np.copy(grid[j])
        #grid[j] = np.copy(tmp)
        print(tmp, '\n', grid[i], '\n', grid[j])
        print('---')
        for g in grid:
            for li, lnk in enumerate(g['links']):
                if lnk == i:
                    g['links'][li] = j
                elif lnk == j:
                    g['links'][li] = i

POS=0
GND=1
GND_LEN=2
LEN=0
MAX_LEN=1
X=0
Y=1
#@jit(nopython=True)
def make_grid(x,y):
    gnd_max = 1.0
    gnd_min = 0.5
    len_mult_max = 1.5
    len_mult_min = 2.0
    # grid[n][0] = pos
    # grid[n][1] = gnd_pos
    # grid[n][2][0] = gnd_len
    # grid[n][2][1] = gnd_max_len
    grid = np.zeros((x*y, 3, 2), dtype=np.float64)
    links = np.zeros((x*y, nlinks), dtype=np.int64)
    link_lens = np.zeros((x*y, nlinks, 2), dtype=np.float64)
    for i in range(x*y):
        g = grid[i]
        g[POS] = [i%x, i//x]
        g[GND] = [i%x, i//x]
        g[GND_LEN][LEN] = 0.
        g[GND_LEN][MAX_LEN] = gnd_min + random() * (gnd_max - gnd_min)
        links[i] = [-1]*nlinks
        link_lens[i] = [[0.,0.]]*nlinks

    for i in range(x*y):
        targets = []
        if (i+1)%x != 0:
            targets.append(i+1)
        if i+x < x*y:
            targets.append(i+x)
        if (i+x+1)%x != 0 and i+x+1 < x*y:
            targets.append(i+x+1)
        
        for it, target in enumerate(targets):
            links[i][it] = target
            tdist2 = (grid[i][POS] - grid[target][POS])**2
            link_lens[i][it][LEN] = sqrt(tdist2[0] + tdist2[1])
            link_lens[i][it][MAX_LEN] = link_lens[i][it][LEN] * (len_mult_min + random() * (len_mult_max - len_mult_min))

    return grid,links,link_lens

#@jit(nopython=True)
def expand_grid(grid):
    for g in grid:
        g[GND][X] *= 1.1
        g[POS][X] *= 1.1

#@jit(nopython=True)
def iterate(grid, links, link_lens, t=0.1, constraint_iterations=1):
    lcs = 0.30 # link constraint strength
    gcs = 0.99 # ground constraint strength
    lef = 1.00 # link expansion factor
    gef = 1.00 # ground link expansion factor

    # One break is allowed per iteration
    # When one link breaks we:
    # Recacluate the constraints then
    # Recalculate the strains in the grid
    one_broke = True
    while one_broke:
        # Apply constraints
        for _ in range(constraint_iterations):
            for g, glinks, glink_lens in zip(grid, links, link_lens):
                for link, link_len in zip(glinks, glink_lens):
                    if link == -1: continue

                    d = g[POS] - grid[link][POS]
                    dmag = sqrt(d[0]**2 + d[1]**2)
                    if dmag > 0.:
                        mv = lcs * 0.5 * (dmag - link_len) * d / dmag
                        g[POS] -= mv
                        grid[link][POS] += mv

            for g in grid:
                # Ground linkage
                if g[GND_LEN][LEN] > -1: # not broken
                    d = g[POS] - g[GND]
                    dmag = sqrt(d[0]**2 + d[1]**2)
                    if dmag > 0.:
                        mv = gcs * (dmag - g[GND_LEN][LEN]) * d / dmag
                        g[POS] -= mv

        # calculate the strain and adjust link length for all the links
        # If we get through all the length adjustments without a break then we
        # can exit the procedure
        one_broke = False
        for g, glinks, glink_lens in zip(grid, links, link_lens):
            if one_broke: break
            for i, (link, link_len) in enumerate(zip(glinks, glink_lens)):
                if link == -1: continue

                d = g[POS] - grid[link][POS]
                dmag = sqrt(d[0]**2 + d[1]**2)
                link_len[LEN] += lef * (dmag - link_len[LEN])

                if link_len[LEN] > link_len[MAX_LEN]:
                    # remove link
                    glinks[i] = -1
                    one_broke = True
                    break

            if g[GND_LEN][LEN] > -1: # not broken
                d = g[POS] - g[GND]
                dmag = sqrt(d[0]**2 + d[1]**2)
                g[GND_LEN][LEN] += gef * (dmag - g[GND_LEN][LEN])

                if g[GND_LEN][LEN] > g[GND_LEN][MAX_LEN]:
                    # remove link
                    g[GND_LEN][LEN] = -1
                    one_broke = True
                    break

def draw(grid, links, frame):
    dwg = svg.Drawing('test%05d.svg'%frame)
    minx = miny =  9999999
    maxx = maxy = -9999999
    for g, glinks in zip(grid, links):
        minx = g[POS][0] if g[POS][0] < minx else minx
        maxx = g[POS][0] if g[POS][0] > maxx else maxx
        miny = g[POS][1] if g[POS][1] < miny else miny
        maxy = g[POS][1] if g[POS][1] > maxy else maxy

        for lnk in [k for k in glinks if k != -1]:
            line = svg.shapes.Line((g[POS][0], g[POS][1]),
                                    (grid[lnk][POS][0], grid[lnk][POS][1]),
                                    stroke='black',
                                    stroke_width=0.1)
            dwg.add(line)

    dwg.viewbox(minx=minx-2, miny=miny-2, 
                width=maxx-minx+2, height=maxy-miny+2)

    dwg.save()

def draw2(grid, links, frame):
    dwg = svg.Drawing('test%05d.svg'%frame)
    minx = miny =  9999999
    maxx = maxy = -9999999
    for g, glinks in zip(grid, links):
        minx = g[POS][0] if g[POS][0] < minx else minx
        maxx = g[POS][0] if g[POS][0] > maxx else maxx
        miny = g[POS][1] if g[POS][1] < miny else miny
        maxy = g[POS][1] if g[POS][1] > maxy else maxy

        for lnk in [k for k in glinks if k != -1]:
            p = (g[POS] + grid[lnk][POS]) / 2.
            d = g[POS] - grid[lnk][POS]
            d = d**2
            r = sqrt(np.sum(d)) / 2.
            line = svg.shapes.Circle((p[0], p[1]), r,
                                    fill='grey',
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

    save_fn = 'default_save.npy'
    load_fn = ''
    for opt,arg in opts:
        if opt in ('-s', '--save'):
            save_fn = arg
        elif opt in ('-l', '--load'):
            load_fn = arg

    if load_fn != '':
        arys = np.loadz(load_fn)
        grid, links, link_lens = arys['grid'], arys['links'], arys['link_lens']
        draw2(grid, links, -1)
    else:
        x,y = 10,10
        grid, links, link_lens = make_grid(x,y)
        #shuffle(grid)
        draw(grid, links, -1)
        for i in range(10):
            expand_grid(grid)
            iterate(grid, links, link_lens)
            draw(grid, links, i)
            np.savez(save_fn + '%05d.npy'%i, grid=grid, links=links, link_lens=link_lens)
            print('.', end='')
            sys.stdout.flush()
        print()

if __name__ == '__main__':
    main()
