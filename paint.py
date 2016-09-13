import numpy as np
from itertools import product
from random import random
from math import sqrt
import svgwrite as svg
from numba import jit, int64, float64
import sys
import time
import getopt

# Init size constant
nlinks = 3
x,y = 50,50

# Array reference constants
POS=0
GND=1
GND_LEN=2
LEN=0
MAX_LEN=1
X=0
Y=1

# Behaviour constants
gnd_min = 0.1
gnd_max = 1.0
len_mult_min = 1.1
len_mult_max = 1.5

lcs = 0.50 # link constraint strength
gcs = 0.99 # ground constraint strength
lef = 0.50 # link expansion factor
gef = 0.75 # ground link expansion factor

allowed_breaks = 5 # the number of link breaks before a re-relaxation is triggered
                   # higher number = higher performance but may impact accuracy if too high

@jit((float64[:,:,:], int64[:,:]))
def shuffle(grid, links):
    tmp_grid = np.zeros((1,3,2), dtype=np.float64)
    tmp_links = np.zeros((1,nlinks), dtype=np.int64)
    for i in range(grid.shape[0]-2):
        j = int(random() * (grid.shape[0] - i))

        np.copyto(tmp_grid[0], grid[i])
        np.copyto(grid[i], grid[j])
        np.copyto(grid[j], tmp_grid[0])

        np.copyto(tmp_links[0], links[i])
        np.copyto(links[i], links[j])
        np.copyto(links[j], tmp_links[0])

        for glink in links:
            for li, link in enumerate(glink):
                if link == i:
                    glink[li] = j
                elif link == j:
                    glink[li] = i

#@jit(nopython=True)
def make_grid(x,y):
    grid = np.zeros((x*y, 3, 2), dtype=np.float64)
    links = np.zeros((x*y, nlinks), dtype=np.int64)
    link_lens = np.zeros((x*y, nlinks, 2), dtype=np.float64)
    for i in range(x*y):
        g = grid[i]
        g[POS] = [i%x - x/2., i//x - y/2.]
        g[GND] = [i%x - x/2., i//x - y/2.]
        g[GND_LEN][LEN] = 0. if random() > 0.5 else -1
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

np.seterr(all='raise')
@jit((float64[:,:,:], int64[:,:], float64[:,:], int64))
def relax(grid, links, link_lens, iterations):

    # accumulate all the vectors and their weights into:
    accum = np.zeros((grid.shape[0], 2), dtype=np.float64)
    weight = np.zeros(grid.shape[0], dtype=np.float64)

    for _ in range(iterations):
        accum.fill(0.)
        weight.fill(0.)
        for i in range(grid.shape[0]):
            for link, link_len in zip(links[i], link_lens[i]):
                if link == -1: continue

                d = grid[i][POS] - grid[link][POS]

                dmag = sqrt(d[0]**2 + d[1]**2)
                if dmag > 0.:
                    mv = lcs * 0.5 * (dmag - link_len[LEN]) * d / dmag
                    accum[i] -= mv
                    accum[link] += mv
                    weight[i] += lcs
                    weight[link] += lcs

            # Ground linkage
            if grid[i][GND_LEN][LEN] > -1: # not broken
                d = grid[i][POS] - grid[i][GND]
                dmag = sqrt(d[0]**2 + d[1]**2)
                if dmag > 0.:
                    mv = gcs * (dmag - grid[i][GND_LEN][LEN]) * d / dmag
                    accum[i] -= mv
                    weight[i] += gcs

        # finally unweight and apply the vectors
        for i in range(grid.shape[0]):
            if weight[i] > 0.:
                grid[i][POS] += accum[i] / weight[i]


@jit((float64[:,:,:], int64[:,:], float64[:,:], int64))
def iterate(grid, links, link_lens, constraint_iterations=5):
    breaks = 1
    # A number of breaks is allowed per iteration
    # When one link breaks we:
    # Recacluate the constraints then
    # Recalculate the strains in the grid
    while breaks > 0:
        # Apply constraints
        relax(grid, links, link_lens, constraint_iterations)

        # calculate the strain and adjust link length for all the links
        # If we get through all the length adjustments without a break then we
        # can exit the procedure
        breaks = 0
        for g, glinks, glink_lens in zip(grid, links, link_lens):
            if breaks >= allowed_breaks: break

            for i, (link, link_len) in enumerate(zip(glinks, glink_lens)):
                if link == -1: continue

                d = g[POS] - grid[link][POS]
                dmag = sqrt(d[0]**2 + d[1]**2)
                link_len[LEN] += lef * (dmag - link_len[LEN])

                if link_len[LEN] > link_len[MAX_LEN]:
                    # remove link
                    glinks[i] = -1
                    breaks += 1
                    if breaks >= allowed_breaks: break

            if g[GND_LEN][LEN] > -1: # not broken
                d = g[POS] - g[GND]
                dmag = sqrt(d[0]**2 + d[1]**2)
                g[GND_LEN][LEN] += gef * (dmag - g[GND_LEN][LEN])

                if g[GND_LEN][LEN] > g[GND_LEN][MAX_LEN]:
                    # remove link
                    g[GND_LEN][LEN] = -1
                    breaks += 1

def draw(grid, links, filename, frame, draw_circles=False):
    dwg = svg.Drawing('%s%05d.svg'%(filename, frame))
    minx = miny =  9999999
    maxx = maxy = -9999999
    for g, glinks in zip(grid, links):
        minx = g[POS][0] if g[POS][0] < minx else minx
        maxx = g[POS][0] if g[POS][0] > maxx else maxx
        miny = g[POS][1] if g[POS][1] < miny else miny
        maxy = g[POS][1] if g[POS][1] > maxy else maxy

        if draw_circles:
            circ = svg.shapes.Circle((g[POS][0], g[POS][1]), 0.1,
                                    fill='blue',
                                    stroke='none',
                                    stroke_width=0.1)
            dwg.add(circ)

        for lnk in [k for k in glinks if k != -1]:
            line = svg.shapes.Line((g[POS][0], g[POS][1]),
                                    (grid[lnk][POS][0], grid[lnk][POS][1]),
                                    stroke='black',
                                    stroke_width=0.1)
            dwg.add(line)

    dwg.viewbox(minx=minx-2, miny=miny-2, 
                width=maxx-minx+4, height=maxy-miny+4)

    dwg.save()

def draw2(grid, links, filename, frame):
    dwg = svg.Drawing('%s%05d.svg'%(filename, frame))
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
                width=maxx-minx+4, height=maxy-miny+4)

    dwg.save()

def print_grid(grid, links, link_lens):
    for i, (g,li,ll) in enumerate(zip(grid, links, link_lens)):
        print(i, g)
        print(i, li)
        print(i, ll)

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
        draw2(grid, links, load_fn, -1)
    else:
        t0 = time.time()
        grid, links, link_lens = make_grid(x,y)
        t1 = time.time()
        print("Make Grid Elapsed:", t1-t0)

        t0 = time.time()
        shuffle(grid, links)
        t1 = time.time()
        print("Shuffle Elapsed:", t1-t0)

        t0 = time.time()
        draw(grid, links, save_fn, -1)
        t1 = time.time()
        print("Draw Elapsed:", t1-t0)

        for i in range(10):
            expand_grid(grid)
            iterate(grid, links, link_lens)
            draw(grid, links, save_fn, i)
            np.savez(save_fn + '%05d'%i, grid=grid, links=links, link_lens=link_lens)
            print('.', end='')
            sys.stdout.flush()
        print()

if __name__ == '__main__':
    t0 = time.time()
    main()
    t1 = time.time()

    print("Elapsed:", t1-t0)
