import numpy as np
import scipy.spatial as spat
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
x,y = 20,20

# Array reference constants
POS=0
GND=1
GND_LEN=2
LEN=0
MAX_LEN=1
X=0
Y=1

# Behaviour constants
# Gnd values are the absolute lengths 
gnd_min = 0.1
gnd_max = 2.0
# These are the multipliers that are applied to the starting length to calc the
# max allowed len.
len_mult_min = 1.1
len_mult_max = 1.5

lcs = 1.00 # link constraint strength
gcs = 1.00 # ground constraint strength

allowed_breaks = 5 # the number of link breaks before a re-relaxation is triggered
                   # higher number = higher performance but may impact accuracy if too high

# Random number generator choice
rng = lambda mini, maxi: np.random.normal((maxi+mini)/2., (maxi-mini)/6.)
#rng = lambda maxi, mini: mini + random() * (maxi-mini)

#gnd_link = lambda i,x,y: 0 # All attached 
#gnd_link = lambda i,x,y: 0. if random() > 0.75 else -1 # Some attached
#gnd_link = lambda i,x,_: 0 if i % x == 0 or i % x == x-1 else -1 # Attach vertical edges only
# Vertical edges always other points sometimes
gnd_link = lambda i,x,_: 0 if i % x == 0 or i % x == x-1 or random() > 0.75 else -1 

max_relax_iterations = 500 # Relaxation has an automatic convergence detector so this is the absolute
                           # max iterations allowed.
                      

#@jit((float64[:,:,:], int64[:,:]))
def shuffle(grid, links, link_lens):
    tmp_grid = np.zeros((1,3,2), dtype=np.float64)
    tmp_links = np.zeros((1,nlinks), dtype=np.int64)
    tmp_lens = np.zeros((1,nlinks,2), dtype=np.float64)
    for i in range(grid.shape[0]-2):
        j = int(random() * (grid.shape[0] - i))

        np.copyto(tmp_grid[0], grid[i])
        np.copyto(grid[i], grid[j])
        np.copyto(grid[j], tmp_grid[0])

        np.copyto(tmp_links[0], links[i])
        np.copyto(links[i], links[j])
        np.copyto(links[j], tmp_links[0])

        np.copyto(tmp_lens[0], link_lens[i])
        np.copyto(link_lens[i], link_lens[j])
        np.copyto(link_lens[j], tmp_lens[0])

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
        g[GND_LEN][LEN] = gnd_link(i,x,y)
        g[GND_LEN][MAX_LEN] = rng(gnd_min, gnd_max)
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
            link_lens[i][it][MAX_LEN] = link_lens[i][it][LEN] * rng(len_mult_min, len_mult_max)

    return grid,links,link_lens

@jit((int64, int64, int64, int64, float64))
def random_grid(x,y, npoints, max_links=100, max_link_dist=1.5):
    grid = np.zeros((npoints, 3, 2), dtype=np.float64)
    links = np.zeros((npoints, max_links), dtype=np.int64)
    links.fill(-1)
    link_lens = np.zeros((npoints, max_links, 2), dtype=np.float64)
    for i in range(npoints):
        grid[i][POS][X] = x * random() - x/2.
        grid[i][POS][Y] = y * random() - y/2.
        grid[i][GND][X] = grid[i][POS][X]
        grid[i][GND][Y] = grid[i][POS][Y]
        grid[i][GND_LEN][LEN] = 0. #gnd_link(i,x,y)
        grid[i][GND_LEN][MAX_LEN] = rng(gnd_min, gnd_max)
        
    for i in range(npoints):
        nlinks = 0
        for j in range(npoints):
            d = grid[i][POS] - grid[j][POS]
            dist = sqrt(d[0]**2 + d[1]**2)
            if dist < max_link_dist:
                if nlinks >= max_links:
                    print('Ran out of links')
                else:
                    links[i][nlinks] = j
                    link_lens[i][nlinks][LEN] = dist
                    link_lens[i][nlinks][MAX_LEN] = link_lens[i][nlinks][LEN] * rng(len_mult_min, len_mult_max)
                    nlinks += 1
    return grid,links,link_lens

#@jit((int64, int64, int64, int64, float64))
def random_triangle_grid(x,y, npoints, max_links=100, max_link_dist=1.5):
    grid = np.zeros((npoints, 3, 2), dtype=np.float64)
    links = np.zeros((npoints, max_links), dtype=np.int64)
    links.fill(-1)
    link_lens = np.zeros((npoints, max_links, 2), dtype=np.float64)
    for i in range(npoints):
        grid[i][POS][X] = x * random() - x/2.
        grid[i][POS][Y] = y * random() - y/2.
        grid[i][GND][X] = grid[i][POS][X]
        grid[i][GND][Y] = grid[i][POS][Y]
        grid[i][GND_LEN][LEN] = 0. #gnd_link(i,x,y)
        grid[i][GND_LEN][MAX_LEN] = rng(gnd_min, gnd_max)

    # could replace with a reduce call but I didn't 
    tri_links = []
    for tri in spat.Delaunay(grid[:,POS]).simplices:
        tri_links += [[tri[0], tri[1]], [tri[1], tri[2]], [tri[2], tri[0]]]
    # order the links within each pair then deduplicate and sort links
    tri_links = [[min(t[0],t[1]), max(t[0],t[1])] for t in tri_links]
    tri_links = list(set(tuple(p) for p in tri_links))
    tri_links.sort(key=lambda x: x[0])
    # Remove all the outer segments (de-convex hull the mesh) because the line
    # segments in the convex hull tend to be much larger make the result look
    # ugly
    for seg in spat.ConvexHull(grid[:,POS]).simplices:
        print(seg)
        seg_sorted = (min(seg[0], seg[1]), max(seg[0],seg[1]))
        if seg_sorted in tri_links:
            print('found')
            tri_links.remove(seg_sorted)

    nlinks = np.zeros(npoints, dtype=np.int64)
    for i,j in tri_links:
        links[i][nlinks[i]] = j
        if nlinks[i] >= max_links:
            print('Ran out of links')
        else:
            links[i][nlinks[i]] = j
            d = grid[i][POS] - grid[j][POS]
            dist = sqrt(d[0]**2 + d[1]**2)
            link_lens[i][nlinks[i]][LEN] = dist
            link_lens[i][nlinks[i]][MAX_LEN] = link_lens[i][nlinks[i]][LEN] * rng(len_mult_min, len_mult_max)
            nlinks[i] += 1
        
    return grid,links,link_lens

#@jit(nopython=True)
def expand_grid(grid):
    for g in grid:
        g[GND][X] *= 1.1

np.seterr(all='raise')
#@jit((float64[:,:,:], int64[:,:], float64[:,:]))
def relax(grid, links, link_lens):
    # accumulate all the vectors and their weights into:
    accum = np.zeros((grid.shape[0], 2), dtype=np.float64)
    weight = np.zeros(grid.shape[0], dtype=np.float64)

    last_err = 99999999.
    err = last_err - 1
    for _ in range(max_relax_iterations):
        # Stop if the error is relatively stable.
        if last_err - err < 0.1: break
        last_err, err = err, 0.

        accum.fill(0.)
        weight.fill(0.)
        for i in range(grid.shape[0]):
            for link, link_len in zip(links[i], link_lens[i]):
                if link == -1: continue

                d = grid[i][POS] - grid[link][POS]

                dmag = sqrt(d[0]**2 + d[1]**2)
                err += dmag - grid[i][GND_LEN][LEN]
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
                err += dmag - grid[i][GND_LEN][LEN]
                if dmag > 0.:
                    mv = gcs * (dmag - grid[i][GND_LEN][LEN]) * d / dmag
                    accum[i] -= mv
                    weight[i] += gcs

        # finally unweight and apply the vectors
        for i in range(grid.shape[0]):
            if weight[i] > 0.:
                grid[i][POS] += accum[i] / weight[i]


#@jit((float64[:,:,:], int64[:,:], float64[:,:]))
def iterate(grid, links, link_lens):
    breaks = 1
    # A number of breaks is allowed per iteration
    # When one link breaks we:
    # Recacluate the constraints then
    # Recalculate the strains in the grid
    while breaks > 0:
        # Apply constraints
        relax(grid, links, link_lens)

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

                if dmag > link_len[MAX_LEN]:
                    # remove link
                    glinks[i] = -1
                    breaks += 1
                    if breaks >= allowed_breaks: break

            if g[GND_LEN][LEN] > -1: # not broken
                d = g[POS] - g[GND]
                dmag = sqrt(d[0]**2 + d[1]**2)

                if dmag > g[GND_LEN][MAX_LEN]:
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
                                    fill='black',
                                    stroke='none',
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
        opts, args = getopt.getopt(sys.argv[1:], "s:l:r", ('save=', 'load='))
    except getopt.GetoptError:
        print('Arg fail')

    save_fn = 'default_save.npy'
    load_fn = ''
    rand_grid = False
    for opt,arg in opts:
        if opt in ('-s', '--save'):
            save_fn = arg
        elif opt in ('-l', '--load'):
            load_fn = arg
        elif opt in ('-r', '--random'):
            rand_grid = True

    if load_fn != '':
        arys = np.load(load_fn)
        grid, links, link_lens = arys['grid'], arys['links'], arys['link_lens']
        draw2(grid, links, load_fn, -1)
    else:
        if rand_grid:
            t0 = time.time()
            grid, links, link_lens = random_triangle_grid(x,y, 800)
            t1 = time.time()
            print("Make Random Grid Elapsed:", t1-t0)
        else: 
            t0 = time.time()
            grid, links, link_lens = make_grid(x,y)
            t1 = time.time()
            print("Make Grid Elapsed:", t1-t0)

            t0 = time.time()
            shuffle(grid, links, link_lens)
            t1 = time.time()
            print("Shuffle Elapsed:", t1-t0)

        t0 = time.time()
        draw(grid, links, save_fn, -1)
        #relax(grid, links, link_lens, 1)
        #draw(grid, links, save_fn, -2)
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
