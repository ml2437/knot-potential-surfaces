#Copyright 2020 Max Lipton
# Email: ml2437@cornell.edu
# Twitter: @Maxematician
# Website: https://e.math.cornell.edu/people/ml2437/

#Setup and basic definitions

import matplotlib as mp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
from time import time
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mayavi import mlab
import concurrent.futures

import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
import chart_studio.plotly as py
import plotly.io as pio



fig = plt.figure();
ax = fig.gca(projection='3d');

#Order of accuracy for Gaussian quadrature
quadorder = 1000;


#Discretization of the domain interval for knot parametrizations
t = np.linspace(0,2*np.pi,1000)

#Knot parametrizations:
#The "numerators" compute ds^2 for each of the parametrizations

#Unknot
def unknotnum(t):
    return 1
def unknotx(t):
    return np.cos(t)
def unknoty(t):
    return np.sin(t)
def unknotz(t):
    return 0


#Trefoil
def trefnum(t):
    return 8 * np.cos(3*t) + 4.5 * np.cos(6*t) + 21.5
def trefx(t):
    return np.sin(t) + 2 * np.sin(2*t)
def trefy(t):
    return np.cos(t) - 2 * np.cos(2*t)
def trefz(t):
    return -1 * np.sin(3*t)


#Figure 8
def fig8num(t):
    return 40 + 36 * np.cos(2*t) + 5 * (np.cos(2*t) ** 2) + 16 * (np.cos(4 * t) ** 2)
def fig8x(t):
    return (2 + np.cos(2*t)) * np.cos(3*t)
def fig8y(t):
    return (2 + np.cos(2*t)) * np.sin(3*t)
def fig8z(t):
    return np.sin(4*t)

#(3,1) Torus Knot
def torus31num(t):
    return 9 * np.cos(t) ** 2 + 36 * np.cos(t) + 37
def torus31x(t):
    return (np.cos(t) + 2) * np.cos(3 * t)
def torus31y(t):
    return (np.cos(t) + 2) * np.sin(3 * t)
def torus31z(t):
    return -1 * np.sin(t)

#Parametrization of the (5,1) torus knot
def torus51x(t):
    return (np.cos(t) + 2) * np.cos(5 * t)
def torus51y(t):
    return (np.cos(t) + 2) * np.sin(5 * t)
def torus51z(t):
    return -1 * np.sin(t)
def torus51num(t):
    return 25 * np.cos(t) ** 2 + 100 * np.cos(t) + 101

#Cinquefoil knot AKA (5,2) torus knot
def cinquenum(t):
    return 0.25 * (24 * np.cos(5*t) + 2 * np.cos(10 * t) + 63)
def cinquex(t):
    return  0.5 * (3 + np.cos(5*t)) * np.cos(2*t)
def cinquey(t):
    return 0.5 * (3 + np.cos(5*t)) * np.sin(2*t)
def cinquez(t):
    return 0.5 * np.sin(5*t)

#Parametrization for the 3-twist knot AKA 5_2
def threetwistnum(t):
    return (49 * (np.sin(7 * t) ** 2) + 4 * (np.sin(2 * t + 0.2) ** 2) + 9 * (np.sin(3 * t + 0.7) ** 2))
def threetwistx(t):
    return  2 * np.cos(2*t + 0.2)
def threetwisty(t):
    return 2 * np.cos(3*t + 0.7)
def threetwistz(t):
    return np.cos(7*t)

#Parametrization for the granny knot (the connected sum of two identical trefoils)
def grannynum(t):
    return 1.59375 * np.cos(2*t) + 8.4375 * np.cos(4 * t) + 2 * np.cos(8 * t) - 4.5 * np.cos(10 * t) + 2.53125 * np.cos(12 * t) + 20.25
def grannyx(t):
    return 0.5 * np.cos(t) - 1.25 * np.cos(3 * t)
def grannyy(t):
    return 1.75 * np.sin(t) + 1.25 * np.sin(3 * t)
def grannyz(t):
    return 0.5 * np.sin(4 * t) - 0.375 * np.sin(6 * t)

#Parametrization for the square knot (the connected sum of two oppositely oriented trefoils)
def squarenum(t):
    return 6.09375 * np.cos(2 * t) + 8.4375 * np.cos(4 * t) + 12.5 * np.cos(10 * t) + 28.2188
def squarex(t):
    return 0.5 * np.cos(t) - 1.25 * np.cos(3 * t)
def squarey(t):
    return 1.75 * np.sin(t) + 1.25 * np.sin(3 * t)
def squarez(t):
    return np.sin(5 * t)

#Parametrization of the endless knot (AKA the 7_4 knot)
def endlessnum(t):
    return 2 * np.cos(4 * t) - 4.5 * np.cos(6 * t) + 24.5 * np.cos(14 * t) + 31
def endlessx(t):
    return np.cos(3 * t)
def endlessy(t):
    return np.sin(2 * t)
def endlessz(t):
    return np.sin(7 * t)



#Uses numpy integration to compute the potential at point (a,b,c) with respect to the given knot type
#Order of the Gaussian quadratures is fixed at 1000
#Use "unknot," "trefoil," "fig8," etc. to specify the knot type
def potential(a,b,c,knottype):
    if knottype == "unknot":
        ans, _ = integrate.fixed_quad(lambda t: 1/(((a - 1 * np.cos(t)) ** 2 + (b - 1 * np.sin(t)) ** 2 + c ** 2) ** (1/2)),0, 2*np.pi, n = quadorder)
        return ans
    elif knottype == "trefoil":
        ans, _ = integrate.fixed_quad(lambda t: np.sqrt(trefnum(t))/(((a - trefx(t)) ** 2 + (b - trefy(t)) ** 2 + (c - trefz(t)) ** 2) ** (1/2)),0, 2*np.pi, n = quadorder)
        return ans
    elif knottype == "fig8":
        ans, _ = integrate.fixed_quad(lambda t: np.sqrt(fig8num(t))/(((a - fig8x(t)) ** 2 + (b - fig8y(t)) ** 2 + (c - fig8z(t)) ** 2) ** (1/2)),0, 2*np.pi, n = quadorder)
        return ans
    elif knottype == "torus31":
        ans, _ = integrate.fixed_quad(lambda t: np.sqrt(torus31num(t))/(((a - torus31x(t)) ** 2 + (b - torus31y(t)) ** 2 + (c - torus31z(t)) ** 2) ** (1/2)),0, 2*np.pi, n = quadorder)
        return ans
    elif knottype == "torus51":
        ans, _ = integrate.fixed_quad(lambda t: np.sqrt(torus51num(t))/(((a - torus51x(t)) ** 2 + (b - torus51y(t)) ** 2 + (c - torus51z(t)) ** 2) ** (1/2)),0, 2*np.pi, n = quadorder)
        return ans
    elif knottype == "cinque":
        ans, _ = integrate.fixed_quad(lambda t: np.sqrt(cinquenum(t))/(((a - cinquex(t)) ** 2 + (b - cinquey(t)) ** 2 + (c - cinquez(t)) ** 2) ** (1/2)),0, 2*np.pi, n = quadorder)
        return ans
    elif knottype == "3twist":
        ans, _ = integrate.fixed_quad(lambda t: np.sqrt(threetwistnum(t))/(((a - threetwistx(t)) ** 2 + (b - threetwisty(t)) ** 2 + (c - threetwistz(t)) ** 2) ** (1/2)),0, 2*np.pi, n = quadorder)
        return ans
    elif knottype == "granny":
        ans, _ = integrate.fixed_quad(lambda t: np.sqrt(grannynum(t))/(((a - grannyx(t)) ** 2 + (b - grannyy(t)) ** 2 + (c - grannyz(t)) ** 2) ** (1/2)),0, 2*np.pi, n = quadorder)
        return ans
    elif knottype == "square":
        ans, _ = integrate.fixed_quad(lambda t: np.sqrt(squarenum(t))/(((a - squarex(t)) ** 2 + (b - squarey(t)) ** 2 + (c - squarez(t)) ** 2) ** (1/2)),0, 2*np.pi, n = quadorder)
        return ans    
    elif knottype == "endless":
        ans, _ = integrate.fixed_quad(lambda t: np.sqrt(endlessnum(t))/(((a - endlessx(t)) ** 2 + (b - endlessy(t)) ** 2 + (c - endlessz(t)) ** 2) ** (1/2)),0, 2*np.pi, n = quadorder)
        return ans    
    else:
        print("Invalid knot type");
        return NaN
    
potential2 = np.vectorize(potential)

#Data for the knot plots
    
punknotx = 1 * np.cos(t)
punknoty = 1 * np.sin(t)
punknotz = 0 * t

ptrefx = trefx(t)
ptrefy = trefy(t)
ptrefz = trefz(t)

pfig8x = fig8x(t)
pfig8y = fig8y(t)
pfig8z = fig8z(t)

ptorus31x = torus31x(t)
ptorus31y = torus31y(t)
ptorus31z = torus31z(t)

ptorus51x = torus51x(t)
ptorus51y = torus51y(t)
ptorus51z = torus51z(t)

pcinquex = cinquex(t)
pcinquey = cinquey(t)
pcinquez = cinquez(t)

p3twistx = threetwistx(t)
p3twisty = threetwisty(t)
p3twistz = threetwistz(t)

pgrannyx = grannyx(t)
pgrannyy = grannyy(t)
pgrannyz = grannyz(t)

psquarex = squarex(t)
psquarey = squarey(t)
psquarez = squarez(t)

pendlessx = endlessx(t)
pendlessy = endlessy(t)
pendlessz = endlessz(t)

#Returns the knot coordinates, given the knot type
def getpts(knottype):
    if knottype == "unknot":
        return punknotx, punknoty, punknotz
    elif knottype == "trefoil":
        return ptrefx, ptrefy, ptrefz
    elif knottype == "fig8":
        return pfig8x, pfig8y, pfig8z
    elif knottype == "torus31":
        return ptorus31x, ptorus31y, ptorus31z
    elif knottype == "torus51":
        return ptorus51x, ptorus51y, ptorus51z
    elif knottype == "cinque":
        return pcinquex, pcinquey, pcinquez
    elif knottype == "3twist":
        return p3twistx, p3twisty, p3twistz
    elif knottype == "granny":
        return pgrannyx, pgrannyy, pgrannyz
    elif knottype == "square":
        return psquarex, psquarey, psquarez
    elif knottype == "endless":
        return pendlessx, pendlessy, pendlessz
    else:
        return None, None, None

#Plot a level potential surface with the quadrature method of evaluation

globalc = potential(0,0,0,'unknot') + 0.5

def setc(knottype):
    global globalc
    globalc = potential(0,0,0,knottype) + 0.5
    #print('Setting global c to: ', globalc)
    return

#Makes the domain with mesh fineness of order n
def makedomain(n):
    return 4 * np.mgrid[-1:1:n*1j, -1:1:n*1j, -1:1:n*1j]

def showsurface(verts, faces, colormap = 'Blues'):
    mlab.triangular_mesh(verts[:,0], verts[:,1], verts[:,2], faces, colormap=colormap)
    mlab.show()
    return None


def generateverts(knottype = 'unknot', n = 50, c = globalc):
    domx, domy, domz = makedomain(n)
    t0 = time()

    vol = potential2(domx, domy, domz, knottype) 
    verts, faces, _, _ = measure.marching_cubes(vol, c, spacing=(1,1,1))

    t1 = time()
    print("Level surface of Phi(x) = ", c)
    print("Time taken: ", (t1 - t0), "seconds")
    return verts, faces
    

def makesurface(knottype = 'unknot', n = 50, c = None, colormap = 'Blues'):
    setc(knottype)
    if c is None:
        c = globalc
    verts, faces = generateverts(knottype, n, c)
    showsurface(verts, faces, colormap = colormap)
    return verts, faces
    

#Display using matplotlib, which is slower than Mayavi
#Comment out showsurface and uncomment showsurface2 in the makesurface function to use matplotlib

def showsurface2(verts, faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:,2], cmap='Spectral', lw=1)
    plt.show()


#This code computes the critical points of the potential
#Compute the electric field, using Gaussian quadrature

def efield(a,b,c,knottype):
    if knottype == "unknot":
        Ex, _ = integrate.fixed_quad(lambda t: (a - np.cos(t))/(((a - np.cos(t)) ** 2 + (b - np.sin(t)) ** 2 + c ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ey, _ = integrate.fixed_quad(lambda t: (b - np.sin(t))/(((a - np.cos(t)) ** 2 + (b - np.sin(t)) ** 2 + c ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ez, _ = integrate.fixed_quad(lambda t: c /(((a - np.cos(t)) ** 2 + (b - np.sin(t)) ** 2 + c ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)        
    elif knottype == "trefoil":
        Ex, _ = integrate.fixed_quad(lambda t: (a - trefx(t)) * np.sqrt(trefnum(t))/(((a - trefx(t)) ** 2 + (b - trefy(t)) ** 2 + (c - trefz(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ey, _ = integrate.fixed_quad(lambda t: (b - trefy(t)) * np.sqrt(trefnum(t))/(((a - trefx(t)) ** 2 + (b - trefy(t)) ** 2 + (c - trefz(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ez, _ = integrate.fixed_quad(lambda t: (c - trefz(t)) * np.sqrt(trefnum(t))/(((a - trefx(t)) ** 2 + (b - trefy(t)) ** 2 + (c - trefz(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
    elif knottype == "fig8":
        Ex, _ = integrate.fixed_quad(lambda t: (a - fig8x(t)) * np.sqrt(fig8num(t))/(((a - fig8x(t)) ** 2 + (b - fig8y(t)) ** 2 + (c - fig8z(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ey, _ = integrate.fixed_quad(lambda t: (b - fig8y(t)) * np.sqrt(fig8num(t))/(((a - fig8x(t)) ** 2 + (b - fig8y(t)) ** 2 + (c - fig8z(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ez, _ = integrate.fixed_quad(lambda t: (c - fig8z(t)) * np.sqrt(fig8num(t))/(((a - fig8x(t)) ** 2 + (b - fig8y(t)) ** 2 + (c - fig8z(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)    
    elif knottype == "torus31":
        Ex, _ = integrate.fixed_quad(lambda t: (a - torus31x(t)) * np.sqrt(torus31num(t))/(((a - torus31x(t)) ** 2 + (b - torus31y(t)) ** 2 + (c - torus31z(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ey, _ = integrate.fixed_quad(lambda t: (b - torus31y(t)) * np.sqrt(torus31num(t))/(((a - torus31x(t)) ** 2 + (b - torus31y(t)) ** 2 + (c - torus31z(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ez, _ = integrate.fixed_quad(lambda t: (c - torus31z(t)) * np.sqrt(torus31num(t))/(((a - torus31x(t)) ** 2 + (b - torus31y(t)) ** 2 + (c - torus31z(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)    
    elif knottype == "torus51":
        Ex, _ = integrate.fixed_quad(lambda t: (a - torus51x(t)) * np.sqrt(torus51num(t))/(((a - torus51x(t)) ** 2 + (b - torus51y(t)) ** 2 + (c - torus51z(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ey, _ = integrate.fixed_quad(lambda t: (b - torus51y(t)) * np.sqrt(torus51num(t))/(((a - torus51x(t)) ** 2 + (b - torus51y(t)) ** 2 + (c - torus51z(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ez, _ = integrate.fixed_quad(lambda t: (c - torus51z(t)) * np.sqrt(torus51num(t))/(((a - torus51x(t)) ** 2 + (b - torus51y(t)) ** 2 + (c - torus51z(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)    
    elif knottype == "cinque":
        Ex, _ = integrate.fixed_quad(lambda t: (a - cinquex(t)) * np.sqrt(cinquenum(t))/(((a - cinquex(t)) ** 2 + (b - cinquey(t)) ** 2 + (c - cinquez(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ey, _ = integrate.fixed_quad(lambda t: (b - cinquey(t)) * np.sqrt(cinquenum(t))/(((a - cinquex(t)) ** 2 + (b - cinquey(t)) ** 2 + (c - cinquez(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ez, _ = integrate.fixed_quad(lambda t: (c - cinquez(t)) * np.sqrt(cinquenum(t))/(((a - cinquex(t)) ** 2 + (b - cinquey(t)) ** 2 + (c - cinquez(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)    
    elif knottype == "3twist":
        Ex, _ = integrate.fixed_quad(lambda t: (a - threetwistx(t)) * np.sqrt(threetwistnum(t))/(((a - threetwistx(t)) ** 2 + (b - threetwisty(t)) ** 2 + (c - threetwistz(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ey, _ = integrate.fixed_quad(lambda t: (b - threetwisty(t)) * np.sqrt(threetwistnum(t))/(((a - threetwistx(t)) ** 2 + (b - threetwisty(t)) ** 2 + (c - threetwistz(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ez, _ = integrate.fixed_quad(lambda t: (c - threetwistz(t)) * np.sqrt(threetwistnum(t))/(((a - threetwistx(t)) ** 2 + (b - threetwisty(t)) ** 2 + (c - threetwistz(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)    
    elif knottype == "granny":
        Ex, _ = integrate.fixed_quad(lambda t: (a - grannyx(t)) * np.sqrt(grannynum(t))/(((a - grannyx(t)) ** 2 + (b - grannyy(t)) ** 2 + (c - grannyz(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ey, _ = integrate.fixed_quad(lambda t: (b - grannyy(t)) * np.sqrt(grannynum(t))/(((a - grannyx(t)) ** 2 + (b - grannyy(t)) ** 2 + (c - grannyz(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ez, _ = integrate.fixed_quad(lambda t: (c - grannyz(t)) * np.sqrt(grannynum(t))/(((a - grannyx(t)) ** 2 + (b - grannyy(t)) ** 2 + (c - grannyz(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)    
    elif knottype == "square":
        Ex, _ = integrate.fixed_quad(lambda t: (a - squarex(t)) * np.sqrt(squarenum(t))/(((a - squarex(t)) ** 2 + (b - squarey(t)) ** 2 + (c - squarez(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ey, _ = integrate.fixed_quad(lambda t: (b - squarey(t)) * np.sqrt(squarenum(t))/(((a - squarex(t)) ** 2 + (b - squarey(t)) ** 2 + (c - squarez(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ez, _ = integrate.fixed_quad(lambda t: (c - squarez(t)) * np.sqrt(squarenum(t))/(((a - squarex(t)) ** 2 + (b - squarey(t)) ** 2 + (c - squarez(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)    
    elif knottype == "endless":
        Ex, _ = integrate.fixed_quad(lambda t: (a - endlessx(t)) * np.sqrt(endlessnum(t))/(((a - endlessx(t)) ** 2 + (b - endlessy(t)) ** 2 + (c - endlessz(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ey, _ = integrate.fixed_quad(lambda t: (b - endlessy(t)) * np.sqrt(endlessnum(t))/(((a - endlessx(t)) ** 2 + (b - endlessy(t)) ** 2 + (c - endlessz(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)
        Ez, _ = integrate.fixed_quad(lambda t: (c - endlessz(t)) * np.sqrt(endlessnum(t))/(((a - endlessx(t)) ** 2 + (b - endlessy(t)) ** 2 + (c - endlessz(t)) ** 2) ** (1.5)),0, 2*np.pi, n = quadorder)     
    else:
        print("Invalid knot type")
        Ex = None
        Ey = None
        Ez = None
    return np.array([Ex,Ey,Ez])


#Computes various partial derivatves in the Jacobian matrix for the electric field
#Note that we only have to compute 5 out of the 9 second order derivatives because mixed partials agree, and
#the potential is harmonic.

def JEentries(a,b,c,knottype):
    if knottype == "unknot":
        phixx, _ = integrate.fixed_quad(lambda t: (2 * (a - unknotx(t)) ** 2 - (b - unknoty(t)) ** 2 - (c - unknotz(t)) ** 2 )* np.sqrt(unknotnum(t))/(((a - unknotx(t)) ** 2 + (b - unknoty(t)) ** 2 + (c - unknotz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixy, _ = integrate.fixed_quad(lambda t: 3 * (unknotx(t) - a) * (unknoty(t) - b) * np.sqrt(unknotnum(t))/(((a - unknotx(t)) ** 2 + (b - unknoty(t)) ** 2 + (c - unknotz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixz, _ = integrate.fixed_quad(lambda t: 3 * (unknotx(t) - a) * (unknotz(t) - c) * np.sqrt(unknotnum(t))/(((a - unknotx(t)) ** 2 + (b - unknoty(t)) ** 2 + (c - unknotz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyy, _ = integrate.fixed_quad(lambda t: (-1 * (a - unknotx(t)) ** 2 + 2 * (b - unknoty(t)) ** 2 - (c - unknotz(t)) ** 2 )* np.sqrt(unknotnum(t))/(((a - unknotx(t)) ** 2 + (b - unknoty(t)) ** 2 + (c - unknotz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyz, _ = integrate.fixed_quad(lambda t: 3 * (unknoty(t) - b) * (unknotz(t) - c) * np.sqrt(unknotnum(t))/(((a - unknotx(t)) ** 2 + (b - unknoty(t)) ** 2 + (c - unknotz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
    elif knottype == "trefoil":
        phixx, _ = integrate.fixed_quad(lambda t: (2 * (a - trefx(t)) ** 2 - (b - trefy(t)) ** 2 - (c - trefz(t)) ** 2 )* np.sqrt(trefnum(t))/(((a - trefx(t)) ** 2 + (b - trefy(t)) ** 2 + (c - trefz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixy, _ = integrate.fixed_quad(lambda t: 3 * (trefx(t) - a) * (trefy(t) - b) * np.sqrt(trefnum(t))/(((a - trefx(t)) ** 2 + (b - trefy(t)) ** 2 + (c - trefz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixz, _ = integrate.fixed_quad(lambda t: 3 * (trefx(t) - a) * (trefz(t) - c) * np.sqrt(trefnum(t))/(((a - trefx(t)) ** 2 + (b - trefy(t)) ** 2 + (c - trefz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyy, _ = integrate.fixed_quad(lambda t: (-1 * (a - trefx(t)) ** 2 + 2 * (b - trefy(t)) ** 2 - (c - trefz(t)) ** 2 )* np.sqrt(trefnum(t))/(((a - trefx(t)) ** 2 + (b - trefy(t)) ** 2 + (c - trefz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyz, _ = integrate.fixed_quad(lambda t: 3 * (trefy(t) - b) * (trefz(t) - c) * np.sqrt(trefnum(t))/(((a - trefx(t)) ** 2 + (b - trefy(t)) ** 2 + (c - trefz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
    elif knottype == "fig8":
        phixx, _ = integrate.fixed_quad(lambda t: (2 * (a - fig8x(t)) ** 2 - (b - fig8y(t)) ** 2 - (c - fig8z(t)) ** 2 )* np.sqrt(fig8num(t))/(((a - fig8x(t)) ** 2 + (b - fig8y(t)) ** 2 + (c - fig8z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixy, _ = integrate.fixed_quad(lambda t: 3 * (fig8x(t) - a) * (fig8y(t) - b) * np.sqrt(fig8num(t))/(((a - fig8x(t)) ** 2 + (b - fig8y(t)) ** 2 + (c - fig8z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixz, _ = integrate.fixed_quad(lambda t: 3 * (fig8x(t) - a) * (fig8z(t) - c) * np.sqrt(fig8num(t))/(((a - fig8x(t)) ** 2 + (b - fig8y(t)) ** 2 + (c - fig8z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyy, _ = integrate.fixed_quad(lambda t: (-1 * (a - fig8x(t)) ** 2 + 2 * (b - fig8y(t)) ** 2 - (c - fig8z(t)) ** 2 )* np.sqrt(fig8num(t))/(((a - fig8x(t)) ** 2 + (b - fig8y(t)) ** 2 + (c - fig8z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyz, _ = integrate.fixed_quad(lambda t: 3 * (fig8y(t) - b) * (fig8z(t) - c) * np.sqrt(fig8num(t))/(((a - fig8x(t)) ** 2 + (b - fig8y(t)) ** 2 + (c - fig8z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
    elif knottype == "torus31":
        phixx, _ = integrate.fixed_quad(lambda t: (2 * (a - torus31x(t)) ** 2 - (b - torus31y(t)) ** 2 - (c - torus31z(t)) ** 2 )* np.sqrt(torus31num(t))/(((a - torus31x(t)) ** 2 + (b - torus31y(t)) ** 2 + (c - torus31z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixy, _ = integrate.fixed_quad(lambda t: 3 * (torus31x(t) - a) * (torus31y(t) - b) * np.sqrt(torus31num(t))/(((a - torus31x(t)) ** 2 + (b - torus31y(t)) ** 2 + (c - torus31z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixz, _ = integrate.fixed_quad(lambda t: 3 * (torus31x(t) - a) * (torus31z(t) - c) * np.sqrt(torus31num(t))/(((a - torus31x(t)) ** 2 + (b - torus31y(t)) ** 2 + (c - torus31z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyy, _ = integrate.fixed_quad(lambda t: (-1 * (a - torus31x(t)) ** 2 + 2 * (b - torus31y(t)) ** 2 - (c - torus31z(t)) ** 2 )* np.sqrt(torus31num(t))/(((a - torus31x(t)) ** 2 + (b - torus31y(t)) ** 2 + (c - torus31z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyz, _ = integrate.fixed_quad(lambda t: 3 * (torus31y(t) - b) * (torus31z(t) - c) * np.sqrt(torus31num(t))/(((a - torus31x(t)) ** 2 + (b - torus31y(t)) ** 2 + (c - torus31z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
    elif knottype == "torus51":
        phixx, _ = integrate.fixed_quad(lambda t: (2 * (a - torus51x(t)) ** 2 - (b - torus51y(t)) ** 2 - (c - torus51z(t)) ** 2 )* np.sqrt(torus51num(t))/(((a - torus51x(t)) ** 2 + (b - torus51y(t)) ** 2 + (c - torus51z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixy, _ = integrate.fixed_quad(lambda t: 3 * (torus51x(t) - a) * (torus51y(t) - b) * np.sqrt(torus51num(t))/(((a - torus51x(t)) ** 2 + (b - torus51y(t)) ** 2 + (c - torus51z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixz, _ = integrate.fixed_quad(lambda t: 3 * (torus51x(t) - a) * (torus51z(t) - c) * np.sqrt(torus51num(t))/(((a - torus51x(t)) ** 2 + (b - torus51y(t)) ** 2 + (c - torus51z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyy, _ = integrate.fixed_quad(lambda t: (-1 * (a - torus51x(t)) ** 2 + 2 * (b - torus51y(t)) ** 2 - (c - torus51z(t)) ** 2 )* np.sqrt(torus51num(t))/(((a - torus51x(t)) ** 2 + (b - torus51y(t)) ** 2 + (c - torus51z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyz, _ = integrate.fixed_quad(lambda t: 3 * (torus51y(t) - b) * (torus51z(t) - c) * np.sqrt(torus51num(t))/(((a - torus51x(t)) ** 2 + (b - torus51y(t)) ** 2 + (c - torus51z(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
    elif knottype == "cinque":
        phixx, _ = integrate.fixed_quad(lambda t: (2 * (a - cinquex(t)) ** 2 - (b - cinquey(t)) ** 2 - (c - cinquez(t)) ** 2 )* np.sqrt(cinquenum(t))/(((a - cinquex(t)) ** 2 + (b - cinquey(t)) ** 2 + (c - cinquez(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixy, _ = integrate.fixed_quad(lambda t: 3 * (cinquex(t) - a) * (cinquey(t) - b) * np.sqrt(cinquenum(t))/(((a - cinquex(t)) ** 2 + (b - cinquey(t)) ** 2 + (c - cinquez(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixz, _ = integrate.fixed_quad(lambda t: 3 * (cinquex(t) - a) * (cinquez(t) - c) * np.sqrt(cinquenum(t))/(((a - cinquex(t)) ** 2 + (b - cinquey(t)) ** 2 + (c - cinquez(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyy, _ = integrate.fixed_quad(lambda t: (-1 * (a - cinquex(t)) ** 2 + 2 * (b - cinquey(t)) ** 2 - (c - cinquez(t)) ** 2 )* np.sqrt(cinquenum(t))/(((a - cinquex(t)) ** 2 + (b - cinquey(t)) ** 2 + (c - cinquez(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyz, _ = integrate.fixed_quad(lambda t: 3 * (cinquey(t) - b) * (cinquez(t) - c) * np.sqrt(cinquenum(t))/(((a - cinquex(t)) ** 2 + (b - cinquey(t)) ** 2 + (c - cinquez(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
    elif knottype == "3twist":
        phixx, _ = integrate.fixed_quad(lambda t: (2 * (a - threetwistx(t)) ** 2 - (b - threetwisty(t)) ** 2 - (c - threetwistz(t)) ** 2 )* np.sqrt(threetwistnum(t))/(((a - threetwistx(t)) ** 2 + (b - threetwisty(t)) ** 2 + (c - threetwistz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixy, _ = integrate.fixed_quad(lambda t: 3 * (threetwistx(t) - a) * (threetwisty(t) - b) * np.sqrt(threetwistnum(t))/(((a - threetwistx(t)) ** 2 + (b - threetwisty(t)) ** 2 + (c - threetwistz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixz, _ = integrate.fixed_quad(lambda t: 3 * (threetwistx(t) - a) * (threetwistz(t) - c) * np.sqrt(threetwistnum(t))/(((a - threetwistx(t)) ** 2 + (b - threetwisty(t)) ** 2 + (c - threetwistz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyy, _ = integrate.fixed_quad(lambda t: (-1 * (a - threetwistx(t)) ** 2 + 2 * (b - threetwisty(t)) ** 2 - (c - threetwistz(t)) ** 2 )* np.sqrt(threetwistnum(t))/(((a - threetwistx(t)) ** 2 + (b - threetwisty(t)) ** 2 + (c - threetwistz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyz, _ = integrate.fixed_quad(lambda t: 3 * (threetwisty(t) - b) * (threetwistz(t) - c) * np.sqrt(threetwistnum(t))/(((a - threetwistx(t)) ** 2 + (b - threetwisty(t)) ** 2 + (c - threetwistz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
    elif knottype == "granny":
        phixx, _ = integrate.fixed_quad(lambda t: (2 * (a - grannyx(t)) ** 2 - (b - grannyy(t)) ** 2 - (c - grannyz(t)) ** 2 )* np.sqrt(grannynum(t))/(((a - grannyx(t)) ** 2 + (b - grannyy(t)) ** 2 + (c - grannyz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixy, _ = integrate.fixed_quad(lambda t: 3 * (grannyx(t) - a) * (grannyy(t) - b) * np.sqrt(grannynum(t))/(((a - grannyx(t)) ** 2 + (b - grannyy(t)) ** 2 + (c - grannyz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixz, _ = integrate.fixed_quad(lambda t: 3 * (grannyx(t) - a) * (grannyz(t) - c) * np.sqrt(grannynum(t))/(((a - grannyx(t)) ** 2 + (b - grannyy(t)) ** 2 + (c - grannyz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyy, _ = integrate.fixed_quad(lambda t: (-1 * (a - grannyx(t)) ** 2 + 2 * (b - grannyy(t)) ** 2 - (c - grannyz(t)) ** 2 )* np.sqrt(grannynum(t))/(((a - grannyx(t)) ** 2 + (b - grannyy(t)) ** 2 + (c - grannyz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyz, _ = integrate.fixed_quad(lambda t: 3 * (grannyy(t) - b) * (grannyz(t) - c) * np.sqrt(grannynum(t))/(((a - grannyx(t)) ** 2 + (b - grannyy(t)) ** 2 + (c - grannyz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
    elif knottype == "square":
        phixx, _ = integrate.fixed_quad(lambda t: (2 * (a - squarex(t)) ** 2 - (b - squarey(t)) ** 2 - (c - squarez(t)) ** 2 )* np.sqrt(squarenum(t))/(((a - squarex(t)) ** 2 + (b - squarey(t)) ** 2 + (c - squarez(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixy, _ = integrate.fixed_quad(lambda t: 3 * (squarex(t) - a) * (squarey(t) - b) * np.sqrt(squarenum(t))/(((a - squarex(t)) ** 2 + (b - squarey(t)) ** 2 + (c - squarez(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixz, _ = integrate.fixed_quad(lambda t: 3 * (squarex(t) - a) * (squarez(t) - c) * np.sqrt(squarenum(t))/(((a - squarex(t)) ** 2 + (b - squarey(t)) ** 2 + (c - squarez(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyy, _ = integrate.fixed_quad(lambda t: (-1 * (a - squarex(t)) ** 2 + 2 * (b - squarey(t)) ** 2 - (c - squarez(t)) ** 2 )* np.sqrt(squarenum(t))/(((a - squarex(t)) ** 2 + (b - squarey(t)) ** 2 + (c - squarez(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyz, _ = integrate.fixed_quad(lambda t: 3 * (squarey(t) - b) * (squarez(t) - c) * np.sqrt(squarenum(t))/(((a - squarex(t)) ** 2 + (b - squarey(t)) ** 2 + (c - squarez(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
    elif knottype == "endless":
        phixx, _ = integrate.fixed_quad(lambda t: (2 * (a - endlessx(t)) ** 2 - (b - endlessy(t)) ** 2 - (c - endlessz(t)) ** 2 )* np.sqrt(endlessnum(t))/(((a - endlessx(t)) ** 2 + (b - endlessy(t)) ** 2 + (c - endlessz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixy, _ = integrate.fixed_quad(lambda t: 3 * (endlessx(t) - a) * (endlessy(t) - b) * np.sqrt(endlessnum(t))/(((a - endlessx(t)) ** 2 + (b - endlessy(t)) ** 2 + (c - endlessz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phixz, _ = integrate.fixed_quad(lambda t: 3 * (endlessx(t) - a) * (endlessz(t) - c) * np.sqrt(endlessnum(t))/(((a - endlessx(t)) ** 2 + (b - endlessy(t)) ** 2 + (c - endlessz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyy, _ = integrate.fixed_quad(lambda t: (-1 * (a - endlessx(t)) ** 2 + 2 * (b - endlessy(t)) ** 2 - (c - endlessz(t)) ** 2 )* np.sqrt(endlessnum(t))/(((a - endlessx(t)) ** 2 + (b - endlessy(t)) ** 2 + (c - endlessz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
        phiyz, _ = integrate.fixed_quad(lambda t: 3 * (endlessy(t) - b) * (endlessz(t) - c) * np.sqrt(endlessnum(t))/(((a - endlessx(t)) ** 2 + (b - endlessy(t)) ** 2 + (c - endlessz(t)) ** 2) ** (2.5)),0, 2*np.pi, n = quadorder)
    else: 
        print("Invalid knot type")
        phixx = None
        phixy = None
        phixz = None
        phiyy = None
        phiyz = None
    return np.array([-1*phixx, -1*phixy, -1*phixz, -1*phiyy, -1*phiyz])
        
        
        
        

#Computes Jacobian of the electric field
def Je(a,b,c,knottype):
    
    J = np.zeros((3,3))
    entries = JEentries(a,b,c,knottype)
    
    
    #Using symmetry and harmonicity, I only need to compute 5 out of 9 second order partial derivatives
    
    J[0,0] = entries[0]
    J[0,1] = entries[1]
    J[0,2] = entries[2]
    
    J[1,0] = entries[1]
    J[1,1] = entries[3]
    J[1,2] = entries[4]
    
    J[2,0] = entries[2]
    J[2,1] = entries[4]
    J[2,2] = -1 * (entries[0] + entries[3])
    
    return J
    

#Perform one iteration of the multivariable Newton method
def newtoniterate(a,b,c,knottype):
    J = Je(a,b,c,knottype)
    fx = efield(a,b,c,knottype).T
    return np.array([a,b,c]).T - np.matmul(np.linalg.inv(J),fx)



m = 20
#Perform m iterations of the multivariable Newton method to find the zero of the electric field
def newton(a,b,c, knottype):
    for i in range(m):
        newx = newtoniterate(a,b,c,knottype).T
        a = newx[0]
        b = newx[1]
        c = newx[2]
        #If the iteration tends to the singularity at infinity, reject the result
        if np.linalg.norm(np.array([a,b,c])) > 10:
            return None
        
    #Test if the given output is indeed a fixed point of E
    if np.linalg.norm(efield(newx[0],newx[1],newx[2],knottype)) > 10e-6:
        return None
    return newx


#Checks if the root is valid, and then append it to the list
#Note the threshold parameter for determining when two roots are the same, or if there is a root on the knot due to
#evaluation error when computing the Gaussian integrals (trying to evaluate the potential or the electric field
#directly on the knot doesn't give a blowup error like it should)
def addroot(root, knottype, zeros):
    for pt in zeros:
        if np.linalg.norm(pt - root) < 10e-3:
            #Reject the root because it is a duplicate
            return
    
    
    knotx, knoty, knotz = getpts(knottype)
    for i in range(np.size(knotx)):
        if np.linalg.norm(np.array([knotx[i],knoty[i],knotz[i]]) - root) < 10e-3:
            #Reject the root because it is on the knot
            return

        
    zeros.append(root)
    print('Root found! ', root, ' Potential: ', potential(root[0],root[1],root[2],knottype))
    return
    


#Search for the critical points given the knot type, and the fineness of the grid of initial guesses
#We take an 8x8x8 cube centered at the origin and break it up into N equally spaced grid points along each side
def criticalsearch(knottype, N): 
    print('Initializing critical point search: ', knottype)
    zeros = []
    mgrid = makedomain(N)
    for i in range(N):
        for j in range(N):
                for k in range(N):
                    root = newton(mgrid[0][i][j][k], mgrid[1][i][j][k], mgrid[2][i][j][k], knottype)
                    if root is not None:
                        addroot(root, knottype, zeros)
                
    print('Root search complete.')
    return np.stack(zeros)
    
   
#Create an animation for the evolution of level surfaces
#Shows the animation of Phi(x) = cmin to Phi(x) = cmax, with specified total of frames, and surface mesh size n   
def createanim(knottype, cmin, cmax, n, totalframes, colorscale = 'inferno'):
    verts = []
    faces = []
    steplength = (cmax - cmin) / totalframes
    
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        knottypes = [knottype] * totalframes
        ns = [n] * totalframes
        cs = [cmin + i * steplength for i in range(totalframes)]
        #results = [executor.submit(generateverts, knottype, n, c=cmin + i * steplength) for i in range(totalframes)]
        results = executor.map(generateverts,knottypes,ns,cs)

        for f in results:
            newverts, newfaces = f
            verts.append(newverts)
            faces.append(newfaces)
        
    
    
   # for i in range(totalframes):
   #     newverts, newfaces = generateverts(knottype, n, cmin + i * steplength)
   #     verts.append(newverts)
   #     faces.append(newfaces)
    
    fig = go.Figure(

        go.Mesh3d(
                    x=verts[0].T[0], y=verts[0].T[1], z=verts[0].T[2],
                    i=faces[0].T[0], j=faces[0].T[1], k=faces[0].T[2],
                    colorscale=colorscale, intensity=verts[0].T[2], showscale=False, hoverinfo='skip',
                    name='Cinquefoil Animation'
                    ),

        layout=go.Layout(
            xaxis=dict(range=[0, 5], autorange=False),
            yaxis=dict(range=[0, 5], autorange=False),
            title=knottype + " animation",
            updatemenus=[dict(
                type="buttons",
                buttons=[dict(label="Play",
                              method="animate",
                              args=[None, {"frame": {"duration": 15}}]),
                         dict(label = "Pause",
                              method = "animate",
                              args = [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}])  
                        ])]
        ),

    frames=[go.Frame(data=[go.Mesh3d(
                    x=verts[i].T[0], y=verts[i].T[1], z=verts[i].T[2],
                    i=faces[i].T[0], j=faces[i].T[1], k=faces[i].T[2],
                    colorscale=colorscale, intensity=verts[i].T[2], showscale=False, hoverinfo='skip',
                    name='Trefoil Animation'
                    )]) for i in range(totalframes)]



    );

    fig.update_layout(scene1=dict(xaxis = dict(
                             visible=False),
                        yaxis = dict(
                             visible=False),
                        zaxis = dict(
                             visible=False),
                            ), 
                         scene_aspectratio=dict(x=1, y=1, z=0.5));

    fig.show()



