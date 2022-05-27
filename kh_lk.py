import taichi as ti
import numpy as np
import matplotlib.cm as cm

ti.init(arch = ti.gpu, default_fp = ti.f64)

# number of grid
NX = 600
NY = 200

# size of grid
dx = 3.0 / NX
dy = 1.0 / NY

# Ratio of Specific Heat
gam = 5.0 / 3.0 

# constant in Jameson's scheme
k2 = 0.5
k4 = 1.0 / 64.0

# initial value
prim1 = ti.Matrix([2.0, 0.5, 0.0, 2.5])
prim2 = ti.Matrix([1.0, -0.5, 0.0, 2.5])

# color
cmap_name = 'Blues_r'

# timestep
cfl = 0.5
dt = ti.field(ti.f64, shape = ())

# conserved quantitiy
w = ti.Vector.field(4, ti.f64, shape = (NX + 6, NY + 6))
w_old = ti.Vector.field(4, ti.f64, shape = (NX + 6, NY + 6))
img = ti.field(ti.f64, shape = (NX + 6, NY + 6))

# flux
F = ti.Vector.field(4, ti.f64, shape = (NX + 4, NY + 3))
G = ti.Vector.field(4, ti.f64, shape = (NX + 3, NY + 4))

def initialize():
    for i in range(NX + 6):
        for j in range(NY + 6):
            x = (i - 2.5) * dx
            y = (j - 2.5) * dy
            w[i, j][0] = 1.0 + np.heaviside(0.5 - y, 0)
            w[i, j][1] = (-0.5 + np.heaviside(0.5 - y, 0)) * w[i, j][0]
            w[i, j][2] = 0.5 * x * np.sin(4 * np.pi * x) * (np.exp(-(y - 0.5) ** 2 * 400)) * w[i, j][0]
            w[i, j][3] = 2.5 * np.ones_like(x) / (gam - 1.0) + 0.5 * (w[i, j][1] ** 2 + w[i, j][2] ** 2) / w[i, j][0]

@ti.kernel
def boundary(w: ti.template()):
    for i, j in ti.ndrange(3, NY + 6):
        w[i, j] = w[NX + i, j]   # Periodic Boundary

    for i, j in ti.ndrange((NX + 3, NX + 6), NY + 6):
        w[i, j] = w[i - NX, j]   # Periodic Boundary

    # slip boundary condition
    for i, j in ti.ndrange(NX + 6, 3):
        w[i, j] = get_conserved(prim1)

    # slip boundary condition
    for i, j in ti.ndrange(NX + 6, (NY + 3, NY + 6)):
        w[i, j] = get_conserved(prim2)

@ti.kernel
def timestep():
    dt[None] = 1.0e5
    for i, j in w:
        p = get_prim(w[i, j])
        a = ti.sqrt(gam * p / w[i, j][0])
        dtij = cfl / ((abs(w[i, j][1] / w[i, j][0]) + a) / dx + (abs(w[i, j][2] / w[i, j][0]) + a) / dy)
        ti.atomic_min(dt[None], dtij)

@ti.func
def get_conserved(prim):
    w = ti.Matrix([prim[0], prim[0] * prim[1], prim[0] * prim[2], prim[3] / (gam - 1) + 0.5 * prim[0] * (prim[1] ** 2 + prim[2] ** 2)])
    return w

@ti.func
def get_prim(w):
    return ((w[3] - 0.5 * (w[1] ** 2 + w[2] ** 2) / w[0]) * (gam - 1))

@ti.kernel
def get_w_old():
    for i, j in w:
        w_old[i, j] = w[i, j]

@ti.kernel
def flux(nx: ti.i32, ny: ti.i32, F:ti.template()):
    for i, j in ti.ndrange((3, NX + 3 + nx), (3, NY + 3 + ny)):
        w0 = ti.Matrix([0.0, 0.0, 0.0, 0.0])
        pp = ti.Matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        w0 = 0.5 * (w[i - nx, j - ny] + w[i, j])
        p = get_prim(w0)
        wn = w0[1] * nx + w0[2] * ny
        H = ti.Matrix([wn, w0[1] * wn / w0[0] + p * nx, wn * w0[2] / w0[0] + p * ny, (w0[3] + p) * wn / w0[0]])
        alpha = abs(w0[1] / w0[0]) + ti.sqrt(gam * p / w0[0])
        for k in ti.static(range(6)):
            pp[k] = get_prim(w[i + (2 - k) * nx, j + (2 - k) * ny])
        epsilon2 = k2 * max(get_mu(pp[0], pp[1], pp[2]), get_mu(pp[1], pp[2], pp[3]), get_mu(pp[2], pp[3], pp[4]), get_mu(pp[3], pp[4], pp[5]))
        epsilon4 = max(0, k4 - epsilon2)
        F[i, j] = H - alpha * (epsilon2 * (w[i, j] - w[i - nx, j - ny]) - epsilon4 * (w[i + nx, j + ny] - 3 * w[i, j] + 3 * w[i - nx, j - ny] - w[i - 2 * nx, j - 2 * ny]))

@ti.func
def get_mu(p1, p2, p3):
    return abs((p1 - 2 * p2 + p3) / (p1 + 2 * p2 + p3))

@ti.kernel
def update(dt: ti.f64):
    for i, j in ti.ndrange((3, NX + 3), (3, NY + 3)):
        w[i, j] = w_old[i, j] + dt * (F[i, j] - F[i + 1, j]) / dx + dt * (G[i, j] - G[i, j + 1]) / dy

@ti.kernel
def image():
    for i, j in img:
        img[i, j] = (w[i, j][0] - 0.8) / (2.0 - 0.8)

gui = ti.GUI("kh", (NX + 6, NY + 6))
cmap = cm.get_cmap(cmap_name)

initialize()
boundary(w)
istep = 0

while gui.running:
    timestep()
    istep += 1
    get_w_old()
 
    for k in range(4):
        flux(1, 0, F)
        flux(0, 1, G)
        dtmin = dt[None] / (4.0 - k)
        update(dtmin)
        boundary(w)

    image()
    if istep % 5 == 0:
        gui.set_image(cmap(img.to_numpy()))
        gui.show()
        #gui.show("test{:03d}.png".format(istep))
