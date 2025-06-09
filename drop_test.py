import numpy as np
import DynamiX

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from DynamiX import set_space_dimension
from DynamiX import System, Particle, SphereToPlaneKV, SpringFixedPoint, Spring
from DynamiX.force_laws.sphere_to_sphere_KV import SphereToSphereKV
from DynamiX import semi_implicit_euler
from DynamiX import stoermer_verlet
set_space_dimension(3)

if __name__ == "__main__":

    system = System(gravity=np.array([0, 0, -9.81]))
    m = 0.3 #mass
    N = 20 # no of particles
    c = 5000 #stifness
    eta = 100 #damping

    l0 = 0.15

    fixed_points = []
    #k = 0
    for i in range(N):
        for j in range(N):
            x0 = i * l0
            y0 = j * l0
            z0 = 0
            #k = 0

            system.add_particles(Particle(m, r0 = np.array([x0, y0, z0])))
            if i == 0:
                point_left = np.array([x0 - l0, y0, z0])
                fixed_points.append(point_left)
                system.add_interactions(SpringFixedPoint(point_left, system.particles[i * N + j], c,eta, l0))
            if i == N - 1:
                point_right = np.array([x0 + l0, y0, z0])
                fixed_points.append(point_right)
                system.add_interactions(SpringFixedPoint(point_right, system.particles[i * N + j], c,eta, l0))
            if j == 0:
                point_below = np.array([x0, y0 - l0, z0])
                fixed_points.append(point_below)
                system.add_interactions(SpringFixedPoint(point_below, system.particles[i * N + j], c,eta, l0))
            if j == N - 1:
                point_above = np.array([x0, y0 + l0, z0])
                fixed_points.append(point_above)
                system.add_interactions(SpringFixedPoint(point_above, system.particles[i * N + j], c,eta, l0))

    print('fixed points:',fixed_points)
    print(len(fixed_points))#for debuging

    current_right = []
    right_array = []
    for i in range(N):
        for j in range(N-1):
            current = system.particles[i*(N) + j]
            current_right.append(i*(N) + j)


            if j < N :
                right = system.particles[(i)*(N) + (j+1)]
                right_array.append((i)*(N) + (j+1))
                system.add_interactions(Spring(current, right, c,eta, l0))
    current_bottom = []
    below_array = []
    for j in range(N):
        for i in range(N-1):
            current = system.particles[i*(N) + j]
            current_bottom.append(i*(N) + j)

            if i < N-1 :
                below = system.particles[((i*N)+N) + (j)]
                below_array.append(((i*N)+N) + (j))

                system.add_interactions(Spring(current, below, c,eta, l0))

    m_b = 200
    c_b = 500000
    eta_b = 100
    h_b = 1.5
    R_b = 0.5  #boulders parameters

    R_p = 0.01

    system.add_particles(Particle(m_b,r0 = np.array([h_b,h_b,h_b]), v0 = [0,0,0]))
    boulder = system.particles[400]
    boulder_index = len(system.particles)-1

    for particle in system.particles[:-1]:
        system.add_interactions(SphereToSphereKV(boulder, particle,R_b,R_p, c_b, eta_b))

    system.assemble()

    tf = 30
    h = 1e-3
    t, q, v = stoermer_verlet(system, h, tf)

    center_particle = system.particles[int(((N*N)/2) + N/2)]
    z_disp = q[:,  int(((N*N)/2) + N/2)*3 +2]
    plt.figure(figsize=(10, 5))
    plt.plot(t, z_disp, label="Z-Displacement of Center Particle")
    plt.xlabel("Time (s)")
    plt.ylabel("Displacement in Z (m)")
    plt.title("Z-Displacement Over Time for Center Net Particle")
    plt.legend()
    plt.grid(False)
    plt.show()
    plt.savefig("time_vs_displacement_plot_drop_test", dpi=300)

    # Create figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-0.2, 3.2])
    ax.set_ylim([-0.2, 3.2])
    ax.set_zlim([-2, 2])

    # Initialize scatter plot for net particles
    net_particles = ax.scatter([], [], [], s=2, c='blue', label="Net Particles")
    boulder_particle, = ax.plot([], [], [], 'ro', markersize=20, label="Boulder")  # Boulder as red sphere

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    plt.title("3D Simulation of Boulder Impact on Net")


    def init():
        net_particles._offsets3d = ([], [], [])
        boulder_particle.set_data([], [])
        boulder_particle.set_3d_properties([])
        return net_particles, boulder_particle


    def update(frame):
        # Get particle positions in 3D for net
        x = q[frame, 0::3]
        y = q[frame, 1::3]
        z = q[frame, 2::3]

        bx = q[frame, boulder_index * 3]
        by = q[frame, boulder_index * 3 + 1]
        bz = q[frame, boulder_index * 3 + 2]

        net_particles._offsets3d = (x[:-1], y[:-1], z[:-1]) # Exclude the last particle(boulder)

        boulder_particle.set_data(bx, by)
        boulder_particle.set_3d_properties(bz)

        return net_particles, boulder_particle


    drop_test_ani = FuncAnimation(fig, update, frames=len(q), init_func=init, blit=False, interval=20)
    #drop_test_ani.save("drop_test_animation_given.gif", writer="ffmpeg", fps=20)
    plt.legend()
    plt.show()
