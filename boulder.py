import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from DynamiX import set_space_dimension
#set_space_dimension(3)
from DynamiX import System, Particle, SphereToPlaneKV
from DynamiX.force_laws.sphere_to_sphere_KV import SphereToSphereKV
from DynamiX.force_laws.spring_fixed_point import SpringFixedPoint
from DynamiX import semi_implicit_euler, stoermer_verlet

if __name__ == "__main__":

    system = System(gravity=np.array([0,0, -9.81])) # uniform gravity with gravitational acceleration [1.0, 1.0]


    m0 = 200
    m1 = 0.3
    c = 500000
    eta = 1000

    h0 = 1.5
    R0 = 0.5
    R1 = 0.05
    h1 = 0

    system.add_particles(
            Particle(m0, r0=np.array([0,0, h0]), v0= np.array([0,0, h0]) )
        )
    system.add_particles(
        Particle(m1, r0=np.array([0.2, 0, h1]),v0= np.array([0,0, h0]))
    )

    #fixed_position = [0, 0.15, 0]
    #system.add_interactions(SpringFixedPoint(fixed_position, system.particles[1], 1e7, eta, 0.15))
    system.add_interactions(
            SphereToSphereKV(system.particles[0], system.particles[1], R0,R1, c, eta)
    )
    system.add_interactions(
        SphereToPlaneKV(system.particles[0], np.array([0,0, 1]), np.zeros(3), R0, c, eta)
    )
    system.add_interactions(
        SphereToPlaneKV(system.particles[1], np.array([0,0, 1]), np.zeros(3), R1, c, eta)
    )
    system.assemble()

    # Define the simulation parameters
    tf = 3              # Final time (s)
    h = 1e-4            # Time step (s)

    # Simulate using both methods
    t, q, v = stoermer_verlet(system, h, tf)

    # Compute lengths
    # Compute gap function over time
    gap = [system.interactions[0].g(qi) for qi in q]

    # Plot the gap over time
    plt.figure(figsize=(10, 5))
    plt.plot(t, gap, label='$g$', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Gap (m)')
    plt.title('Gap Over Time')
    plt.grid(True)
    plt.show()

    ###########
    # Animation Setup

    fps = 20
    N_frames = (tf - system.t0) * fps
    N = int((tf - system.t0) / h)
    frac = int(np.ceil(N / N_frames))
    q_frames = q[::frac]
    N_frames = q_frames.shape[0]

    # Set up the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 2])

    radii = np.array([0.5, 0.1])  # for particle sizes


    particles = ax.scatter([], [], [], s=[], color='blue')  # Scale sizes


    def init():
        particles._offsets3d = ([], [], [])
        particles.set_sizes([])
        return particles,


    def update(frame):
        x = q_frames[frame, 0::3]
        y = q_frames[frame, 1::3]
        z = q_frames[frame, 2::3]
        sizes = radii[:len(x)] * 500
        # Update particle positions and sizes dynamically
        particles._offsets3d = (x, y, z)
        particles.set_sizes(sizes)  # to visibilly show the size difference

        return particles,


    ani = FuncAnimation(fig, update, frames=N_frames, init_func=init, blit=False, interval=1000 / fps)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title('3D Sphere-to-Sphere Contact Animation')
    plt.show()
    