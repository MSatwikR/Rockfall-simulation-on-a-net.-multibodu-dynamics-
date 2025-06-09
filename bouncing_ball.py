import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from DynamiX import set_space_dimension
set_space_dimension(2) # this overrides the standard value of 3

from DynamiX import System, Particle, SphereToPlaneKV
from DynamiX.force_laws.sphere_to_sphere_KV import SphereToSphereKV
from DynamiX import semi_implicit_euler

if __name__ == "__main__":

    system = System(gravity=np.array([0, -10.])) # uniform gravity with gravitational acceleration [1.0, 1.0]

    m = 1
    c = 5000
    eta = 10

    y0 = 2
    R = 0.05

    system.add_particles(
            Particle(m, r0=np.array([0, y0]))
        )
    system.add_interactions(
            SphereToSphereKV(system.particles[0], np.array([0, 1]), np.zeros(2), R, c, eta)
    )
    
    system.assemble()

    # Define the simulation parameters
    tf = 5              # Final time (s)
    h = 1e-4            # Time step (s)

    # Simulate using both methods
    t, q, v = semi_implicit_euler(system, h, tf)

    # Compute lengths
    gap = [system.interactions[0].g(qi) for qi in q]

    # Plot the lengths over time on the same plot
    plt.figure(figsize=(10, 5))
    plt.plot(t, gap, label='$g$', linestyle='-')
    plt.xlabel('Time (s)')
    plt.ylabel('Gap (m)')
    plt.title('Gap Over Time')
    # plt.legend()
    plt.grid(True)

    plt.show()

    ###########
    # Animation

    # Animation setup
    fps = 20
    N_frames = (tf - system.t0) * fps
    N = int((tf - system.t0) / h)
    frac = int(np.ceil(N / N_frames))
    q_frames = q[::frac]
    N_frames = q_frames.shape[0]

    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)

    # Draw particles and trails
    particles, = ax.plot([], [], 'bo', markersize=8)

    # draw the horizontal plane
    ax.plot([-2, 2], [0, 0], 'k')


    # Function to initialize the animation
    def init():
        particles.set_data([], [])
        return particles, 

    # Function to update the animation
    def update(frame):
        # Get particle positions
        x = q_frames[frame, 0::2]
        y = q_frames[frame, 1::2]
        particles.set_data(x, y)


        return particles,

    # Create and display the animation
    ani = FuncAnimation(fig, update, frames=N_frames, init_func=init, blit=True, interval=1000 / fps)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Bouncing Ball System')
    plt.show()
    