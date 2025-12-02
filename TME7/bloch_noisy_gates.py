import random

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import CSS4_COLORS

from random import choice

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.array([[1, 0], [0, 1]], dtype=complex)

paulis= [X,Y,Z]

# --- Helper to later get a random color for a vector on the Bloch sphere (cannot be black, cyan, red, or green to avoid confusing with axes)
forbidden_colors = ['black', 'cyan', 'red', 'green']
colors_CSS4 = list(CSS4_COLORS.keys())

for color in forbidden_colors:
    if color in colors_CSS4:
        del CSS4_COLORS[color]

colors = list(CSS4_COLORS.keys())


# --- We define a utility function to plot the empty Bloch sphere, such that later on we can reuse it to plot some trajectories and points on it.
def bloch_sphere(ax):
    u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]

    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)

    # Get rid of colored axes planes
    # First remove fill
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Now set color to white (or whatever is "invisible")
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # Plot the sphere surface
    ax.plot_wireframe(x, y, z, color="lightgray", alpha=0.15)

    # Add colored orthonormal axes with alpha 0.6
    alpha = 0.5
    ax.quiver(0,0,0,1,0,0,color='r',linewidth=2, arrow_length_ratio=0.1, alpha=alpha)
    ax.quiver(0,0,0,0,1,0,color='g',linewidth=2, arrow_length_ratio=0.1, alpha=alpha)
    ax.quiver(0,0,0,0,0,1,color='cyan',linewidth=2, arrow_length_ratio=0.1, alpha=alpha)
    ax.text(1.2, 0, 0, 'X', color='r', fontsize=18, alpha=0.6)  # Red for X
    ax.text(0, 1.2, 0, 'Y', color='g', fontsize=18, alpha=0.6)  # Green for Y
    ax.text(0, 0, 1.2, 'Z', color='cyan', fontsize=18, alpha=0.6)  # Blue for Z

    # Add black orthonormal axes with alpha 0.3
    ax.plot([-1.2, 1.2], [0, 0], [0, 0], color='black', alpha=0.3, linewidth=1)
    ax.plot([0, 0], [-1.2, 1.2], [0, 0], color='black', alpha=0.3, linewidth=1)
    ax.plot([0, 0], [0, 0], [-1.2, 1.2], color='black', alpha=0.3, linewidth=1)

    # Set axis limits to zoom in
    ax.set_xlim([-1.01, 1.01])
    ax.set_ylim([-1.01, 1.01])
    ax.set_zlim([-1.01, 1.01])

    # Hide the axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_box_aspect([1,1,1])
    ax.set_axis_off()


# --- Noise / channel functions (Put your functions there) ---
def phase_damping(r: np.ndarray, t: float, T2: float) -> np.ndarray:
    """
    Apply phase damping to a Bloch vector.
    :param r: Bloch vector (x, y, z)
    :param t: time
    :param T2: phase damping time constant
    :return: new Bloch vector after phase damping
    """
    x, y, z = r
    new_x = np.exp(-t / T2) * x
    new_y = np.exp(-t / T2) * y
    new_z = z
    return np.array([new_x, new_y, new_z], dtype=float)


def amplitude_damping(r: np.ndarray, t: float, T1: float) -> np.ndarray:
    """
    Apply amplitude damping to a Bloch vector.
    :param r: Bloch vector (x, y, z)
    :param t: time
    :param T1: amplitude damping time constant
    :return: new Bloch vector after amplitude damping
    """
    x, y, z = r
    new_x = np.exp(-t / (2 * T1)) * x
    new_y = np.exp(-t / (2 * T1)) * y
    new_z = np.exp(-t / T1) * z + (1 - np.exp(-t / T1))
    return np.array([new_x, new_y, new_z], dtype=float)


def depolarizing_noise(r: np.ndarray, p: float) -> np.ndarray:
    """
    Apply depolarizing noise to a Bloch vector.
    :param r: Bloch vector (x, y, z)
    :param p: depolarizing probability
    :return: new Bloch vector after depolarizing noise
    """
    new_r = (1 - p) * r
    return new_r

def bloch_to_rho(r: np.ndarray) -> np.ndarray:
    """
    Convert a Bloch vector to its corresponding density matrix.
    :param r: Bloch vector (x, y, z)
    :return: 2x2 density matrix
    """
    x,y,z = r
    return np.array(1/2 * (I + x*X + y*Y + z*Z))


# --- Define line styles for different channels ---
CHANNEL_STYLES = {
    "Phase Damping": "--",
    "Amplitude Damping": "-",
    "Depolarizing Noise": ":"
}


# --- Generate trajectories for each channel ---
def generate_trajectory(gate, channel, r0, time_steps, use_time=True, p_0=0.0, **kwargs):
    """
    Generate a trajectory of a Bloch vector under a given quantum channel.
    :param channel_func: function implementing the quantum channel
    :param r0: initial Bloch vector
    :param time_steps: array of time steps
    :param use_time: whether the channel function requires time as an argument
    :param kwargs: additional parameters for the channel function
    :return: array of Bloch vectors representing the trajectory
    """
    trajectory = [r0]
    r = r0
    for t in time_steps[1:]:
        if channel == 0:
            r = (1-p_0) * rho_to_bloch(I @ gate  @ bloch_to_rho(r) @ gate.conj().T @ I)
            r += (p_0 / 3) * rho_to_bloch(X @ gate  @ bloch_to_rho(r) @ gate.conj().T @ X)
            r += (p_0 / 3) * rho_to_bloch(Y @ gate  @ bloch_to_rho(r) @ gate.conj().T @ Y)
            r += (p_0 / 3) * rho_to_bloch(Z @ gate  @ bloch_to_rho(r) @ gate.conj().T @ Z)

        elif channel == 1:
            K_0 = np.sqrt(1-p_0) * I
            K_1 = np.sqrt(p_0) * (ket0 @ ket0.conj().T)
            K_2 = np.sqrt(p_0) * (ket1 @ ket1.conj().T)
            Ks = [K_0, K_1, K_2]
            count = 0
            for i in Ks:
                if count == 0:
                    r = rho_to_bloch(i @ gate @ bloch_to_rho(r) @ gate.conj().T @ i.conj().T)
                else:
                    r += rho_to_bloch(i @ gate @ bloch_to_rho(r) @ gate.conj().T @ i.conj().T)
                count+=1
        else:
            K_0 = np.array([[1,0],[0, np.sqrt(1-p_0)]])
            K_1 = np.array([[0,np.sqrt(p_0)],[0, 0]])
            Ks = [K_0, K_1]
            count = 0
            for i in Ks:
                if count == 0:
                    r = rho_to_bloch(i @ gate @ bloch_to_rho(r) @ gate.conj().T @ i.conj().T)
                else:
                    r += rho_to_bloch(i @ gate @ bloch_to_rho(r) @ gate.conj().T @ i.conj().T)
                count+=1
        trajectory.append(r)
    return np.array(trajectory)


# --- Plotting function for trajectories ---
def plot_trajectories(trajectories, time_steps, interval=100, save_anim=False, anim_name='bloch_trajectories.gif'):
    """
    Plot and animate multiple Bloch sphere trajectories.
    :param trajectories: list of tuples (trajectory, label, color, linestyle)
    :param time_steps: array of time steps
    :param interval: interval between frames in milliseconds
    :param save_anim: whether to save the animation as a GIF
    :param anim_name: name of the output GIF file
    :return: None
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    bloch_sphere(ax)

    lines = []
    points = []

    added_labels = set()  # avoid duplicate legend entries

    for traj, label, color, linestyle in trajectories:
        # Only add label the first time we see this channel
        legend_label = label if label not in added_labels else "_nolegend_"
        added_labels.add(label)

        line, = ax.plot([], [], [], linestyle=linestyle, color=color, label=legend_label)
        point, = ax.plot([], [], [], 'o', color=color)

        lines.append(line)
        points.append(point)

    def update(frame):
        for i, (traj, _, _, _) in enumerate(trajectories):
            lines[i].set_data(traj[:frame, 0], traj[:frame, 1])
            lines[i].set_3d_properties(traj[:frame, 2])

            points[i].set_data(traj[frame-1:frame, 0], traj[frame-1:frame, 1])
            points[i].set_3d_properties(traj[frame-1:frame, 2])

        return lines + points

    ani = FuncAnimation(fig, update, frames=len(time_steps), interval=interval, blit=True)

    # Display legend such that linestyle lines are black
    legend_lines = []
    for label in added_labels:
        linestyle = CHANNEL_STYLES[label]
        legend_line, = ax.plot([], [], [], linestyle=linestyle, color='black', label=label)
        legend_lines.append(legend_line)
    ax.legend(handles=legend_lines, loc='upper right')

    plt.title('Bloch Sphere Trajectories')

    if save_anim:  # Save as MP4 if ffmpeg is available
        try:
            writer = FFMpegWriter(fps=1000 // interval)
            ani.save(anim_name.replace('.gif', '.mp4'), writer=writer)
            print(f"Animation saved as MP4: {anim_name.replace('.gif', '.mp4')}")
        except Exception as e:
            print(f"Failed to save as MP4: {e}")
            print(f"Attempting to save as GIF instead")
            try:
                writer = PillowWriter(fps=1000 // interval)
                ani.save(anim_name, writer=writer)
                print(f"Animation saved as GIF: {anim_name}")
            except Exception as gif_error:
                print(f"Failed to save as GIF: {gif_error}")
                print("Animation could not be saved in any format.")
    else:
        plt.show()

def rho_to_bloch(rho: np.ndarray) -> np.ndarray:
    """
    Convert a density matrix to its corresponding Bloch vector.
    :param rho: 2x2 density matrix
    :return: Bloch vector (x, y, z)
    """
    bloch_vector = [np.trace(rho@X, dtype=float), np.trace(rho@Y, dtype=float), np.trace(rho@Z, dtype=float)]
    return np.array(bloch_vector)


if __name__ == "__main__":
    # Initial a list of random Bloch vectors as starting points
    number_of_vectors = 4
    ket0 = np.array([[1], [0]], dtype=complex)
    rho0 = ket0 @ ket0.conj().T
    ket1 = np.array([[0], [1]], dtype=complex)
    rho1 = ket1 @ ket1.conj().T
    ket_plus = 1/np.sqrt(2)*(ket0+ket1)
    rhoplus = ket_plus @ ket_plus.conj().T
    ket_i = 1/np.sqrt(2)*(ket0+1j*ket1)
    rhoi = ket_i @ ket_i.conj().T
    r0_list_normalized = [rho_to_bloch(rho0), rho_to_bloch(rho1), rho_to_bloch(rhoplus), rho_to_bloch(rhoi)]

    # Assign one color per initial vector
    vector_colors = [choice(colors) for _ in range(len(r0_list_normalized))]

    # Time steps
    time_steps = np.linspace(start=0, stop=5, num=200)

    # Collect all trajectories with style metadata
    trajectories = []
    for r0, color in zip(r0_list_normalized, vector_colors):

        traj_depolarizing = generate_trajectory(X, 0, r0, time_steps, p_0=0.02)
        trajectories.append((traj_depolarizing, 'Depolarizing Noise', color, CHANNEL_STYLES["Depolarizing Noise"]))

        traj_phase = generate_trajectory(X, 1, r0, time_steps, p_0=0.02)
        trajectories.append((traj_phase, 'Phase Damping', color, CHANNEL_STYLES["Phase Damping"]))


        traj_amplitude = generate_trajectory(X, 2, r0, time_steps, p_0=0.02)
        trajectories.append((traj_amplitude, 'Amplitude Damping', color, CHANNEL_STYLES["Amplitude Damping"]))

    plot_trajectories(
        trajectories, time_steps,
        interval=100,
        save_anim=True,
        anim_name='bloch_trajectories.gif'
    )
