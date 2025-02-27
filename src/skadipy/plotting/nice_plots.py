import numpy as np
import matplotlib.pyplot as plt
import skadipy.allocator
import typing

from skadipy.plotting.nice_colors import colors, darker_colors, ligther_colors


def nice_plot_histories(tau_cmd, tau_alloc, indices=[0, 1, 5], dt=1.0):
    labels = [r"$F_x$", r"$F_y$", r"$F_z$", r"$M_x$", r"$M_y$", r"$M_z$"]

    fig, ax = plt.subplots(len(indices), 1, figsize=(8, 8))

    fig.tight_layout(pad=1.5)

    t = np.linspace(0, dt * len(tau_cmd), len(tau_cmd))

    for i in range(len(indices)):
        ax[i].plot(
            t,
            tau_cmd[:, indices[i]],
            label="Input " + labels[indices[i]],
            linewidth=1,
            linestyle="-",
            color="black",
        )

        for j in range(len(tau_alloc)):
            ax[i].plot(
                t,
                tau_alloc[j][:, indices[i]],
                label="Output " + labels[indices[i]],
                linewidth=1,
                linestyle="-",
                color=colors[j],
            )

        ax[i].set_xlabel("Time [s]")
        ax[i].set_ylabel(labels[indices[i]] + " [N]")

        ax[i].grid(True)
        ax[i].legend()

    return fig, ax


def nice_plot_2d_allocation(
    tau_cmd: np.ndarray,
    allocators: typing.List[skadipy.allocator.AllocatorBase],
    tau_hist: np.ndarray,
    dt=1.0,
    npoints=150,
):

    fig, ax = plt.subplots(2, 1, height_ratios=[4, 1], figsize=(5.5, 6))

    t = np.linspace(0, dt * len(tau_cmd), len(tau_cmd))

    fig.tight_layout(pad=0.2)

    step_size = len(tau_cmd) // npoints

    ax[0].scatter(
        tau_cmd[::step_size, 0],
        tau_cmd[::step_size, 1],
        s=5,
        label="Input forces",
        color="black",
    )

    for allocator, F, color in zip(allocators, tau_hist, colors):
        ax[0].scatter(F[::step_size, 0], F[::step_size, 1], s=5, color=color)
        for i in range(0, len(tau_cmd), step_size):
            ax[0].plot(
                [tau_cmd[i, 0], F[i, 0]],
                [tau_cmd[i, 1], F[i, 1]],
                "--",
                color=color,
                lw=0.5,
                label="_nolegend_",
            )

    time_label_count = 6
    for _t, tau in zip(
        t[:: (len(t)) // time_label_count],
        tau_cmd[:: (len(tau_cmd) // time_label_count)],
    ):
        ax[0].text(
            tau[0],
            tau[1],
            f"$t={_t:4.0f}s$",
            bbox=dict(facecolor="white", alpha=0.5),
            fontsize=10,
            ha="center",
        )

    ax[0].annotate(
        r"$\tau_0$",
        xy=(tau_hist[0, 0, 0], tau_hist[0, 0, 1]),
        xycoords="data",
        xytext=(30, 30),
        bbox=dict(facecolor="white", alpha=0.5),
        textcoords="offset points",
        arrowprops=dict(facecolor="black", arrowstyle="->"),
    )
    ax[0].grid(True)
    ax[0].axis("equal")
    ax[0].legend()
    ax[0].set_xlabel(r"$F_x$ [N]")
    ax[0].set_ylabel(r"$F_y$ [N]")

    ax[1].plot(
        t,
        tau_cmd[:, 5],
        label=r"Input $M_z$",
        linewidth=1.5,
        linestyle="-",
        color="black",
    )

    for j in range(len(tau_hist)):
        ax[1].plot(
            t,
            tau_hist[j][:, 5],
            label=r"Output $M_z$",
            linewidth=1.5,
            linestyle="-",
            color=colors[j],
        )

    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("$M_z$ [Nm]")

    ax[1].grid(True)
    ax[1].legend()

    return fig, ax


def nice_plot_angles(xi_hist, dt=1.0):
    angles = []

    t = np.linspace(0, dt * len(xi_hist[0]), len(xi_hist[0]))

    for xi in xi_hist:
        a = np.empty((len(xi), 2))
        for i, u in enumerate(xi):
            a2 = np.arctan2(u[2], u[1])
            a3 = np.arctan2(u[4], u[3])
            a[i] = np.array([a2])
        angles.append(a)

    for _, angle in enumerate(angles):
        angle[0:3, 0] = None

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i, angle in enumerate(angles):
        ax.plot(t, np.degrees(angle[:, 0]), color=colors[i])

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\alpha_1$ [Deg]")
    ax.grid(True)

    return fig, ax


def nice_plot_the_thing(tau_cmd, tau_alloc, xi_hist, dt=1.0):
    # Plot angles, F_x and M_z

    fig, ax = plt.subplots(3, 1, figsize=(6, 4.5), sharex=True)

    fig.tight_layout(pad=0.0)

    t = np.linspace(0, dt * len(xi_hist[0]), len(xi_hist[0]))

    labels = [r"$F_x$", r"$F_y$", r"$F_z$", r"$M_x$", r"$M_y$", r"$M_z$"]
    units = ["[N]", "[N]", "[N]", "[Nm]", "[Nm]", "[Nm]"]
    line_styles = ["-", "-", "-", "-.", "-.", "--"]

    indices = [0, 5]
    for i in range(len(indices)):
        ax[i].plot(
            t,
            tau_cmd[:, indices[i]],
            linewidth=1.5,
            linestyle="-",
            color="black",
        )

        for j in range(len(tau_alloc)):
            ax[i].plot(
                t,
                tau_alloc[j][:, indices[i]],
                linewidth=1.5,
                linestyle=line_styles[j],
                color=colors[j],
            )

        ax[i].set_ylabel(labels[indices[i]] + " " + units[indices[i]])

        ax[i].grid(True)

    angles = []
    for xi in xi_hist:
        a = np.empty((len(xi), 2))
        for i, u in enumerate(xi):
            a2 = np.arctan2(u[2], u[1])
            a3 = np.arctan2(u[4], u[3])
            a[i] = np.array([a2])
        angles.append(a)

    for _, angle in enumerate(angles):
        angle[0:3, 0] = None

    for i, angle in enumerate(angles):
        ax[2].plot(
            t,
            np.degrees(angle[:, 0]),
            color=colors[i],
            linestyle=line_styles[i],
            linewidth=2.0
        )

    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel(r"$\alpha_1$ [Deg]")
    ax[2].grid(True)

    return fig, ax


def nice_plot_angles_reference_filter(xi_hist, xi_desired_hist, dt=1.0):
    angles = []
    angles_desired = []

    t = np.linspace(0, dt * len(xi_hist[0]), len(xi_hist[0]))

    for xi, xi_d in zip(xi_hist, xi_desired_hist):
        a = np.empty((len(xi), 2))
        a_d = np.empty((len(xi), 2))
        for i, (u, u_d) in enumerate(zip(xi, xi_d)):

            a[i] = np.array([np.arctan2(u[4], u[3])])
            a_d[i] = np.array([np.arctan2(u_d[4], u_d[3])])

        angles.append(a)

        angles_desired.append(a_d)

    for _, angle in enumerate(angles):
        angle[0:3, 0] = None

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i, (angle, angle_desired) in enumerate(zip(angles, angles_desired)):
        ax.plot(t, np.degrees(np.unwrap(angle[:, 0])), color=colors[i])
        ax.plot(
            t, np.degrees(np.unwrap(angle_desired[:, 0])), "-.", color=darker_colors[i]
        )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\alpha_2$ [Deg]")
    ax.grid(True)

    return fig, ax


def nice_plot_thruster_forces(xi_hist, dt=1.0):

    fig, ax = plt.subplots(3, 1, figsize=(8, 8))

    t = np.linspace(0, dt * len(xi_hist[0]), len(xi_hist[0]))

    fig.tight_layout(pad=1.5)
    for i, xi in enumerate(xi_hist):
        F_0 = xi[:, 0]
        F_1 = np.linalg.norm(xi[:, 1:2], axis=1)
        F_2 = np.linalg.norm(xi[:, 2:3], axis=1)
        ax[0].plot(t, F_0, color=colors[i])
        ax[1].plot(t, F_1, color=colors[i])
        ax[2].plot(t, F_2, color=colors[i])

        ax[0].set_ylabel("Tunnel [N]")
        ax[1].set_ylabel("Port [N]")
        ax[2].set_ylabel("Starboard [N]")
        for j in range(3):
            ax[j].set_xlabel("Time [s]")
            ax[j].grid(True)

    return fig, ax


def nice_plot_thruster_forces_reference_filter(xi_hist, xi_desired_hist, dt=1.0):

    fig, ax = plt.subplots(3, 1, figsize=(6, 5), sharex=True)

    t = np.linspace(0, dt * len(xi_hist[0]), len(xi_hist[0]))

    fig.tight_layout(pad=1.5)
    for i, (xi, xi_d) in enumerate(zip(xi_hist, xi_desired_hist)):
        F_0 = xi[:, 0]
        F_1 = np.linalg.norm(xi[:, 1:2], axis=1)
        F_2 = np.linalg.norm(xi[:, 3:4], axis=1)

        F_d0 = xi_d[:, 0]
        F_d1 = np.linalg.norm(xi_d[:, 1:2], axis=1)
        F_d2 = np.linalg.norm(xi_d[:, 3:4], axis=1)

        ax[2].plot(t, F_0, "-", color=darker_colors[i], linewidth=2.0)
        ax[2].plot(t, F_d0, "-.", color=ligther_colors[i])
        ax[0].plot(t, F_1, "-", color=darker_colors[i], linewidth=2.0)
        ax[0].plot(t, F_d1, "-.", color=ligther_colors[i])
        ax[1].plot(t, F_2, "-", color=darker_colors[i], linewidth=2.0)
        ax[1].plot(t, F_d2, "-.", color=ligther_colors[i])

        ax[2].set_ylabel("Tunnel [N]")
        ax[0].set_ylabel("Port [N]")
        ax[1].set_ylabel("Starboard [N]")
        for j in range(3):
            # ax[j].set_xlabel("Time [s]")
            ax[j].grid(True)
        ax[-1].set_xlabel("Time [s]")

    return fig, ax


def nice_plot_theta_histories(theta_hist, dt=1.0):

    fig, ax = plt.subplots(2, 1, figsize=(6, 3), height_ratios=[1, 1], sharex=True)

    t = np.linspace(0, dt * len(theta_hist[0]), len(theta_hist[0]))

    for i in range(theta_hist.shape[0]):
        ax[0].plot(t, theta_hist[i, :, 0], "-", color=colors[i], linewidth=2.0)
    ax[0].set_ylabel(r"$\theta_1$")

    for i in range(theta_hist.shape[0]):
        ax[1].plot(t, theta_hist[i, :, 1], "-", color=colors[i], linewidth=2.0)
    ax[1].set_ylabel(r"$\theta_2$")

    for a in ax:
        a.grid(True)

    ax[-1].set_xlabel("Time [s]")

    return fig, ax
