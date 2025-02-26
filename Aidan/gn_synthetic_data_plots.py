import csv
import matplotlib.pyplot as plt
import numpy as np

JITTER_AMOUNT = 0.005
ALPHA = 1


def add_jitter(values, jitter_amount=JITTER_AMOUNT, cap=False):
    jittered = []
    for v in values:
        new_v = v + np.random.uniform(-jitter_amount, jitter_amount)
        if cap and new_v >= 1:
            new_v = 1
        jittered.append(new_v)
    return jittered


master_results = []
with open("../Tony/benchmark/benchmark_log_FULL_corrected.csv", "r", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        master_results.append(row)

group_to_data = {"SPARSE": [], "INTERMEDIATE": [], "DENSE": []}
for r in master_results:
    group_to_data[r["ScenarioDensity"]].append(r)


def set_dynamic_xlim(axlist, xvals):
    if xvals:
        mn, mx = min(xvals), max(xvals)
        margin = (mx - mn) * 0.05
        for ax in axlist:
            ax.set_xlim(mn - margin, mx + margin)


def set_dynamic_ylim(ax, values):
    if values:
        mn, mx = min(values), max(values)
        mg = (mx - mn) * 0.05
        ax.set_ylim(mn - mg, mx + mg)


for label in ["SPARSE", "INTERMEDIATE", "DENSE"]:
    data = group_to_data[label]

    # Arrays for RING
    x_ring = []
    t_ring_gn, t_ring_gnx = [], []
    m_ring_gn, m_ring_gnx = [], []

    # Arrays for SBM
    x_sbm = []
    t_sbm_gn, t_sbm_gnx = [], []
    m_sbm_gn, m_sbm_gnx = [], []

    for r in data:
        num_edges = float(r["NumEdges"])

        gn_time = float(r["GN_Time"])
        gnx_time = float(r["GNX_Time"])
        gn_mem = float(r["GN_Memory"])
        gnx_mem = float(r["GNX_Memory"])

        if "RING" in r["ScenarioName"]:
            x_ring.append(num_edges)
            t_ring_gn.append(gn_time)
            t_ring_gnx.append(gnx_time)
            m_ring_gn.append(gn_mem)
            m_ring_gnx.append(gnx_mem)
        else:  # SBM
            x_sbm.append(num_edges)
            t_sbm_gn.append(gn_time)
            t_sbm_gnx.append(gnx_time)
            m_sbm_gn.append(gn_mem)
            m_sbm_gnx.append(gnx_mem)

    # Plot RING in one figure
    fig_ring, axs_ring = plt.subplots(1, 2, figsize=(14, 6))
    ax_time_r = axs_ring[0]
    ax_mem_r = axs_ring[1]

    ax_time_r.scatter(
        x_ring, add_jitter(t_ring_gn), marker="o", label="Our Implementation", alpha=ALPHA
    )
    ax_time_r.scatter(
        x_ring, add_jitter(t_ring_gnx), marker="o", label="Network X Implementation", alpha=ALPHA
    )
    ax_time_r.set_title(f"{label} (Ring) - Time")
    ax_time_r.set_xlabel("#Edges")
    ax_time_r.set_ylabel("Time (s)")
    ax_time_r.legend()
    set_dynamic_xlim([ax_time_r], x_ring)
    set_dynamic_ylim(ax_time_r, t_ring_gn + t_ring_gnx)

    ax_mem_r.scatter(x_ring, add_jitter(m_ring_gn), marker="o", label="Our Implementation", alpha=ALPHA)
    ax_mem_r.scatter(
        x_ring, add_jitter(m_ring_gnx), marker="o", label="Network X Implementation", alpha=ALPHA
    )
    ax_mem_r.set_title(f"{label} (Ring) - Memory")
    ax_mem_r.set_xlabel("#Edges")
    ax_mem_r.set_ylabel("Memory (bytes)")
    ax_mem_r.legend()
    set_dynamic_xlim([ax_mem_r], x_ring)
    set_dynamic_ylim(ax_mem_r, m_ring_gn + m_ring_gnx)

    fig_ring.tight_layout()
    fig_ring.savefig(f"z_{label.lower()}_ring_gn_gnx.png")
    plt.close(fig_ring)

    # Plot SBM in another figure
    fig_sbm, axs_sbm = plt.subplots(1, 2, figsize=(14, 6))
    ax_time_s = axs_sbm[0]
    ax_mem_s = axs_sbm[1]

    ax_time_s.scatter(x_sbm, add_jitter(t_sbm_gn), marker="x", label="Our Implementation", alpha=ALPHA)
    ax_time_s.scatter(
        x_sbm, add_jitter(t_sbm_gnx), marker="x", label="Network X Implementation", alpha=ALPHA
    )
    ax_time_s.set_title(f"{label} (SBM) - Time")
    ax_time_s.set_xlabel("#Edges")
    ax_time_s.set_ylabel("Time (s)")
    ax_time_s.legend()
    set_dynamic_xlim([ax_time_s], x_sbm)
    set_dynamic_ylim(ax_time_s, t_sbm_gn + t_sbm_gnx)

    ax_mem_s.scatter(x_sbm, add_jitter(m_sbm_gn), marker="x", label="Our Implementation", alpha=ALPHA)
    ax_mem_s.scatter(x_sbm, add_jitter(m_sbm_gnx), marker="x", label="Network X Implementation", alpha=ALPHA)
    ax_mem_s.set_title(f"{label} (SBM) - Memory")
    ax_mem_s.set_xlabel("#Edges")
    ax_mem_s.set_ylabel("Memory (bytes)")
    ax_mem_s.legend()
    set_dynamic_xlim([ax_mem_s], x_sbm)
    set_dynamic_ylim(ax_mem_s, m_sbm_gn + m_sbm_gnx)

    fig_sbm.tight_layout()
    fig_sbm.savefig(f"z_{label.lower()}_sbm_gn_gnx.png")
    plt.close(fig_sbm)
