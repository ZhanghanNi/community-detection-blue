import csv
import matplotlib.pyplot as plt
import numpy as np

# global parameters
JITTER_AMOUNT = 0.005
ALPHA = 1

VISUALIZE_GN   = True
VISUALIZE_LP   = False
VISUALIZE_ALP  = False
VISUALIZE_LV   = True
VISUALIZE_BVNS = True
VISUALIZE_GNX  = True
VISUALIZE_LNX  = True

def add_jitter(values, jitter_amount=JITTER_AMOUNT, cap=False):
    jittered = []
    for v in values:
        new_v = v + np.random.uniform(-jitter_amount, jitter_amount)
        if cap and new_v >= 1:
            new_v = 1
        jittered.append(new_v)
    return jittered

master_results = []
with open("benchmark_log_FULL_corrected.csv", "r", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        master_results.append(row)

group_to_data = {"SPARSE": [], "INTERMEDIATE": [], "DENSE": []}
for r in master_results:
    group_to_data[r["ScenarioDensity"]].append(r)

for label in ["SPARSE", "INTERMEDIATE", "DENSE"]:
    data = group_to_data[label]

    # Arrays for RING
    x_ring = []
    # Time arrays
    t_ring_gn, t_ring_lp, t_ring_alp = [], [], []
    t_ring_lv, t_ring_bvns, t_ring_gnx, t_ring_lnx = [], [], [], []
    # Memory arrays
    m_ring_gn, m_ring_lp, m_ring_alp = [], [], []
    m_ring_lv, m_ring_bvns, m_ring_gnx, m_ring_lnx = [], [], [], []
    # Modularity arrays
    mod_ring_gn, mod_ring_lp, mod_ring_alp = [], [], []
    mod_ring_lv, mod_ring_bvns, mod_ring_gnx, mod_ring_lnx = [], [], [], []
    # Conductance arrays
    cond_ring_gn, cond_ring_lp, cond_ring_alp = [], [], []
    cond_ring_lv, cond_ring_bvns, cond_ring_gnx, cond_ring_lnx = [], [], [], []
    # ARI arrays
    ari_ring_gn, ari_ring_lp, ari_ring_alp = [], [], []
    ari_ring_lv, ari_ring_bvns, ari_ring_gnx, ari_ring_lnx = [], [], [], []

    # Arrays for SBM
    x_sbm = []
    # Time arrays
    t_sbm_gn, t_sbm_lp, t_sbm_alp = [], [], []
    t_sbm_lv, t_sbm_bvns, t_sbm_gnx, t_sbm_lnx = [], [], [], []
    # Memory arrays
    m_sbm_gn, m_sbm_lp, m_sbm_alp = [], [], []
    m_sbm_lv, m_sbm_bvns, m_sbm_gnx, m_sbm_lnx = [], [], [], []
    # Modularity arrays
    mod_sbm_gn, mod_sbm_lp, mod_sbm_alp = [], [], []
    mod_sbm_lv, mod_sbm_bvns, mod_sbm_gnx, mod_sbm_lnx = [], [], [], []
    # Conductance arrays
    cond_sbm_gn, cond_sbm_lp, cond_sbm_alp = [], [], []
    cond_sbm_lv, cond_sbm_bvns, cond_sbm_gnx, cond_sbm_lnx = [], [], [], []
    # ARI arrays
    ari_sbm_gn, ari_sbm_lp, ari_sbm_alp = [], [], []
    ari_sbm_lv, ari_sbm_bvns, ari_sbm_gnx, ari_sbm_lnx = [], [], [], []

    for r in data:
        num_edges = float(r["NumEdges"])
        gn_time  = float(r["GN_Time"])
        lp_time  = float(r["LP_Time"])
        alp_time = float(r["ALP_Time"])
        lv_time  = float(r["LV_Time"])
        bvns_time = float(r["BVNS_Time"])
        gnx_time = float(r["GNX_Time"])
        lnx_time = float(r["LNX_Time"])

        gn_mem  = float(r["GN_Memory"])
        lp_mem  = float(r["LP_Memory"])
        alp_mem = float(r["ALP_Memory"])
        lv_mem  = float(r["LV_Memory"])
        bvns_mem = float(r["BVNS_Memory"])
        gnx_mem = float(r["GNX_Memory"])
        lnx_mem = float(r["LNX_Memory"])

        gn_mod  = float(r["GN_Modularity"]) if r["GN_Modularity"] != "nan" else float('nan')
        lp_mod  = float(r["LP_Modularity"]) if r["LP_Modularity"] != "nan" else float('nan')
        alp_mod = float(r["ALP_Modularity"]) if r["ALP_Modularity"] != "nan" else float('nan')
        lv_mod  = float(r["LV_Modularity"]) if r["LV_Modularity"] != "nan" else float('nan')
        bvns_mod = float(r["BVNS_Modularity"]) if r["BVNS_Modularity"] != "nan" else float('nan')
        gnx_mod = float(r["GNX_Modularity"]) if r["GNX_Modularity"] != "nan" else float('nan')
        lnx_mod = float(r["LNX_Modularity"]) if r["LNX_Modularity"] != "nan" else float('nan')

        gn_cond  = float(r["GN_Conductance"]) if r["GN_Conductance"] != "nan" else float('nan')
        lp_cond  = float(r["LP_Conductance"]) if r["LP_Conductance"] != "nan" else float('nan')
        alp_cond = float(r["ALP_Conductance"]) if r["ALP_Conductance"] != "nan" else float('nan')
        lv_cond  = float(r["LV_Conductance"]) if r["LV_Conductance"] != "nan" else float('nan')
        bvns_cond = float(r["BVNS_Conductance"]) if r["BVNS_Conductance"] != "nan" else float('nan')
        gnx_cond = float(r["GNX_Conductance"]) if r["GNX_Conductance"] != "nan" else float('nan')
        lnx_cond = float(r["LNX_Conductance"]) if r["LNX_Conductance"] != "nan" else float('nan')

        gn_ari  = float(r["GN_GroundTruth"]) if r["GN_GroundTruth"] != "nan" else float('nan')
        lp_ari  = float(r["LP_GroundTruth"]) if r["LP_GroundTruth"] != "nan" else float('nan')
        alp_ari = float(r["ALP_GroundTruth"]) if r["ALP_GroundTruth"] != "nan" else float('nan')
        lv_ari  = float(r["LV_GroundTruth"]) if r["LV_GroundTruth"] != "nan" else float('nan')
        bvns_ari = float(r["BVNS_GroundTruth"]) if r["BVNS_GroundTruth"] != "nan" else float('nan')
        gnx_ari = float(r["GNX_GroundTruth"]) if r["GNX_GroundTruth"] != "nan" else float('nan')
        lnx_ari = float(r["LNX_GroundTruth"]) if r["LNX_GroundTruth"] != "nan" else float('nan')

        if "RING" in r["ScenarioName"]:
            x_ring.append(num_edges)

            t_ring_gn.append(gn_time)
            t_ring_lp.append(lp_time)
            t_ring_alp.append(alp_time)
            t_ring_lv.append(lv_time)
            t_ring_bvns.append(bvns_time)
            t_ring_gnx.append(gnx_time)
            t_ring_lnx.append(lnx_time)

            m_ring_gn.append(gn_mem)
            m_ring_lp.append(lp_mem)
            m_ring_alp.append(alp_mem)
            m_ring_lv.append(lv_mem)
            m_ring_bvns.append(bvns_mem)
            m_ring_gnx.append(gnx_mem)
            m_ring_lnx.append(lnx_mem)

            mod_ring_gn.append(gn_mod)
            mod_ring_lp.append(lp_mod)
            mod_ring_alp.append(alp_mod)
            mod_ring_lv.append(lv_mod)
            mod_ring_bvns.append(bvns_mod)
            mod_ring_gnx.append(gnx_mod)
            mod_ring_lnx.append(lnx_mod)

            cond_ring_gn.append(gn_cond)
            cond_ring_lp.append(lp_cond)
            cond_ring_alp.append(alp_cond)
            cond_ring_lv.append(lv_cond)
            cond_ring_bvns.append(bvns_cond)
            cond_ring_gnx.append(gnx_cond)
            cond_ring_lnx.append(lnx_cond)

            ari_ring_gn.append(gn_ari)
            ari_ring_lp.append(lp_ari)
            ari_ring_alp.append(alp_ari)
            ari_ring_lv.append(lv_ari)
            ari_ring_bvns.append(bvns_ari)
            ari_ring_gnx.append(gnx_ari)
            ari_ring_lnx.append(lnx_ari)
        else:  # SBM
            x_sbm.append(num_edges)

            t_sbm_gn.append(gn_time)
            t_sbm_lp.append(lp_time)
            t_sbm_alp.append(alp_time)
            t_sbm_lv.append(lv_time)
            t_sbm_bvns.append(bvns_time)
            t_sbm_gnx.append(gnx_time)
            t_sbm_lnx.append(lnx_time)

            m_sbm_gn.append(gn_mem)
            m_sbm_lp.append(lp_mem)
            m_sbm_alp.append(alp_mem)
            m_sbm_lv.append(lv_mem)
            m_sbm_bvns.append(bvns_mem)
            m_sbm_gnx.append(gnx_mem)
            m_sbm_lnx.append(lnx_mem)

            mod_sbm_gn.append(gn_mod)
            mod_sbm_lp.append(lp_mod)
            mod_sbm_alp.append(alp_mod)
            mod_sbm_lv.append(lv_mod)
            mod_sbm_bvns.append(bvns_mod)
            mod_sbm_gnx.append(gnx_mod)
            mod_sbm_lnx.append(lnx_mod)

            cond_sbm_gn.append(gn_cond)
            cond_sbm_lp.append(lp_cond)
            cond_sbm_alp.append(alp_cond)
            cond_sbm_lv.append(lv_cond)
            cond_sbm_bvns.append(bvns_cond)
            cond_sbm_gnx.append(gnx_cond)
            cond_sbm_lnx.append(lnx_cond)

            ari_sbm_gn.append(gn_ari)
            ari_sbm_lp.append(lp_ari)
            ari_sbm_alp.append(alp_ari)
            ari_sbm_lv.append(lv_ari)
            ari_sbm_bvns.append(bvns_ari)
            ari_sbm_gnx.append(gnx_ari)
            ari_sbm_lnx.append(lnx_ari)

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

    # Plot RING in one figure
    fig_ring, axs_ring = plt.subplots(2, 3, figsize=(18, 10))
    ax_time_r = axs_ring[0, 0]
    ax_mem_r  = axs_ring[0, 1]
    ax_mod_r  = axs_ring[1, 2]
    ax_cond_r = axs_ring[1, 0]
    ax_ari_r  = axs_ring[1, 1]

    # Plot time (RING)
    if VISUALIZE_GN:
        ax_time_r.scatter(x_ring, add_jitter(t_ring_gn), marker='o', label='GN', alpha=ALPHA)
    if VISUALIZE_LP:
        ax_time_r.scatter(x_ring, add_jitter(t_ring_lp), marker='o', label='LP', alpha=ALPHA)
    if VISUALIZE_ALP:
        ax_time_r.scatter(x_ring, add_jitter(t_ring_alp), marker='o', label='ALP', alpha=ALPHA)
    if VISUALIZE_LV:
        ax_time_r.scatter(x_ring, add_jitter(t_ring_lv), marker='o', label='LV', alpha=ALPHA)
    if VISUALIZE_BVNS:
        ax_time_r.scatter(x_ring, add_jitter(t_ring_bvns), marker='o', label='BVNS', alpha=ALPHA)
    if VISUALIZE_GNX:
        ax_time_r.scatter(x_ring, add_jitter(t_ring_gnx), marker='o', label='GNX', alpha=ALPHA)
    if VISUALIZE_LNX:
        ax_time_r.scatter(x_ring, add_jitter(t_ring_lnx), marker='o', label='LNX', alpha=ALPHA)
    ax_time_r.set_title(f"{label} (Ring) - Time")
    ax_time_r.set_xlabel("#Edges")
    ax_time_r.set_ylabel("Time (s)")
    ax_time_r.legend()
    time_data = []
    if VISUALIZE_GN:   time_data += t_ring_gn
    if VISUALIZE_LP:   time_data += t_ring_lp
    if VISUALIZE_ALP:  time_data += t_ring_alp
    if VISUALIZE_LV:   time_data += t_ring_lv
    if VISUALIZE_BVNS: time_data += t_ring_bvns
    if VISUALIZE_GNX:  time_data += t_ring_gnx
    if VISUALIZE_LNX:  time_data += t_ring_lnx
    set_dynamic_xlim([ax_time_r], x_ring)
    set_dynamic_ylim(ax_time_r, time_data)

    # Plot memory (RING)
    if VISUALIZE_GN:
        ax_mem_r.scatter(x_ring, add_jitter(m_ring_gn), marker='o', label='GN', alpha=ALPHA)
    if VISUALIZE_LP:
        ax_mem_r.scatter(x_ring, add_jitter(m_ring_lp), marker='o', label='LP', alpha=ALPHA)
    if VISUALIZE_ALP:
        ax_mem_r.scatter(x_ring, add_jitter(m_ring_alp), marker='o', label='ALP', alpha=ALPHA)
    if VISUALIZE_LV:
        ax_mem_r.scatter(x_ring, add_jitter(m_ring_lv), marker='o', label='LV', alpha=ALPHA)
    if VISUALIZE_BVNS:
        ax_mem_r.scatter(x_ring, add_jitter(m_ring_bvns), marker='o', label='BVNS', alpha=ALPHA)
    if VISUALIZE_GNX:
        ax_mem_r.scatter(x_ring, add_jitter(m_ring_gnx), marker='o', label='GNX', alpha=ALPHA)
    if VISUALIZE_LNX:
        ax_mem_r.scatter(x_ring, add_jitter(m_ring_lnx), marker='o', label='LNX', alpha=ALPHA)
    ax_mem_r.set_title(f"{label} (Ring) - Memory")
    ax_mem_r.set_xlabel("#Edges")
    ax_mem_r.set_ylabel("Memory (bytes)")
    ax_mem_r.legend()
    mem_data = []
    if VISUALIZE_GN:   mem_data += m_ring_gn
    if VISUALIZE_LP:   mem_data += m_ring_lp
    if VISUALIZE_ALP:  mem_data += m_ring_alp
    if VISUALIZE_LV:   mem_data += m_ring_lv
    if VISUALIZE_BVNS: mem_data += m_ring_bvns
    if VISUALIZE_GNX:  mem_data += m_ring_gnx
    if VISUALIZE_LNX:  mem_data += m_ring_lnx
    set_dynamic_xlim([ax_mem_r], x_ring)
    set_dynamic_ylim(ax_mem_r, mem_data)

    # Plot modularity (RING)
    if VISUALIZE_GN:
        ax_mod_r.scatter(x_ring, add_jitter(mod_ring_gn), marker='o', label='GN', alpha=ALPHA)
    if VISUALIZE_LP:
        ax_mod_r.scatter(x_ring, add_jitter(mod_ring_lp), marker='o', label='LP', alpha=ALPHA)
    if VISUALIZE_ALP:
        ax_mod_r.scatter(x_ring, add_jitter(mod_ring_alp), marker='o', label='ALP', alpha=ALPHA)
    if VISUALIZE_LV:
        ax_mod_r.scatter(x_ring, add_jitter(mod_ring_lv), marker='o', label='LV', alpha=ALPHA)
    if VISUALIZE_BVNS:
        ax_mod_r.scatter(x_ring, add_jitter(mod_ring_bvns), marker='o', label='BVNS', alpha=ALPHA)
    if VISUALIZE_GNX:
        ax_mod_r.scatter(x_ring, add_jitter(mod_ring_gnx), marker='o', label='GNX', alpha=ALPHA)
    if VISUALIZE_LNX:
        ax_mod_r.scatter(x_ring, add_jitter(mod_ring_lnx), marker='o', label='LNX', alpha=ALPHA)
    ax_mod_r.set_title(f"{label} (Ring) - Modularity")
    ax_mod_r.set_xlabel("#Edges")
    ax_mod_r.set_ylabel("Modularity")
    ax_mod_r.legend()
    mod_data = []
    if VISUALIZE_GN:   mod_data += mod_ring_gn
    if VISUALIZE_LP:   mod_data += mod_ring_lp
    if VISUALIZE_ALP:  mod_data += mod_ring_alp
    if VISUALIZE_LV:   mod_data += mod_ring_lv
    if VISUALIZE_BVNS: mod_data += mod_ring_bvns
    if VISUALIZE_GNX:  mod_data += mod_ring_gnx
    if VISUALIZE_LNX:  mod_data += mod_ring_lnx
    set_dynamic_xlim([ax_mod_r], x_ring)
    set_dynamic_ylim(ax_mod_r, mod_data)

    # Plot conductance (RING)
    if VISUALIZE_GN:
        ax_cond_r.scatter(x_ring, add_jitter(cond_ring_gn), marker='o', label='GN', alpha=ALPHA)
    if VISUALIZE_LP:
        ax_cond_r.scatter(x_ring, add_jitter(cond_ring_lp), marker='o', label='LP', alpha=ALPHA)
    if VISUALIZE_ALP:
        ax_cond_r.scatter(x_ring, add_jitter(cond_ring_alp), marker='o', label='ALP', alpha=ALPHA)
    if VISUALIZE_LV:
        ax_cond_r.scatter(x_ring, add_jitter(cond_ring_lv), marker='o', label='LV', alpha=ALPHA)
    if VISUALIZE_BVNS:
        ax_cond_r.scatter(x_ring, add_jitter(cond_ring_bvns), marker='o', label='BVNS', alpha=ALPHA)
    if VISUALIZE_GNX:
        ax_cond_r.scatter(x_ring, add_jitter(cond_ring_gnx), marker='o', label='GNX', alpha=ALPHA)
    if VISUALIZE_LNX:
        ax_cond_r.scatter(x_ring, add_jitter(cond_ring_lnx), marker='o', label='LNX', alpha=ALPHA)
    ax_cond_r.set_title(f"{label} (Ring) - Conductance")
    ax_cond_r.set_xlabel("#Edges")
    ax_cond_r.set_ylabel("Conductance")
    ax_cond_r.legend()
    cond_data = []
    if VISUALIZE_GN:   cond_data += cond_ring_gn
    if VISUALIZE_LP:   cond_data += cond_ring_lp
    if VISUALIZE_ALP:  cond_data += cond_ring_alp
    if VISUALIZE_LV:   cond_data += cond_ring_lv
    if VISUALIZE_BVNS: cond_data += cond_ring_bvns
    if VISUALIZE_GNX:  cond_data += cond_ring_gnx
    if VISUALIZE_LNX:  cond_data += cond_ring_lnx
    set_dynamic_xlim([ax_cond_r], x_ring)
    set_dynamic_ylim(ax_cond_r, cond_data)

    # Plot ARI (RING)
    if VISUALIZE_GN:
        ax_ari_r.scatter(x_ring, add_jitter(ari_ring_gn), marker='o', label='GN', alpha=ALPHA)
    if VISUALIZE_LP:
        ax_ari_r.scatter(x_ring, add_jitter(ari_ring_lp), marker='o', label='LP', alpha=ALPHA)
    if VISUALIZE_ALP:
        ax_ari_r.scatter(x_ring, add_jitter(ari_ring_alp), marker='o', label='ALP', alpha=ALPHA)
    if VISUALIZE_LV:
        ax_ari_r.scatter(x_ring, add_jitter(ari_ring_lv), marker='o', label='LV', alpha=ALPHA)
    if VISUALIZE_BVNS:
        ax_ari_r.scatter(x_ring, add_jitter(ari_ring_bvns), marker='o', label='BVNS', alpha=ALPHA)
    if VISUALIZE_GNX:
        ax_ari_r.scatter(x_ring, add_jitter(ari_ring_gnx), marker='o', label='GNX', alpha=ALPHA)
    if VISUALIZE_LNX:
        ax_ari_r.scatter(x_ring, add_jitter(ari_ring_lnx), marker='o', label='LNX', alpha=ALPHA)
    ax_ari_r.set_title(f"{label} (Ring) - ARI")
    ax_ari_r.set_xlabel("#Edges")
    ax_ari_r.set_ylabel("ARI")
    ax_ari_r.legend()
    ari_data = []
    if VISUALIZE_GN:   ari_data += ari_ring_gn
    if VISUALIZE_LP:   ari_data += ari_ring_lp
    if VISUALIZE_ALP:  ari_data += ari_ring_alp
    if VISUALIZE_LV:   ari_data += ari_ring_lv
    if VISUALIZE_BVNS: ari_data += ari_ring_bvns
    if VISUALIZE_GNX:  ari_data += ari_ring_gnx
    if VISUALIZE_LNX:  ari_data += ari_ring_lnx
    set_dynamic_xlim([ax_ari_r], x_ring)
    set_dynamic_ylim(ax_ari_r, ari_data)

    fig_ring.tight_layout()
    fig_ring.savefig(f"s_{label.lower()}_ring.png")
    plt.close(fig_ring)

    # Plot SBM in another figure
    fig_sbm, axs_sbm = plt.subplots(2, 3, figsize=(18, 10))
    ax_time_s = axs_sbm[0, 0]
    ax_mem_s  = axs_sbm[0, 1]
    ax_mod_s  = axs_sbm[1, 2]
    ax_cond_s = axs_sbm[1, 0]
    ax_ari_s  = axs_sbm[1, 1]

    # Plot time (SBM)
    if VISUALIZE_GN:
        ax_time_s.scatter(x_sbm, add_jitter(t_sbm_gn), marker='x', label='GN', alpha=ALPHA)
    if VISUALIZE_LP:
        ax_time_s.scatter(x_sbm, add_jitter(t_sbm_lp), marker='x', label='LP', alpha=ALPHA)
    if VISUALIZE_ALP:
        ax_time_s.scatter(x_sbm, add_jitter(t_sbm_alp), marker='x', label='ALP', alpha=ALPHA)
    if VISUALIZE_LV:
        ax_time_s.scatter(x_sbm, add_jitter(t_sbm_lv), marker='x', label='LV', alpha=ALPHA)
    if VISUALIZE_BVNS:
        ax_time_s.scatter(x_sbm, add_jitter(t_sbm_bvns), marker='x', label='BVNS', alpha=ALPHA)
    if VISUALIZE_GNX:
        ax_time_s.scatter(x_sbm, add_jitter(t_sbm_gnx), marker='x', label='GNX', alpha=ALPHA)
    if VISUALIZE_LNX:
        ax_time_s.scatter(x_sbm, add_jitter(t_sbm_lnx), marker='x', label='LNX', alpha=ALPHA)
    ax_time_s.set_title(f"{label} (SBM) - Time")
    ax_time_s.set_xlabel("#Edges")
    ax_time_s.set_ylabel("Time (s)")
    ax_time_s.legend()
    time_data_s = []
    if VISUALIZE_GN:   time_data_s += t_sbm_gn
    if VISUALIZE_LP:   time_data_s += t_sbm_lp
    if VISUALIZE_ALP:  time_data_s += t_sbm_alp
    if VISUALIZE_LV:   time_data_s += t_sbm_lv
    if VISUALIZE_BVNS: time_data_s += t_sbm_bvns
    if VISUALIZE_GNX:  time_data_s += t_sbm_gnx
    if VISUALIZE_LNX:  time_data_s += t_sbm_lnx
    set_dynamic_xlim([ax_time_s], x_sbm)
    set_dynamic_ylim(ax_time_s, time_data_s)

    # Plot memory (SBM)
    if VISUALIZE_GN:
        ax_mem_s.scatter(x_sbm, add_jitter(m_sbm_gn), marker='x', label='GN', alpha=ALPHA)
    if VISUALIZE_LP:
        ax_mem_s.scatter(x_sbm, add_jitter(m_sbm_lp), marker='x', label='LP', alpha=ALPHA)
    if VISUALIZE_ALP:
        ax_mem_s.scatter(x_sbm, add_jitter(m_sbm_alp), marker='x', label='ALP', alpha=ALPHA)
    if VISUALIZE_LV:
        ax_mem_s.scatter(x_sbm, add_jitter(m_sbm_lv), marker='x', label='LV', alpha=ALPHA)
    if VISUALIZE_BVNS:
        ax_mem_s.scatter(x_sbm, add_jitter(m_sbm_bvns), marker='x', label='BVNS', alpha=ALPHA)
    if VISUALIZE_GNX:
        ax_mem_s.scatter(x_sbm, add_jitter(m_sbm_gnx), marker='x', label='GNX', alpha=ALPHA)
    if VISUALIZE_LNX:
        ax_mem_s.scatter(x_sbm, add_jitter(m_sbm_lnx), marker='x', label='LNX', alpha=ALPHA)
    ax_mem_s.set_title(f"{label} (SBM) - Memory")
    ax_mem_s.set_xlabel("#Edges")
    ax_mem_s.set_ylabel("Memory (bytes)")
    ax_mem_s.legend()
    mem_data_s = []
    if VISUALIZE_GN:   mem_data_s += m_sbm_gn
    if VISUALIZE_LP:   mem_data_s += m_sbm_lp
    if VISUALIZE_ALP:  mem_data_s += m_sbm_alp
    if VISUALIZE_LV:   mem_data_s += m_sbm_lv
    if VISUALIZE_BVNS: mem_data_s += m_sbm_bvns
    if VISUALIZE_GNX:  mem_data_s += m_sbm_gnx
    if VISUALIZE_LNX:  mem_data_s += m_sbm_lnx
    set_dynamic_xlim([ax_mem_s], x_sbm)
    set_dynamic_ylim(ax_mem_s, mem_data_s)

    # Plot modularity (SBM)
    if VISUALIZE_GN:
        ax_mod_s.scatter(x_sbm, add_jitter(mod_sbm_gn), marker='x', label='GN', alpha=ALPHA)
    if VISUALIZE_LP:
        ax_mod_s.scatter(x_sbm, add_jitter(mod_sbm_lp), marker='x', label='LP', alpha=ALPHA)
    if VISUALIZE_ALP:
        ax_mod_s.scatter(x_sbm, add_jitter(mod_sbm_alp), marker='x', label='ALP', alpha=ALPHA)
    if VISUALIZE_LV:
        ax_mod_s.scatter(x_sbm, add_jitter(mod_sbm_lv), marker='x', label='LV', alpha=ALPHA)
    if VISUALIZE_BVNS:
        ax_mod_s.scatter(x_sbm, add_jitter(mod_sbm_bvns), marker='x', label='BVNS', alpha=ALPHA)
    if VISUALIZE_GNX:
        ax_mod_s.scatter(x_sbm, add_jitter(mod_sbm_gnx), marker='x', label='GNX', alpha=ALPHA)
    if VISUALIZE_LNX:
        ax_mod_s.scatter(x_sbm, add_jitter(mod_sbm_lnx), marker='x', label='LNX', alpha=ALPHA)
    ax_mod_s.set_title(f"{label} (SBM) - Modularity")
    ax_mod_s.set_xlabel("#Edges")
    ax_mod_s.set_ylabel("Modularity")
    ax_mod_s.legend()
    mod_data_s = []
    if VISUALIZE_GN:   mod_data_s += mod_sbm_gn
    if VISUALIZE_LP:   mod_data_s += mod_sbm_lp
    if VISUALIZE_ALP:  mod_data_s += mod_sbm_alp
    if VISUALIZE_LV:   mod_data_s += mod_sbm_lv
    if VISUALIZE_BVNS: mod_data_s += mod_sbm_bvns
    if VISUALIZE_GNX:  mod_data_s += mod_sbm_gnx
    if VISUALIZE_LNX:  mod_data_s += mod_sbm_lnx
    set_dynamic_xlim([ax_mod_s], x_sbm)
    set_dynamic_ylim(ax_mod_s, mod_data_s)

    # Plot conductance (SBM)
    if VISUALIZE_GN:
        ax_cond_s.scatter(x_sbm, add_jitter(cond_sbm_gn), marker='x', label='GN', alpha=ALPHA)
    if VISUALIZE_LP:
        ax_cond_s.scatter(x_sbm, add_jitter(cond_sbm_lp), marker='x', label='LP', alpha=ALPHA)
    if VISUALIZE_ALP:
        ax_cond_s.scatter(x_sbm, add_jitter(cond_sbm_alp), marker='x', label='ALP', alpha=ALPHA)
    if VISUALIZE_LV:
        ax_cond_s.scatter(x_sbm, add_jitter(cond_sbm_lv), marker='x', label='LV', alpha=ALPHA)
    if VISUALIZE_BVNS:
        ax_cond_s.scatter(x_sbm, add_jitter(cond_sbm_bvns), marker='x', label='BVNS', alpha=ALPHA)
    if VISUALIZE_GNX:
        ax_cond_s.scatter(x_sbm, add_jitter(cond_sbm_gnx), marker='x', label='GNX', alpha=ALPHA)
    if VISUALIZE_LNX:
        ax_cond_s.scatter(x_sbm, add_jitter(cond_sbm_lnx), marker='x', label='LNX', alpha=ALPHA)
    ax_cond_s.set_title(f"{label} (SBM) - Conductance")
    ax_cond_s.set_xlabel("#Edges")
    ax_cond_s.set_ylabel("Conductance")
    ax_cond_s.legend()
    cond_data_s = []
    if VISUALIZE_GN:   cond_data_s += cond_sbm_gn
    if VISUALIZE_LP:   cond_data_s += cond_sbm_lp
    if VISUALIZE_ALP:  cond_data_s += cond_sbm_alp
    if VISUALIZE_LV:   cond_data_s += cond_sbm_lv
    if VISUALIZE_BVNS: cond_data_s += cond_sbm_bvns
    if VISUALIZE_GNX:  cond_data_s += cond_sbm_gnx
    if VISUALIZE_LNX:  cond_data_s += cond_sbm_lnx
    set_dynamic_xlim([ax_cond_s], x_sbm)
    set_dynamic_ylim(ax_cond_s, cond_data_s)

    # Plot ARI (SBM)
    if VISUALIZE_GN:
        ax_ari_s.scatter(x_sbm, add_jitter(ari_sbm_gn), marker='x', label='GN', alpha=ALPHA)
    if VISUALIZE_LP:
        ax_ari_s.scatter(x_sbm, add_jitter(ari_sbm_lp), marker='x', label='LP', alpha=ALPHA)
    if VISUALIZE_ALP:
        ax_ari_s.scatter(x_sbm, add_jitter(ari_sbm_alp), marker='x', label='ALP', alpha=ALPHA)
    if VISUALIZE_LV:
        ax_ari_s.scatter(x_sbm, add_jitter(ari_sbm_lv), marker='x', label='LV', alpha=ALPHA)
    if VISUALIZE_BVNS:
        ax_ari_s.scatter(x_sbm, add_jitter(ari_sbm_bvns), marker='x', label='BVNS', alpha=ALPHA)
    if VISUALIZE_GNX:
        ax_ari_s.scatter(x_sbm, add_jitter(ari_sbm_gnx), marker='x', label='GNX', alpha=ALPHA)
    if VISUALIZE_LNX:
        ax_ari_s.scatter(x_sbm, add_jitter(ari_sbm_lnx), marker='x', label='LNX', alpha=ALPHA)
    ax_ari_s.set_title(f"{label} (SBM) - ARI")
    ax_ari_s.set_xlabel("#Edges")
    ax_ari_s.set_ylabel("ARI")
    ax_ari_s.legend()
    ari_data_s = []
    if VISUALIZE_GN:   ari_data_s += ari_sbm_gn
    if VISUALIZE_LP:   ari_data_s += ari_sbm_lp
    if VISUALIZE_ALP:  ari_data_s += ari_sbm_alp
    if VISUALIZE_LV:   ari_data_s += ari_sbm_lv
    if VISUALIZE_BVNS: ari_data_s += ari_sbm_bvns
    if VISUALIZE_GNX:  ari_data_s += ari_sbm_gnx
    if VISUALIZE_LNX:  ari_data_s += ari_sbm_lnx
    set_dynamic_xlim([ax_ari_s], x_sbm)
    set_dynamic_ylim(ax_ari_s, ari_data_s)

    fig_sbm.tight_layout()
    fig_sbm.savefig(f"s_{label.lower()}_sbm.png")
    plt.close(fig_sbm)