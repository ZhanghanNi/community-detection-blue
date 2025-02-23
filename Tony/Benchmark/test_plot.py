import csv
import matplotlib.pyplot as plt

master_results = []
with open("benchmark_log_withNX.csv", "r", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        master_results.append(row)

group_to_data = {"SPARSE": [], "INTERMEDIATE": [], "DENSE": []}
for r in master_results:
    group_to_data[r["ScenarioDensity"]].append(r)


for label in ["SPARSE", "INTERMEDIATE", "DENSE"]:
    data = group_to_data[label]

    # Arrays for Ring
    x_ring = []
    t_ring_gn, t_ring_lp, t_ring_alp, t_ring_lv, t_ring_bvns = [], [], [], [], []
    m_ring_gn, m_ring_lp, m_ring_alp, m_ring_lv, m_ring_bvns = [], [], [], [], []
    mod_ring_gn, mod_ring_lp, mod_ring_alp, mod_ring_lv, mod_ring_bvns = [], [], [], [], []
    cond_ring_gn, cond_ring_lp, cond_ring_alp, cond_ring_lv, cond_ring_bvns = [], [], [], [], []
    ari_ring_gn, ari_ring_lp, ari_ring_alp, ari_ring_lv, ari_ring_bvns = [], [], [], [], []

    # Arrays for SBM
    x_sbm = []
    t_sbm_gn, t_sbm_lp, t_sbm_alp, t_sbm_lv, t_sbm_bvns = [], [], [], [], []
    m_sbm_gn, m_sbm_lp, m_sbm_alp, m_sbm_lv, m_sbm_bvns = [], [], [], [], []
    mod_sbm_gn, mod_sbm_lp, mod_sbm_alp, mod_sbm_lv, mod_sbm_bvns = [], [], [], [], []
    cond_sbm_gn, cond_sbm_lp, cond_sbm_alp, cond_sbm_lv, cond_sbm_bvns = [], [], [], [], []
    ari_sbm_gn, ari_sbm_lp, ari_sbm_alp, ari_sbm_lv, ari_sbm_bvns = [], [], [], [], []

    for r in data:
        num_edges = float(r["NumEdges"])
        gn_time = float(r["GN_Time"])
        lp_time = float(r["LP_Time"])
        alp_time = float(r["ALP_Time"])
        lv_time = float(r["LV_Time"])
        bvns_time = float(r["BVNS_Time"])

        gn_mem = float(r["GN_Memory"])
        lp_mem = float(r["LP_Memory"])
        alp_mem = float(r["ALP_Memory"])
        lv_mem = float(r["LV_Memory"])
        bvns_mem = float(r["BVNS_Memory"])

        gn_mod = float(r["GN_Modularity"]) if r["GN_Modularity"] != "nan" else float('nan')
        lp_mod = float(r["LP_Modularity"]) if r["LP_Modularity"] != "nan" else float('nan')
        alp_mod = float(r["ALP_Modularity"]) if r["ALP_Modularity"] != "nan" else float('nan')
        lv_mod = float(r["LV_Modularity"]) if r["LV_Modularity"] != "nan" else float('nan')
        bvns_mod = float(r["BVNS_Modularity"]) if r["BVNS_Modularity"] != "nan" else float('nan')

        gn_cond = float(r["GN_Conductance"]) if r["GN_Conductance"] != "nan" else float('nan')
        lp_cond = float(r["LP_Conductance"]) if r["LP_Conductance"] != "nan" else float('nan')
        alp_cond = float(r["ALP_Conductance"]) if r["ALP_Conductance"] != "nan" else float('nan')
        lv_cond = float(r["LV_Conductance"]) if r["LV_Conductance"] != "nan" else float('nan')
        bvns_cond = float(r["BVNS_Conductance"]) if r["BVNS_Conductance"] != "nan" else float('nan')

        gn_ari = float(r["GN_GroundTruth"]) if r["GN_GroundTruth"] != "nan" else float('nan')
        lp_ari = float(r["LP_GroundTruth"]) if r["LP_GroundTruth"] != "nan" else float('nan')
        alp_ari = float(r["ALP_GroundTruth"]) if r["ALP_Conductance"] != "nan" else float('nan')
        lv_ari = float(r["LV_GroundTruth"]) if r["LV_GroundTruth"] != "nan" else float('nan')
        bvns_ari = float(r["BVNS_GroundTruth"]) if r["BVNS_GroundTruth"] != "nan" else float('nan')

        if "RING" in r["ScenarioName"]:
            x_ring.append(num_edges)

            t_ring_gn.append(gn_time)
            t_ring_lp.append(lp_time)
            t_ring_alp.append(alp_time)
            t_ring_lv.append(lv_time)
            t_ring_bvns.append(bvns_time)

            m_ring_gn.append(gn_mem)
            m_ring_lp.append(lp_mem)
            m_ring_alp.append(alp_mem)
            m_ring_lv.append(lv_mem)
            m_ring_bvns.append(bvns_mem)

            mod_ring_gn.append(gn_mod)
            mod_ring_lp.append(lp_mod)
            mod_ring_alp.append(alp_mod)
            mod_ring_lv.append(lv_mod)
            mod_ring_bvns.append(bvns_mod)

            cond_ring_gn.append(gn_cond)
            cond_ring_lp.append(lp_cond)
            cond_ring_alp.append(alp_cond)
            cond_ring_lv.append(lv_cond)
            cond_ring_bvns.append(bvns_cond)

            ari_ring_gn.append(gn_ari)
            ari_ring_lp.append(lp_ari)
            ari_ring_alp.append(alp_ari)
            ari_ring_lv.append(lv_ari)
            ari_ring_bvns.append(bvns_ari)
        else:  # SBM
            x_sbm.append(num_edges)

            t_sbm_gn.append(gn_time)
            t_sbm_lp.append(lp_time)
            t_sbm_alp.append(alp_time)
            t_sbm_lv.append(lv_time)
            t_sbm_bvns.append(bvns_time)

            m_sbm_gn.append(gn_mem)
            m_sbm_lp.append(lp_mem)
            m_sbm_alp.append(alp_mem)
            m_sbm_lv.append(lv_mem)
            m_sbm_bvns.append(bvns_mem)

            mod_sbm_gn.append(gn_mod)
            mod_sbm_lp.append(lp_mod)
            mod_sbm_alp.append(alp_mod)
            mod_sbm_lv.append(lv_mod)
            mod_sbm_bvns.append(bvns_mod)

            cond_sbm_gn.append(gn_cond)
            cond_sbm_lp.append(lp_cond)
            cond_sbm_alp.append(alp_cond)
            cond_sbm_lv.append(lv_cond)
            cond_sbm_bvns.append(bvns_cond)

            ari_sbm_gn.append(gn_ari)
            ari_sbm_lp.append(lp_ari)
            ari_sbm_alp.append(alp_ari)
            ari_sbm_lv.append(lv_ari)
            ari_sbm_bvns.append(bvns_ari)

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
    ax_time_r = axs_ring[0,0]
    ax_mem_r = axs_ring[0,1]
    ax_mod_r = axs_ring[0,2]
    ax_cond_r = axs_ring[1,0]
    ax_ari_r = axs_ring[1,1]

    ax_time_r.scatter(x_ring, t_ring_gn, marker='o', label='GN')
    ax_time_r.scatter(x_ring, t_ring_lp, marker='o', label='LP')
    ax_time_r.scatter(x_ring, t_ring_alp, marker='o', label='ALP')
    ax_time_r.scatter(x_ring, t_ring_lv, marker='o', label='LV')
    ax_time_r.scatter(x_ring, t_ring_bvns, marker='o', label='BVNS')
    ax_time_r.set_title(f"{label} (Ring) - Time")
    ax_time_r.set_xlabel("#Edges")
    ax_time_r.set_ylabel("Time (s)")
    ax_time_r.legend()
    set_dynamic_xlim([ax_time_r], x_ring)
    set_dynamic_ylim(ax_time_r, t_ring_gn + t_ring_lp + t_ring_alp + t_ring_lv + t_ring_bvns)

    ax_mem_r.scatter(x_ring, m_ring_gn, marker='o', label='GN')
    ax_mem_r.scatter(x_ring, m_ring_lp, marker='o', label='LP')
    ax_mem_r.scatter(x_ring, m_ring_alp, marker='o', label='ALP')
    ax_mem_r.scatter(x_ring, m_ring_lv, marker='o', label='LV')
    ax_mem_r.scatter(x_ring, m_ring_bvns, marker='o', label='BVNS')
    ax_mem_r.set_title(f"{label} (Ring) - Memory")
    ax_mem_r.set_xlabel("#Edges")
    ax_mem_r.set_ylabel("Memory (bytes)")
    ax_mem_r.legend()
    set_dynamic_xlim([ax_mem_r], x_ring)
    set_dynamic_ylim(ax_mem_r, m_ring_gn + m_ring_lp + m_ring_alp + m_ring_lv + m_ring_bvns)

    ax_mod_r.scatter(x_ring, mod_ring_gn, marker='o', label='GN')
    ax_mod_r.scatter(x_ring, mod_ring_lp, marker='o', label='LP')
    ax_mod_r.scatter(x_ring, mod_ring_alp, marker='o', label='ALP')
    ax_mod_r.scatter(x_ring, mod_ring_lv, marker='o', label='LV')
    ax_mod_r.scatter(x_ring, mod_ring_bvns, marker='o', label='BVNS')
    ax_mod_r.set_title(f"{label} (Ring) - Modularity")
    ax_mod_r.set_xlabel("#Edges")
    ax_mod_r.set_ylabel("Modularity")
    ax_mod_r.legend()
    set_dynamic_xlim([ax_mod_r], x_ring)
    set_dynamic_ylim(ax_mod_r, mod_ring_gn + mod_ring_lp + mod_ring_alp + mod_ring_lv + mod_ring_bvns)

    ax_cond_r.scatter(x_ring, cond_ring_gn, marker='o', label='GN')
    ax_cond_r.scatter(x_ring, cond_ring_lp, marker='o', label='LP')
    ax_cond_r.scatter(x_ring, cond_ring_alp, marker='o', label='ALP')
    ax_cond_r.scatter(x_ring, cond_ring_lv, marker='o', label='LV')
    ax_cond_r.scatter(x_ring, cond_ring_bvns, marker='o', label='BVNS')
    ax_cond_r.set_title(f"{label} (Ring) - Conductance")
    ax_cond_r.set_xlabel("#Edges")
    ax_cond_r.set_ylabel("Conductance")
    ax_cond_r.legend()
    set_dynamic_xlim([ax_cond_r], x_ring)
    set_dynamic_ylim(ax_cond_r, cond_ring_gn + cond_ring_lp + cond_ring_alp + cond_ring_lv + cond_ring_bvns)

    ax_ari_r.scatter(x_ring, ari_ring_gn, marker='o', label='GN')
    ax_ari_r.scatter(x_ring, ari_ring_lp, marker='o', label='LP')
    ax_ari_r.scatter(x_ring, ari_ring_alp, marker='o', label='ALP')
    ax_ari_r.scatter(x_ring, ari_ring_lv, marker='o', label='LV')
    ax_ari_r.scatter(x_ring, ari_ring_bvns, marker='o', label='BVNS')
    ax_ari_r.set_title(f"{label} (Ring) - ARI")
    ax_ari_r.set_xlabel("#Edges")
    ax_ari_r.set_ylabel("ARI")
    ax_ari_r.legend()
    set_dynamic_xlim([ax_ari_r], x_ring)
    set_dynamic_ylim(ax_ari_r, ari_ring_gn + ari_ring_lp + ari_ring_alp + ari_ring_lv + ari_ring_bvns)

    fig_ring.tight_layout()
    fig_ring.savefig(f"{label.lower()}_ring.png")
    plt.close(fig_ring)

    # Plot SBM in another figure
    fig_sbm, axs_sbm = plt.subplots(2, 3, figsize=(18, 10))
    ax_time_s = axs_sbm[0,0]
    ax_mem_s = axs_sbm[0,1]
    ax_mod_s = axs_sbm[0,2]
    ax_cond_s = axs_sbm[1,0]
    ax_ari_s = axs_sbm[1,1]

    ax_time_s.scatter(x_sbm, t_sbm_gn, marker='x', label='GN')
    ax_time_s.scatter(x_sbm, t_sbm_lp, marker='x', label='LP')
    ax_time_s.scatter(x_sbm, t_sbm_alp, marker='x', label='ALP')
    ax_time_s.scatter(x_sbm, t_sbm_lv, marker='x', label='LV')
    ax_time_s.scatter(x_sbm, t_sbm_bvns, marker='x', label='BVNS')
    ax_time_s.set_title(f"{label} (SBM) - Time")
    ax_time_s.set_xlabel("#Edges")
    ax_time_s.set_ylabel("Time (s)")
    ax_time_s.legend()
    set_dynamic_xlim([ax_time_s], x_sbm)
    set_dynamic_ylim(ax_time_s, t_sbm_gn + t_sbm_lp + t_sbm_alp + t_sbm_lv + t_sbm_bvns)

    ax_mem_s.scatter(x_sbm, m_sbm_gn, marker='x', label='GN')
    ax_mem_s.scatter(x_sbm, m_sbm_lp, marker='x', label='LP')
    ax_mem_s.scatter(x_sbm, m_sbm_alp, marker='x', label='ALP')
    ax_mem_s.scatter(x_sbm, m_sbm_lv, marker='x', label='LV')
    ax_mem_s.scatter(x_sbm, m_sbm_bvns, marker='x', label='BVNS')
    ax_mem_s.set_title(f"{label} (SBM) - Memory")
    ax_mem_s.set_xlabel("#Edges")
    ax_mem_s.set_ylabel("Memory (bytes)")
    ax_mem_s.legend()
    set_dynamic_xlim([ax_mem_s], x_sbm)
    set_dynamic_ylim(ax_mem_s, m_sbm_gn + m_sbm_lp + m_sbm_alp + m_sbm_lv + m_sbm_bvns)

    ax_mod_s.scatter(x_sbm, mod_sbm_gn, marker='x', label='GN')
    ax_mod_s.scatter(x_sbm, mod_sbm_lp, marker='x', label='LP')
    ax_mod_s.scatter(x_sbm, mod_sbm_alp, marker='x', label='ALP')
    ax_mod_s.scatter(x_sbm, mod_sbm_lv, marker='x', label='LV')
    ax_mod_s.scatter(x_sbm, mod_sbm_bvns, marker='x', label='BVNS')
    ax_mod_s.set_title(f"{label} (SBM) - Modularity")
    ax_mod_s.set_xlabel("#Edges")
    ax_mod_s.set_ylabel("Modularity")
    ax_mod_s.legend()
    set_dynamic_xlim([ax_mod_s], x_sbm)
    set_dynamic_ylim(ax_mod_s, mod_sbm_gn + mod_sbm_lp + mod_sbm_alp + mod_sbm_lv + mod_sbm_bvns)

    ax_cond_s.scatter(x_sbm, cond_sbm_gn, marker='x', label='GN')
    ax_cond_s.scatter(x_sbm, cond_sbm_lp, marker='x', label='LP')
    ax_cond_s.scatter(x_sbm, cond_sbm_alp, marker='x', label='ALP')
    ax_cond_s.scatter(x_sbm, cond_sbm_lv, marker='x', label='LV')
    ax_cond_s.scatter(x_sbm, cond_sbm_bvns, marker='x', label='BVNS')
    ax_cond_s.set_title(f"{label} (SBM) - Conductance")
    ax_cond_s.set_xlabel("#Edges")
    ax_cond_s.set_ylabel("Conductance")
    ax_cond_s.legend()
    set_dynamic_xlim([ax_cond_s], x_sbm)
    set_dynamic_ylim(ax_cond_s, cond_sbm_gn + cond_sbm_lp + cond_sbm_alp + cond_sbm_lv + cond_sbm_bvns)

    ax_ari_s.scatter(x_sbm, ari_sbm_gn, marker='x', label='GN')
    ax_ari_s.scatter(x_sbm, ari_sbm_lp, marker='x', label='LP')
    ax_ari_s.scatter(x_sbm, ari_sbm_alp, marker='x', label='ALP')
    ax_ari_s.scatter(x_sbm, ari_sbm_lv, marker='x', label='LV')
    ax_ari_s.scatter(x_sbm, ari_sbm_bvns, marker='x', label='BVNS')
    ax_ari_s.set_title(f"{label} (SBM) - ARI")
    ax_ari_s.set_xlabel("#Edges")
    ax_ari_s.set_ylabel("ARI")
    ax_ari_s.legend()
    set_dynamic_xlim([ax_ari_s], x_sbm)
    set_dynamic_ylim(ax_ari_s, ari_sbm_gn + ari_sbm_lp + ari_sbm_alp + ari_sbm_lv + ari_sbm_bvns)

    fig_sbm.tight_layout()
    fig_sbm.savefig(f"{label.lower()}_sbm.png")
    plt.close(fig_sbm)