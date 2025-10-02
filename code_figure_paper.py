#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored plotting script:
- Programmatic access to epsilon subgroups (epsilon_1 ... epsilon_N).
- No hard-coded epsilon blocks; uses helpers and loops.
- Compatible with the HDF5 layout where each epsilon lives under the same parent group
  (e.g., f['Bruit_commoners']['data variables']['epsilon_{i+1}'])
- Reproduces the same figures as the original, for epsilon ≈ 0.01, 0.03, 0.10
  and the collapse-rate figures (vs time and vs epsilon).
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from scipy.stats import kurtosis, skew
from matplotlib import cm

# -------------------- HDF5 I/O helpers --------------------

H5_PATH = "Handy_donnees_full"  # unchanged input file name

def open_sources(h5_path=H5_PATH):
    f = h5.File(h5_path, "r")
    g_root = f["Bruit_commoners"]
    g_vars = g_root["data variables"]
    g_collapse = g_root["data de collapse et date"]
    return f, g_vars, g_collapse

def list_eps_groups(g_vars):
    # Return sorted epsilon subgroup names like ["epsilon_1", ..., "epsilon_20"]
    return sorted([k for k in g_vars.keys() if k.startswith("epsilon_")],
                  key=lambda s: int(s.split("_")[1]))

def load_eps_group(g_vars, idx):
    # idx is zero-based; groups are 1-based
    name = f"epsilon_{idx+1}"
    return g_vars[name]

def load_big_arrays(g_eps):
    # Load raw arrays for a given epsilon group
    C1 = np.array(g_eps["Commoners_big_fig1"])
    C2 = np.array(g_eps["Commoners_big_fig2"])
    C3 = np.array(g_eps["Commoners_big_fig3"])
    C4 = np.array(g_eps["Commoners_big_fig4"])

    N1 = np.array(g_eps["Nature_big_fig1"])
    N2 = np.array(g_eps["Nature_big_fig2"])
    N3 = np.array(g_eps["Nature_big_fig3"])
    N4 = np.array(g_eps["Nature_big_fig4"])

    R1 = np.array(g_eps["Ressources_big_fig1"])
    R2 = np.array(g_eps["Ressources_big_fig2"])
    R3 = np.array(g_eps["Ressources_big_fig3"])
    R4 = np.array(g_eps["Ressources_big_fig4"])
    return (C1,C2,C3,C4), (N1,N2,N3,N4), (R1,R2,R3,R4)

def stats_triplet(A):
    # returns mean, var, skew, kurtosis along axis=1 (time dimension)
    mu  = np.mean(A, axis=1)
    var = np.var(A, axis=1)
    sk  = skew(A, axis=1)
    ku  = kurtosis(A, axis=1, fisher=False)
    return mu, var, sk, ku

# -------------------- Model constants (for lines/labels only) --------------------

Nombre_annee = 1500
N = int(Nombre_annee * 10)
t = np.linspace(0, Nombre_annee, N+1, dtype="float32")

Lambda=np.float32(100) # nature carrying capacity 
beta_c=np.float32(0.03) # birth rate
gamma=np.float32(0.01) # regeneration rate of nature
alpha_M=np.float32(0.07) # famine death rate
s=np.float32(0.0005) # subsistence salary per capita
alpha_m=np.float32(0.01) # normal death rate
rho=np.float32(0.005) # threshold wealth per capita
n=np.float32((alpha_M-beta_c)/(alpha_M-alpha_m))
X_m=np.float32((gamma*(Lambda)**2)/(4*n*s))
delta_opt=np.float32(((2*n*s)/Lambda))

def carrying_capacity_line(delta_multiplier):
    delta = np.float32(delta_multiplier * delta_opt)
    X = np.float32((gamma/delta)*(Lambda - n*(s/delta))) # Carrying capacity
    y = X / (2*X_m)
    return [0, Nombre_annee], [y, y]



def find_eps_index(target):
    # find exact match first, then nearest
    arr = np.asarray(EPSILON)
    try:
        return np.where(arr == float(target))[0][0]
    except ValueError:
        return int(np.argmin(np.abs(arr - float(target))))

# Which epsilons we want figures for (to reproduce original outputs)
EPSILON = [0.03,0.1]
EPS_IDX = [find_eps_index(v) for v in EPSILON]

# Delta multipliers, matching fig1..fig4 ordering in the file
DELTA_MULTS = [1.0, 2.5, 4.0, 5.5]

# -------------------- Plot helpers --------------------

def plot_means_with_errorbars(ax, muC, varC, muN, varN, muR, varR, delta_mult):
    # Draw humans (C), nature (N), wealth (R) with variance as yerr
    line_x, line_y = carrying_capacity_line(delta_mult)
    ax.errorbar(t, muC, yerr=varC, ecolor='lightskyblue', color='blue')
    ax.errorbar(t, muN, yerr=varN, ecolor='lawngreen', color='green')
    ax.errorbar(t, muR, yerr=varR, ecolor='grey', color='black')
    ax.plot(line_x, line_y, color='red', linewidth=0.7)

def decorate_panel_titles(axs):
    axs[0,0].set_title(r'$\delta=\delta_{opt*}$')
    axs[0,1].set_title(r'$\delta=2.5*\delta_{opt*}$')
    axs[1,0].set_title(r'$\delta=4*\delta_{opt*}$')
    axs[1,1].set_title(r'$\delta=5.5*\delta_{opt*}$')

def plot_four_panels_for_epsilon(fig_suptitle, C_stats, N_stats, R_stats, out_pdf):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,7))
    fig.suptitle(fig_suptitle)

    (muC1,varC1,skC1,kuC1), (muC2,varC2,skC2,kuC2), (muC3,varC3,skC3,kuC3), (muC4,varC4,skC4,kuC4) = C_stats
    (muN1,varN1,skN1,kuN1), (muN2,varN2,skN2,kuN2), (muN3,varN3,skN3,kuN3), (muN4,varN4,skN4,kuN4) = N_stats
    (muR1,varR1,skR1,kuR1), (muR2,varR2,skR2,kuR2), (muR3,varR3,skR3,kuR3), (muR4,varR4,skR4,kuR4) = R_stats

    plot_means_with_errorbars(ax[0,0], muC1, varC1, muN1, varN1, muR1, varR1, DELTA_MULTS[0])
    plot_means_with_errorbars(ax[0,1], muC2, varC2, muN2, varN2, muR2, varR2, DELTA_MULTS[1])
    plot_means_with_errorbars(ax[1,0], muC3, varC3, muN3, varN3, muR3, varR3, DELTA_MULTS[2])
    plot_means_with_errorbars(ax[1,1], muC4, varC4, muN4, varN4, muR4, varR4, DELTA_MULTS[3])

    ax[0,1].legend(['carrying capacity','humans', 'nature', 'wealth'], shadow=True)
    decorate_panel_titles(ax)
    plt.savefig(out_pdf, format='pdf')
    plt.close()

def plot_skewness_panels(C_stats, N_stats, R_stats, suptitle, out_pdf):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,7))
    fig.suptitle(suptitle)

    skC = [C_stats[i][2] for i in range(4)]
    skN = [N_stats[i][2] for i in range(4)]
    skR = [R_stats[i][2] for i in range(4)]

    ax[0,0].plot(t, skC[0]); ax[0,0].plot(t, skN[0]); ax[0,0].plot(t, skR[0])
    ax[0,1].plot(t, skC[1]); ax[0,1].plot(t, skN[1]); ax[0,1].plot(t, skR[1])
    ax[1,0].plot(t, skC[2]); ax[1,0].plot(t, skN[2]); ax[1,0].plot(t, skR[2])
    ax[1,1].plot(t, skC[3]); ax[1,1].plot(t, skN[3]); ax[1,1].plot(t, skR[3])

    decorate_panel_titles(ax)
    plt.savefig(out_pdf, format='pdf')
    plt.close()

def plot_kurtosis_panels(C_stats, N_stats, R_stats, suptitle, out_pdf):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,7))
    fig.suptitle(suptitle)

    kuC = [C_stats[i][3] for i in range(4)]
    kuN = [N_stats[i][3] for i in range(4)]
    kuR = [R_stats[i][3] for i in range(4)]
    ligne_kurtosis = np.ones_like(t) * 3

    ax[0,0].semilogy(t, kuC[0]); ax[0,0].semilogy(t, kuN[0]); ax[0,0].semilogy(t, kuR[0]); ax[0,0].semilogy(t, ligne_kurtosis)
    ax[0,1].semilogy(t, kuC[1]); ax[0,1].semilogy(t, kuN[1]); ax[0,1].semilogy(t, kuR[1]); ax[0,1].semilogy(t, ligne_kurtosis)
    ax[1,0].semilogy(t, kuC[2]); ax[1,0].semilogy(t, kuN[2]); ax[1,0].semilogy(t, kuR[2]); ax[1,0].semilogy(t, ligne_kurtosis)
    ax[1,1].semilogy(t, kuC[3]); ax[1,1].semilogy(t, kuN[3]); ax[1,1].semilogy(t, kuR[3]); ax[1,1].semilogy(t, ligne_kurtosis)

    decorate_panel_titles(ax)
    plt.savefig(out_pdf, format='pdf')
    plt.close()

# -------------------- Collapse helpers --------------------

def load_collapse(g_collapse):
    c1 = np.array(g_collapse["Collapse_fig_1"])
    c2 = np.array(g_collapse["Collapse_fig_2"])
    c3 = np.array(g_collapse["Collapse_fig_3"])
    c4 = np.array(g_collapse["Collapse_fig_4"])
    return [c1, c2, c3, c4]

def normalize_collapse_inplace(carr_list, denom=1000.0):
    # Convert per-time-bin counts into cumulative normalized rates for every epsilon row
    for i in range(len(carr_list)):
        M = carr_list[i]
        for r in range(M.shape[0]):
            M[r, :] = np.cumsum(M[r, :]) / denom

# -------------------- Main plotting flow --------------------

def main():
    f, g_vars, g_collapse = open_sources()
    eps_groups = list_eps_groups(g_vars)

    # Collapse arrays
    c1, c2, c3, c4 = load_collapse(g_collapse)
    normalize_collapse_inplace([c1, c2, c3, c4], denom=1000.0)
    EPSILON1= np.arange(0,0.4,0.01)
    # Produce summary collapse figures (vs epsilon at final time)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,7))
    ax[0,0].plot(EPSILON1, c1[:, -1]); ax[0,0].set_ylim(0,1)
    ax[0,1].plot(EPSILON1, c2[:, -1]); ax[0,1].set_ylim(0,1)
    ax[1,0].plot(EPSILON1, c3[:, -1]); ax[1,0].set_ylim(0,1)
    ax[1,1].plot(EPSILON1, c4[:, -1]); ax[1,1].set_ylim(0,1)
    plt.savefig('handacks/collapse_rate/epsilon/collapse_rate_epsilon.pdf', format='pdf')
    plt.close()

    # Collapse vs time, for epsilon ~ 0.03 and 0.10 (indices derived from EPSILON)
    for eps_val, out_name in [(0.03, 'handacks/collapse_rate/time2/collapse_rate_time_epsilon003.pdf'),
                              (0.10, 'handacks/collapse_rate/time2/collapse_rate_time_epsilon010.pdf')]:
        idx = find_eps_index(eps_val)
        fig = plt.figure()
        gs = fig.add_gridspec(1, 2, hspace=0.05, wspace=0.05)
        (ax3, ax4) = gs.subplots(sharex='col', sharey='row',)
        ax3.grid(); ax4.grid()
        ax3.plot(t, c3[idx, :], linewidth=3); ax3.set_ylim(0,1)
        ax4.plot(t, c4[idx, :], linewidth=3); ax4.set_ylim(0,1)
        fig.supylabel('collapse rate', x=0.08, fontsize=18)
        ax4.set_xlabel('t [years]', fontsize=18); ax3.set_xlabel('t [years]', fontsize=18)
        plt.savefig(out_name, format='pdf')
        plt.close()

    # For each requested epsilon (0.01, 0.03, 0.10), compute stats and make three figure sets:
    # (means+variance errorbars), (skewness), (kurtosis)
    for eps_idx in EPS_IDX:
        g_eps = load_eps_group(g_vars, eps_idx)
        (C1,C2,C3,C4), (N1,N2,N3,N4), (R1,R2,R3,R4) = load_big_arrays(g_eps)

        C_stats = [stats_triplet(A) for A in (C1,C2,C3,C4)]
        N_stats = [stats_triplet(A) for A in (N1,N2,N3,N4)]
        R_stats = [stats_triplet(A) for A in (R1,R2,R3,R4)]

        eps_value = EPSILON[eps_idx]
        title_prefix = f"SOCIETE EQUITABLE EULER, dt = 0.1, bruit commoners, epsilon = {eps_value:.2f}, nombre de tirage = 1000"

        # Means + error bars
        plot_four_panels_for_epsilon(
            title_prefix,
            C_stats, N_stats, R_stats,
            out_pdf=f'handacks/figures/epsilon={eps_value:.2f}_nombre_de_tirage=1000.pdf'
        )

        # Skewness
        plot_skewness_panels(
            C_stats, N_stats, R_stats,
            suptitle=f'{title_prefix}, skewness',
            out_pdf=f'handacks/skewness/epsilon={eps_value:.2f}_nombre_de_tirage=1000_skew.pdf'
        )

        # Kurtosis
        plot_kurtosis_panels(
            C_stats, N_stats, R_stats,
            suptitle=f'{title_prefix}, kurtosis',
            out_pdf=f'handacks/kurtosis/epsilon={eps_value:.2f}_nombre_de_tirage=1000_kurtosis.pdf'
        )

    # Optional: 3D surfaces (unchanged, but now using normalized collapse arrays)
    for carr, title in zip([c1,c2,c3,c4], ['delta_opt','2.5*delta_opt','4*delta_opt','5.5*delta_opt']):
        fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
        X, Y = np.meshgrid(t, EPSILON1)
        ax.plot_surface(X, Y, carr, cmap=cm.coolwarm)
        ax.set_xlabel('time'); ax.set_ylabel('epsilon'); ax.set_zlabel('collapse_rate')
        ax.set_title(title)
        plt.show()

    f.close()

if __name__ == "__main__":
    main()
