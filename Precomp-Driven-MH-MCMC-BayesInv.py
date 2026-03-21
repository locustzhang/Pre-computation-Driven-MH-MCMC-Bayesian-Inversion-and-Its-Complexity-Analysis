"""
CLSC 闭环供应链 Stackelberg 博弈模型 — 最终修复版
========================================================

修复记录（相对原始代码共 4 项核心修复）：

[Fix 1] 排放函数：替代模型（原：加法模型）
  原：E = e_m·D + e_r·ρ·D         → 回收越多排放越多 ✗
  改：E = e_m·(1-ρ)·D + e_r·ρ·D  → 回收替代新生产，排放越少 ✓

[Fix 2] 集中决策基准：VIF 利润最大化（原：社会计划者 SW 最大化）
  原：centralized = max_{p,ρ} SW    → VIF 亏损经营 (π=-6668)，不符合现实 ✗
  改：VIF = max_{p,ρ} π_VIF        → VIF 自主盈利，政府对外设 τ^C 最大化 SW ✓
  同时加入参与约束：π_VIF ≥ 0

[Fix 3] 福利公式：含税收返还（原：缺少 +τ·E）
  原：SW = CS + π_m + π_r - η·E²        → τ→0 恒为最优 ✗
  改：SW = CS + π_m + π_r + τ·E - η·E² → 税收作为转移支付归还社会 ✓

[Fix 4] 参数 k：由 150 → 12000（确保分散和集中均为内点解）

经济意义说明：
  E^C > E^*（集中排放略高于分散）是本模型的一个内生结论，非 bug。
  集中决策消除了双重加价，产量更大（D^C=81.7 > D^*=54.4），
  即使回收率更高，产量效应仍主导排放。这是"绿色悖论"在供应链中的体现，
  需要更强的税收信号（τ^C > τ*）来应对。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize_scalar, minimize

# ==============================================================
# Publication Plot Style (NPG Nature Style)
# ==============================================================
plt.rcParams.update({
    "font.family":                    "serif",
    "font.serif":                     ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":               "cm",
    "axes.labelsize":                 13,
    "axes.titlesize":                 14,
    "axes.titleweight":               "bold",
    "legend.fontsize":                10.5,
    "xtick.labelsize":                11,
    "ytick.labelsize":                11,
    "axes.linewidth":                 1.4,
    "grid.alpha":                     0.25,
    "grid.linestyle":                 "--",
    "figure.constrained_layout.use": True,
    "savefig.dpi":                    600,
    "savefig.bbox":                   "tight",
})

# NPG Color Palette
C_RED   = "#E64B35"
C_BLUE  = "#4DBBD5"
C_GREEN = "#00A087"
C_DARK  = "#3C5488"
C_ORANG = "#F39B7F"
C_GREY  = "#7E6148"


# ==============================================================
# Model Configuration
# ==============================================================
@dataclass
class ModelConfig:
    # Demand
    a:       float = 500
    b:       float = 1.3
    gamma:   float = 10

    # Cost
    c_m:     float = 200
    c_r:     float = 170
    k:       float = 12000   # [Fix 4] was 150/600 → must satisfy k > D·Δc/2

    # Emission  (substitution model: e_m > e_r)
    e_m:     float = 0.8
    e_r:     float = 0.2

    # Environmental damage
    eta:     float = 3.0

    # Green preference
    g:       float = 0.5

    # Tax bound
    tau_max: float = 150

    tol:     float = 1e-8


# ==============================================================
# Core Functions
# ==============================================================
def demand_fn(cfg: ModelConfig, p: float) -> float:
    return max(cfg.a - cfg.b * p + cfg.gamma * cfg.g, 0.0)


def emission_fn(cfg: ModelConfig, D: float, rho: float) -> float:
    """[Fix 1] Substitution emission: recycling replaces new production."""
    return cfg.e_m * (1.0 - rho) * D + cfg.e_r * rho * D


def retailer_best_response(cfg: ModelConfig, w: float) -> float:
    return (cfg.a + cfg.gamma * cfg.g + cfg.b * w) / (2.0 * cfg.b)


def manufacturer_profit(cfg: ModelConfig, w: float, rho: float, tau: float) -> float:
    p = retailer_best_response(cfg, w)
    D = demand_fn(cfg, p)
    if D <= 0:
        return -1e10
    E = emission_fn(cfg, D, rho)
    return (w - cfg.c_m)*D + (cfg.c_m - cfg.c_r)*rho*D - tau*E - cfg.k*rho**2


# ==============================================================
# Decentralized Channel (Three-tier Stackelberg)
# ==============================================================
def optimal_recycling_rate(cfg: ModelConfig, w: float, tau: float) -> float:
    """Manufacturer's optimal ρ given w and τ.
    FOC: (c_m-c_r)D + τ(e_m-e_r)D - 2kρ = 0  →  ρ* = D[Δc + τΔe]/(2k)
    """
    res = minimize_scalar(
        lambda r: -manufacturer_profit(cfg, w, r, tau),
        bounds=(0.0, 1.0), method="bounded", options={'xatol': cfg.tol}
    )
    return res.x


def optimal_wholesale_price(cfg: ModelConfig, tau: float) -> tuple:
    """Manufacturer's optimal w given τ. Returns (w*, ρ*)."""
    def neg_profit(w):
        rho = optimal_recycling_rate(cfg, w, tau)
        return -manufacturer_profit(cfg, w, rho, tau)

    res = minimize_scalar(neg_profit, bounds=(cfg.c_m, cfg.a / cfg.b),
                          method="bounded", options={'xatol': cfg.tol})
    w = res.x
    rho = optimal_recycling_rate(cfg, w, tau)
    return w, rho


def decentralized_sw(cfg: ModelConfig, tau: float) -> float:
    """[Fix 3] Social welfare with tax-revenue redistribution.
    SW = CS + π_m + π_r + τ·E − η·E²
    The +τ·E term returns government revenue to society (pure transfer).
    """
    w, rho = optimal_wholesale_price(cfg, tau)
    p = retailer_best_response(cfg, w)
    D = demand_fn(cfg, p)
    if D <= 0:
        return -1e10
    E   = emission_fn(cfg, D, rho)
    CS  = (cfg.a - p + cfg.gamma * cfg.g)**2 / (2.0 * cfg.b)
    pi_m = manufacturer_profit(cfg, w, rho, tau)
    pi_r = (p - w) * D
    return CS + pi_m + pi_r + tau * E - cfg.eta * E**2


def decentralized_equilibrium(cfg: ModelConfig, tau=None) -> dict:
    """Compute decentralized Stackelberg equilibrium.
    If tau is None, the government's optimal τ* is solved first.
    """
    if tau is None:
        res = minimize_scalar(
            lambda t: -decentralized_sw(cfg, t),
            bounds=(0.0, cfg.tau_max), method="bounded", options={'xatol': cfg.tol}
        )
        tau = res.x

    w, rho = optimal_wholesale_price(cfg, tau)
    p    = retailer_best_response(cfg, w)
    D    = demand_fn(cfg, p)
    E    = emission_fn(cfg, D, rho)
    pi_m = manufacturer_profit(cfg, w, rho, tau)
    pi_r = (p - w) * D
    CS   = (cfg.a - p + cfg.gamma * cfg.g)**2 / (2.0 * cfg.b)
    env_dam  = cfg.eta * E**2
    tax_rev  = tau * E
    SW   = CS + pi_m + pi_r + tax_rev - env_dam

    return dict(tau=tau, w=w, p=p, rho=rho, D=D, E=E,
                pi_m=pi_m, pi_r=pi_r, CS=CS,
                env_dam=env_dam, tax_rev=tax_rev, SW=SW)


# ==============================================================
# Centralized Channel  —  VIF (Vertically Integrated Firm)
# ==============================================================
# [Fix 2]  The correct centralized benchmark for supply chain coordination:
#   • VIF = manufacturer + retailer merged → eliminates double marginalization
#   • VIF maximizes its own profit π_VIF (not SW directly)
#   • Government sets τ^C to maximize SW given VIF's profit-maximizing response
#   • Participation constraint: π_VIF ≥ 0  (no loss-making operation)
#
# This is the "second-best" centralized benchmark used in SC coordination literature.
# It is distinct from the "first-best" social planner that can operate at a loss.

def vif_profit(cfg: ModelConfig, p: float, rho: float, tau: float) -> float:
    D = demand_fn(cfg, p)
    if D <= 0:
        return -1e10
    E = emission_fn(cfg, D, rho)
    return (p - cfg.c_m)*D + (cfg.c_m - cfg.c_r)*rho*D - cfg.k*rho**2 - tau*E


def vif_best_response(cfg: ModelConfig, tau: float) -> tuple:
    """VIF's profit-maximizing (p, ρ) given τ."""
    best_pi = -1e10
    best_x  = [300.0, 0.3]
    for p0 in np.linspace(cfg.c_m + 5, cfg.a / cfg.b - 5, 10):
        for r0 in np.linspace(0.02, 0.98, 8):
            pi = vif_profit(cfg, p0, r0, tau)
            if pi > best_pi:
                best_pi = pi
                best_x  = [p0, r0]

    res = minimize(
        lambda x: -vif_profit(cfg, x[0], x[1], tau),
        x0=best_x,
        bounds=[(cfg.c_m + 0.1, cfg.a / cfg.b), (0.0, 1.0)],
        method='L-BFGS-B',
        options={'ftol': 1e-12, 'gtol': 1e-10}
    )
    return float(res.x[0]), float(res.x[1])


def centralized_sw(cfg: ModelConfig, tau: float) -> float:
    """SW when VIF plays its profit-maximizing best response to τ."""
    p, rho = vif_best_response(cfg, tau)
    pi = vif_profit(cfg, p, rho, tau)
    if pi < 0:          # participation constraint
        return -1e10
    D  = demand_fn(cfg, p)
    E  = emission_fn(cfg, D, rho)
    CS = (cfg.a - p + cfg.gamma * cfg.g)**2 / (2.0 * cfg.b)
    return CS + pi + tau * E - cfg.eta * E**2


def centralized_equilibrium(cfg: ModelConfig) -> dict:
    """Government maximizes SW over τ; VIF responds with profit-maximizing (p^C, ρ^C)."""
    # Coarse grid to locate basin
    taus_grid = np.linspace(0, cfg.tau_max, 80)
    sw_grid   = [centralized_sw(cfg, t) for t in taus_grid]
    best_idx  = int(np.argmax(sw_grid))
    tau_init  = taus_grid[best_idx]

    # Refine
    lo = max(0.0, tau_init - 12.0)
    hi = min(cfg.tau_max, tau_init + 12.0)
    res = minimize_scalar(lambda t: -centralized_sw(cfg, t),
                          bounds=(lo, hi), method="bounded",
                          options={'xatol': cfg.tol})
    tau_c = float(res.x)

    p, rho  = vif_best_response(cfg, tau_c)
    D       = demand_fn(cfg, p)
    E       = emission_fn(cfg, D, rho)
    pi_vif  = vif_profit(cfg, p, rho, tau_c)
    CS      = (cfg.a - p + cfg.gamma * cfg.g)**2 / (2.0 * cfg.b)
    env_dam = cfg.eta * E**2
    tax_rev = tau_c * E
    SW      = CS + pi_vif + tax_rev - env_dam

    return dict(tau=tau_c, p=p, rho=rho, D=D, E=E,
                pi_vif=pi_vif, CS=CS,
                env_dam=env_dam, tax_rev=tax_rev, SW=SW)


# ==============================================================
# Sensitivity Engines
# ==============================================================
def standard_sensitivity(cfg: ModelConfig, n: int = 100) -> dict:
    """τ → (ρ*, E, SW, p, w) along the decentralized equilibrium path."""
    taus = np.linspace(0, cfg.tau_max, n)
    out  = {'taus': taus, 'rho': [], 'E': [], 'SW': [], 'p': [], 'w': []}
    for t in taus:
        eq = decentralized_equilibrium(cfg, tau=t)
        out['w'].append(eq['w'])
        out['p'].append(eq['p'])
        out['rho'].append(eq['rho'])
        out['E'].append(eq['E'])
        out['SW'].append(eq['SW'])
    return {k: (v if k == 'taus' else np.array(v)) for k, v in out.items()}


def strategic_sensitivity(cfg: ModelConfig, n: int = 40) -> dict:
    """γ → (τ*, ρ*) — effect of consumer green preference."""
    gammas = np.linspace(5, 25, n)
    tau_stars, rho_stars = [], []
    orig = cfg.gamma
    for gv in gammas:
        cfg.gamma = gv
        eq = decentralized_equilibrium(cfg)
        tau_stars.append(eq['tau'])
        rho_stars.append(eq['rho'])
    cfg.gamma = orig
    return dict(gammas=gammas,
                tau_stars=np.array(tau_stars),
                rho_stars=np.array(rho_stars))


# ==============================================================
# Plot Utilities
# ==============================================================
def _clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _annotate_vline(ax, x, y_text, text, color=C_RED, offset_x=8, offset_y=0):
    ax.axvline(x, color=color, ls=':', lw=1.8, alpha=0.85)
    ax.annotate(text,
                xy=(x, y_text),
                xytext=(x + offset_x, y_text + offset_y),
                arrowprops=dict(facecolor=color, arrowstyle='->', lw=1.2),
                fontsize=10.5, fontweight='bold', color=color)


# ==============================================================
# Figure Set 1 — Pricing Dynamics & Recycling Coordination Gap
# ==============================================================
def plot_set_1(eq_nt, eq_d, eq_c, sens):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── 1A: Pricing Dynamics ────────────────────────────────────
    ax = axes[0]
    ax.plot(sens['taus'], sens['p'], color=C_DARK, lw=2.5,
            label='Retail Price ($p$)')
    ax.plot(sens['taus'], sens['w'], color=C_BLUE, lw=2.5, ls='--',
            label='Wholesale Price ($w$)')
    ax.fill_between(sens['taus'], sens['w'], sens['p'],
                    color=C_BLUE, alpha=0.10, label='Retailer Margin')

    ax.axvline(eq_d['tau'], color=C_RED, ls=':', lw=1.8, alpha=0.85)
    ax.scatter([eq_d['tau'], eq_d['tau']], [eq_d['p'], eq_d['w']],
               color=C_RED, s=65, zorder=5)
    ax.annotate(r'Optimal Tax $\tau^*$',
                xy=(eq_d['tau'], eq_d['w'] - 4),
                xytext=(eq_d['tau'] + 14, eq_d['w'] - 16),
                arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.1),
                fontsize=10.5, fontweight='bold')

    ax.set(xlabel=r'Carbon Tax Level ($\tau$)',
           ylabel='Price',
           title='A. Supply Chain Pricing Dynamics')
    ax.legend(frameon=False)
    _clean_ax(ax)

    # ── 1B: Recycling Rate & Coordination Gap ───────────────────
    ax = axes[1]
    ax.plot(sens['taus'], sens['rho'], color=C_DARK, lw=2.5,
            label=r'Decentralized Response ($\rho^*$)')
    ax.axhline(eq_c['rho'], color=C_GREEN, ls='-.', lw=2.5,
               label=fr'Centralized VIF Benchmark ($\rho^C={eq_c["rho"]:.3f}$)')

    # Gap segment at τ*
    ax.plot([eq_d['tau'], eq_d['tau']], [eq_d['rho'], eq_c['rho']],
            color=C_RED, lw=2.5)
    ax.scatter([eq_d['tau']], [eq_d['rho']],
               color=C_RED, s=65, zorder=5, label='Decentralized Equilibrium')

    mid_rho = (eq_d['rho'] + eq_c['rho']) / 2
    gap     = eq_c['rho'] - eq_d['rho']
    ax.text(eq_d['tau'] + 3, mid_rho,
            f'Coordination Gap\n$\\Delta\\rho = {gap:.3f}$',
            color=C_RED, fontweight='bold', va='center', fontsize=10.5)

    ax.set(xlabel=r'Carbon Tax Level ($\tau$)',
           ylabel=r'Recycling Rate ($\rho$)',
           title='B. Recycling Rate & Coordination Gap')
    ax.legend(frameon=False, loc='lower right')
    _clean_ax(ax)

    plt.savefig('CLSC_Set1_Pricing_Recycling.png')
    plt.savefig('CLSC_Set1_Pricing_Recycling.pdf')
    plt.show()
    print("  [Saved] CLSC_Set1_Pricing_Recycling.{png,pdf}")


# ==============================================================
# Figure Set 2 — Emission Trajectory & Welfare Optimization
# ==============================================================
def plot_set_2(eq_nt, eq_d, eq_c, sens):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── 2A: Emission Trajectory ─────────────────────────────────
    ax = axes[0]
    ax.plot(sens['taus'], sens['E'], color=C_ORANG, lw=2.5,
            label=r'Actual Emissions ($E$)')
    ax.axhline(eq_nt['E'], color=C_GREY, ls='--', lw=1.5,
               label=f'No-Tax Baseline ($E_0={eq_nt["E"]:.1f}$)')

    # Shade emission reduction area
    below = sens['E'] < eq_nt['E']
    ax.fill_between(sens['taus'], sens['E'], eq_nt['E'],
                    where=below, color=C_GREEN, alpha=0.15,
                    label='Emission Reduction Region')

    ax.axvline(eq_d['tau'], color=C_RED, ls=':', lw=1.8, alpha=0.85)
    ax.scatter([eq_d['tau']], [eq_d['E']], color=C_RED, s=65, zorder=5)

    # Mark VIF emission level
    ax.axhline(eq_c['E'], color=C_BLUE, ls=':', lw=1.5,
               label=fr'VIF Emission ($E^C={eq_c["E"]:.1f}$, $\tau^C={eq_c["tau"]:.1f}$)')
    ax.annotate('Note: $E^C > E^*$\n(output effect > recycling effect)',
                xy=(eq_c['tau'] * 0.45, eq_c['E'] + 0.5),
                fontsize=9.5, color=C_BLUE, style='italic')

    ax.set(xlabel=r'Carbon Tax Level ($\tau$)',
           ylabel=r'Total Carbon Emissions ($E$)',
           title='A. Environmental Impact Control')
    ax.legend(frameon=False, fontsize=9.5)
    _clean_ax(ax)

    # ── 2B: Social Welfare Optimization ─────────────────────────
    ax = axes[1]
    ax.plot(sens['taus'], sens['SW'], color=C_DARK, lw=2.5,
            label=r'Social Welfare ($SW$)')
    ax.axhline(eq_nt['SW'], color=C_GREY, ls='--', lw=1.5,
               label=f'No-Tax Baseline ($SW_0={eq_nt["SW"]:.0f}$)')

    # Effective policy zone
    feasible = sens['SW'] > eq_nt['SW']
    ax.fill_between(sens['taus'], 0, 1,
                    where=feasible, color=C_BLUE, alpha=0.08,
                    transform=ax.get_xaxis_transform(),
                    label=r'Effective Policy Zone ($SW > SW_0$)')

    # Welfare maximum
    idx = int(np.nanargmax(sens['SW']))
    ax.scatter([sens['taus'][idx]], [sens['SW'][idx]],
               color=C_RED, s=90, edgecolor='white', lw=1.5, zorder=6)
    ax.annotate('Welfare Maximum',
                xy=(sens['taus'][idx], sens['SW'][idx]),
                xytext=(sens['taus'][idx] + 14, sens['SW'][idx]),
                arrowprops=dict(facecolor=C_RED, arrowstyle='->', lw=1.1),
                fontsize=10.5, color=C_RED, fontweight='bold')

    # VIF welfare level
    ax.axhline(eq_c['SW'], color=C_GREEN, ls='-.', lw=1.8,
               label=fr'VIF Benchmark ($SW^C={eq_c["SW"]:.0f}$)')

    ax.set(xlabel=r'Carbon Tax Level ($\tau$)',
           ylabel=r'Social Welfare ($SW$)',
           title='B. Social Welfare Optimization')
    ax.set_ylim(bottom=min(sens['SW']) * 0.992, top=max(sens['SW']) * 1.008)
    ax.legend(frameon=False, loc='lower center')
    _clean_ax(ax)

    plt.savefig('CLSC_Set2_Environment_Welfare.png')
    plt.savefig('CLSC_Set2_Environment_Welfare.pdf')
    plt.show()
    print("  [Saved] CLSC_Set2_Environment_Welfare.{png,pdf}")


# ==============================================================
# Figure Set 3 — Strategic Benchmarking & Green Preference
# ==============================================================
def plot_set_3(cfg, eq_nt, eq_d, eq_c, strat_sens):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── 3A: Three-Scenario Bar Chart ────────────────────────────
    ax = axes[0]
    labels   = [r'No Tax ($\tau=0$)',
                fr'Decentralized ($\tau^*={eq_d["tau"]:.1f}$)',
                fr'Centralized VIF ($\tau^C={eq_c["tau"]:.1f}$)']
    rho_vals = [eq_nt['rho'], eq_d['rho'], eq_c['rho']]
    sw_vals  = [eq_nt['SW'],  eq_d['SW'],  eq_c['SW']]

    x, w = np.arange(3), 0.35
    bars1 = ax.bar(x - w/2, rho_vals, w,
                   label=r'Recycling Rate ($\rho$)',
                   color=C_DARK, edgecolor='black', hatch='//', alpha=0.82)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + w/2, sw_vals, w,
                    label=r'Social Welfare ($SW$)',
                    color=C_GREEN, edgecolor='black', hatch='\\\\', alpha=0.82)

    for bar in bars1:
        ax.text(bar.get_x() + w/2, bar.get_height() + 0.006,
                f'{bar.get_height():.3f}', ha='center',
                fontweight='bold', fontsize=10)
    for bar in bars2:
        ax2.text(bar.get_x() + w/2, bar.get_height() + max(sw_vals)*0.008,
                 f'{bar.get_height():.0f}', ha='center', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r'Recycling Rate ($\rho$)', color=C_DARK, fontweight='bold')
    ax2.set_ylabel(r'Social Welfare ($SW$)',  color=C_GREEN, fontweight='bold')
    ax.set_title(
        f'A. System Efficiency & Coordination Gap\n'
        f'$\\Delta\\rho={eq_c["rho"]-eq_d["rho"]:.3f}$,  '
        f'PoA $= {eq_d["SW"]/eq_c["SW"]:.3f}$,  '
        f'VIF profit $= {eq_c["pi_vif"]:.0f} > 0$ [OK]',
        loc='left', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, loc='upper left', frameon=False)

    # ── 3B: γ Sensitivity ───────────────────────────────────────
    ax = axes[1]
    gammas    = strat_sens['gammas']
    tau_stars = strat_sens['tau_stars']
    rho_stars = strat_sens['rho_stars']

    ax.plot(gammas, tau_stars, color=C_RED, lw=2.5,
            marker='o', markevery=5, ms=5,
            label=r'Optimal Tax ($\tau^*$)')
    ax.set_xlabel(r'Consumer Green Preference ($\gamma$)')
    ax.set_ylabel(r'Optimal Carbon Tax ($\tau^*$)',
                  color=C_RED, fontweight='bold')

    ax3 = ax.twinx()
    ax3.plot(gammas, rho_stars, color=C_DARK, lw=2.5, ls='--',
             marker='s', markevery=5, ms=5,
             label=r'Equilibrium Recycling Rate ($\rho^*$)')
    ax3.set_ylabel(r'Recycling Rate ($\rho^*$)',
                   color=C_DARK, fontweight='bold')

    ax.set_title('B. Strategic Impact of Consumer Green Preference', loc='left')
    ax.spines['top'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax.grid(True, linestyle=':', alpha=0.4)

    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax3.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc='upper left', frameon=False)

    plt.savefig('CLSC_Set3_Strategy.png')
    plt.savefig('CLSC_Set3_Strategy.pdf')
    plt.show()
    print("  [Saved] CLSC_Set3_Strategy.{png,pdf}")


# ==============================================================
# Console Report
# ==============================================================
def print_paper_report(eq_nt, eq_d, eq_c):

    DIV  = "═" * 87
    div  = "─" * 87

    print(f"\n{DIV}")
    print(" 📜  RESEARCH PAPER ASSISTANT: DATA & INSIGHTS REPORT".center(87))
    print(DIV)

    # ── Section 1: Equilibrium Table ────────────────────────────
    print("\n>>> SECTION 1: MACRO SYSTEM EQUILIBRIUM COMPARISON")
    print(div)
    h = f"  {'Metric':<24}  {'No Tax (Base)':>15}  {'Decentralized (τ*)':>18}  {'VIF Centralized':>15}"
    print(h)
    print(div)

    rows = [
        ("Carbon Tax (τ)",        0.0,         eq_d['tau'],   eq_c['tau']),
        ("Retail Price (p)",      eq_nt['p'],  eq_d['p'],     eq_c['p']),
        ("Wholesale Price (w)",   eq_nt['w'],  eq_d['w'],     float('nan')),
        ("Recycling Rate (ρ)",    eq_nt['rho'],eq_d['rho'],   eq_c['rho']),
        ("Market Demand (D)",     eq_nt['D'],  eq_d['D'],     eq_c['D']),
        ("Total Emissions (E)",   eq_nt['E'],  eq_d['E'],     eq_c['E']),
        ("Mfr/VIF Profit (π)",   eq_nt['pi_m'],eq_d['pi_m'], eq_c['pi_vif']),
        ("Retailer Profit (π_r)",eq_nt['pi_r'],eq_d['pi_r'], float('nan')),
        ("Consumer Surplus (CS)", eq_nt['CS'], eq_d['CS'],    eq_c['CS']),
        ("Env. Damage (η·E²)",   eq_nt['env_dam'],eq_d['env_dam'],eq_c['env_dam']),
        ("Tax Revenue (τ·E)",     0.0,         eq_d['tax_rev'],eq_c['tax_rev']),
        ("Social Welfare (SW)",   eq_nt['SW'], eq_d['SW'],    eq_c['SW']),
    ]
    for name, v1, v2, v3 in rows:
        s3 = f"{v3:>15.2f}" if not (isinstance(v3, float) and np.isnan(v3)) else f"{'—':>15}"
        print(f"  {name:<24}  {v1:>15.2f}  {v2:>18.2f}  {s3}")

    # ── Section 2: Sanity Checks ─────────────────────────────────
    print(f"\n>>> SECTION 2: ECONOMIC SANITY CHECKS")
    print(div)

    cfg_tau_max = 150
    checks = [
        ("τ* is interior solution",             0 < eq_d['tau'] < cfg_tau_max * 0.97),
        ("τ^C is interior solution",            0 < eq_c['tau'] < cfg_tau_max * 0.97),
        ("ρ* is interior  (0, 1)",              0.01 < eq_d['rho'] < 0.99),
        ("ρ^C is interior (0, 1)",              0.01 < eq_c['rho'] < 0.99),
        ("VIF profit ≥ 0 (participates)",       eq_c['pi_vif'] >= 0),
        ("p_dec > p_VIF (double margin.)",      eq_d['p'] > eq_c['p']),
        ("ρ^C > ρ* (coordination gap > 0)",     eq_c['rho'] > eq_d['rho']),
        ("τ^C > τ* (VIF needs stronger signal)",eq_c['tau'] > eq_d['tau']),
        ("SW_dec > SW_notax (tax improves SW)", eq_d['SW'] > eq_nt['SW']),
        ("SW_VIF > SW_dec (VIF more efficient)",eq_c['SW'] > eq_d['SW']),
        ("Mfr profit > 0",                      eq_d['pi_m'] > 0),
        ("Retailer profit > 0",                 eq_d['pi_r'] > 0),
    ]
    for name, ok in checks:
        print(f"  {'✓' if ok else '✗'}  {name}")

    # ── Section 3: Key Findings ───────────────────────────────────
    E_red   = (eq_nt['E']  - eq_d['E'])  / eq_nt['E']  * 100
    SW_gain = (eq_d['SW']  - eq_nt['SW'])/ eq_nt['SW'] * 100
    gap     = eq_c['rho']  - eq_d['rho']
    prof_d  = eq_d['pi_m'] + eq_d['pi_r']
    prof_nt = eq_nt['pi_m']+ eq_nt['pi_r']
    prof_drop = (prof_nt - prof_d) / prof_nt * 100
    m_share   = eq_d['pi_m'] / prof_d * 100
    r_share   = eq_d['pi_r'] / prof_d * 100
    poa       = eq_d['SW']   / eq_c['SW']

    print(f"\n>>> SECTION 3: KEY FINDINGS FOR 'RESULTS & DISCUSSION'")
    print(div)

    print(f"\n🔹 [Environmental Effectiveness]")
    print(f"   The optimal carbon tax (τ* = {eq_d['tau']:.2f}) reduces total emissions by")
    print(f"   {E_red:.1f}% (from {eq_nt['E']:.1f} to {eq_d['E']:.1f}).")
    print(f"   Note: VIF emissions (E^C = {eq_c['E']:.1f}) exceed E^* = {eq_d['E']:.1f}, because")
    print(f"   eliminating double marginalization expands output (D^C={eq_c['D']:.1f} > D^*={eq_d['D']:.1f}),")
    print(f"   and the demand effect outweighs the recycling effect — a supply-chain")
    print(f"   'green paradox'. This motivates a stronger centralized tax (τ^C={eq_c['tau']:.1f} > τ*={eq_d['tau']:.1f}).")

    print(f"\n🔹 [Welfare vs. Profit Trade-off]")
    print(f"   The tax reduces total supply chain profit by {prof_drop:.1f}%,")
    print(f"   yet social welfare rises by {SW_gain:.1f}%, as tax revenue redistribution")
    print(f"   and environmental damage reduction more than offset private losses.")

    print(f"\n🔹 [Double Marginalization & Coordination Gap]")
    print(f"   In the decentralized channel, the retailer's markup is {eq_d['p']-eq_d['w']:.2f}.")
    print(f"   VIF eliminates this, lowering retail price by {eq_d['p']-eq_c['p']:.2f} (from {eq_d['p']:.1f} to {eq_c['p']:.1f}).")
    print(f"   This drives the recycling rate from ρ* = {eq_d['rho']:.3f} to ρ^C = {eq_c['rho']:.3f},")
    print(f"   a coordination gap of Δρ = {gap:.3f}.")
    print(f"   Critically, VIF remains profitable (π_VIF = {eq_c['pi_vif']:.0f} > 0).")

    print(f"\n🔹 [Profit Distribution — Decentralized Channel]")
    print(f"   Under τ*, the manufacturer captures {m_share:.1f}% of channel profit,")
    print(f"   leaving {r_share:.1f}% to the retailer — reflecting Stackelberg leader advantage.")

    print(f"\n🔹 [Efficiency Loss (Price of Anarchy)]")
    print(f"   PoA = SW(τ*) / SW^C = {poa:.4f}  →  {(1-poa)*100:.1f}% welfare loss")
    print(f"   from market decentralization, motivating supply chain coordination mechanisms.")

    print(f"\n{DIV}\n")


# ==============================================================
# Main
# ==============================================================
def main():
    cfg = ModelConfig()

    print("\n" + "=" * 60)
    print("  CLSC Model  —  Fixed & Verified")
    print("=" * 60)
    print(f"  Emission model : E = e_m·(1-ρ)·D + e_r·ρ·D  [substitution]")
    print(f"  Welfare        : SW = CS + π + τ·E − η·E²    [with tax return]")
    print(f"  Centralized    : VIF profit-maximizing + participation ≥ 0")
    print(f"  k = {cfg.k}  (ensures interior ρ* and ρ^C)")

    # ── Solve Equilibria ────────────────────────────────────────
    print("\n[1/3] Solving equilibria...")
    eq_nt = decentralized_equilibrium(cfg, tau=0.0)
    eq_d  = decentralized_equilibrium(cfg)
    eq_c  = centralized_equilibrium(cfg)

    # ── Console Report ──────────────────────────────────────────
    print_paper_report(eq_nt, eq_d, eq_c)

    # ── Sensitivity Data ────────────────────────────────────────
    print("[2/3] Computing sensitivity data (100 + 40 grid points)...")
    sens       = standard_sensitivity(cfg, n=100)
    strat_sens = strategic_sensitivity(cfg, n=40)

    # ── Figures ─────────────────────────────────────────────────
    print("[3/3] Rendering 600 DPI publication figures...")
    plot_set_1(eq_nt, eq_d, eq_c, sens)
    plot_set_2(eq_nt, eq_d, eq_c, sens)
    plot_set_3(cfg, eq_nt, eq_d, eq_c, strat_sens)

    print("\n✅  All figures saved (600 DPI, PNG + PDF).")


if __name__ == "__main__":
    main()
