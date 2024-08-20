"""
Create all plots
"""

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from scipy.ndimage import gaussian_filter


modelnames = ["CoAtNet", "CNN", "MaxVit"]
jetdefs = ["AK08", "AK10", "AK14"]


def sic(xbg, xsi, npoints=500):
    """Calculate epsilon_s/sqrt(epsilon_b)"""
    tpr = np.linspace(1, 0, npoints)
    return tpr/np.vectorize(lambda t: 1e-10 + np.sqrt(np.sum(xbg > t)/len(xbg)))(np.quantile(xsi, 1-tpr))


def significance(s, b, delta=0.06):
    """Calculate exclusion significance. Expression from eq. 1,6 PhysRevD.92.115018 """
    if delta == 0:
        return np.sqrt(2*(s-b*np.log(1+s/b)))
    db = delta*b
    x = np.sqrt((s+b)**2-4*s*b*db**2/(b+db**2))
    return np.sqrt(2*(s-b*np.log((b+s+x)/(2*b))-b**2/db**2*np.log((b-s+x)/(2*b)))-(b+s-x)*(1+b/(db**2)))

def get_best_epoch(datadir, jetdef, modelname):
    """Calculates the epoch with lowest validation loss"""
    epoch = 1
    val_loss_acc = []
    epochs = []
    while True:
        path = os.path.join(datadir, f"predictions/val_loss_acc_{jetdef}_{modelname}_{epoch}.npy")
        epochs.append(epoch)
        if not os.path.isfile(path):
            break
        epoch += 1
        val_loss_acc.append(np.load(path))
    epoch = epochs[np.argmin(np.stack(val_loss_acc)[:, 0])]
    return epoch

def plot_img():
    """Plot average jet image"""
    norm = LogNorm(vmin=1e-8, vmax=1)
    img_bg = np.load("imgBG.npy")
    img_si = np.load("imgSI.npy")
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(np.sum(img_bg, axis=0), norm=norm, cmap="Blues")
    axs[1].imshow(np.sum(img_si, axis=0), norm=norm, cmap="Blues")
    diff = np.sum(img_si, axis=0)-np.sum(img_bg, axis=0)
    diff[np.abs(diff) < 1e-8] = 0
    p3=axs[2].imshow(diff, norm=SymLogNorm(1e-8,clip=False,vmax=1,vmin=-1),cmap="RdBu")
    fig.colorbar(p3, ax=axs, ticks=[1, 1e-2, 1e-4, 1e-6, 1e-8, -1e-8, -1e-6, -1e-4, -1e-2, -1])
    axs[0].set_title("BG")
    axs[1].set_title("SI")
    axs[2].set_title("SI-BG")
    plt.savefig("plots/jetimages.pdf", bbox_inches="tight")


def plot_sic_single_jet():
    """Plot sics for single jet images"""
    _, ax = plt.subplots(figsize=(8, 5), nrows=1, ncols=1)
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    ax.set_ylim(1, 10)
    cols = ["tab:blue", "tab:orange", "tab:red"]
    ls = [":", "-.", "-"]
    tpr = np.linspace(1, 0, 500)
    custom_lines = [Line2D([0], [0], marker='o', lw=0, color=cols[0]),
                    Line2D([0], [0], marker='o', lw=0, color=cols[1]),
                    Line2D([0], [0], marker='o', lw=0, color=cols[2]),
                    Line2D([0], [0], color="k", ls=ls[0]),
                    Line2D([0], [0], color="k", ls=ls[1]),
                    Line2D([0], [0], color="k", ls=ls[2])]
    nsampe = 100
    with tqdm(total=nsampe*len(modelnames)*len(jetdefs)) as pbar:
        for i, modelname in enumerate(modelnames):
            for j, jetdef in enumerate(jetdefs):
                epoch = get_best_epoch("Data",jetdef, modelname)
                #print("Best epoch : ", epoch)
                ypred = np.load(f"Data/predictions/y_pred_{jetdef}_{modelname}_{epoch}.npy")
                ytrue = np.load(f"Data/predictions/y_true_{jetdef}_{modelname}_{epoch}.npy")
                xbg_c = ypred[ytrue == 0]
                xsi_c = ypred[ytrue == 1]
                sics = []
                for _ in range(nsampe):
                    xbg = np.random.choice(xbg_c, xbg_c.shape[0], replace=True)
                    xsi = np.random.choice(xsi_c, xsi_c.shape[0], replace=True)
                    sics.append(sic(xbg, xsi))
                    pbar.update()
                sic_eps = np.mean(np.vstack(sics), axis=0)
                std_eps = np.std(np.vstack(sics), axis=0)
                ax.grid(True)
                ax.plot(tpr, sic_eps, linestyle=ls[j], c=cols[i], label=f"{jetdef} {modelname}")
                ax.fill_between(tpr, sic_eps-std_eps, sic_eps+std_eps, color=cols[i], alpha=0.2)

    ax.set_ylabel(r"$\epsilon_S/\sqrt{\epsilon_B}$")
    ax.set_xlabel(r"$\epsilon_S$")
    ax.legend(custom_lines, ['CoAtNet', 'CNN', 'MaxViT', "AK08", "AK10", "AK14"], ncols=2, columnspacing=0.5, handletextpad=0.05, loc='upper right', frameon=False, borderaxespad=0.1)
    plt.savefig("plots/base_classifiers_sic.pdf", bbox_inches="tight")


def plot_sig_gbdt_jets():
    """Plot sics for combined three hardest jets"""
    _, ax = plt.subplots(figsize=(8, 5), nrows=1, ncols=1)
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    ax.set_ylim(1, 12)
    cols = ["tab:blue", "tab:orange", "tab:red"]
    ls = [":", "-.", "-"]
    tpr = np.linspace(1, 0, 500)
    custom_lines = [Line2D([0], [0], marker='o', lw=0, color=cols[0]),
                    Line2D([0], [0], marker='o', lw=0, color=cols[1]),
                    Line2D([0], [0], marker='o', lw=0, color=cols[2]),
                    Line2D([0], [0], color="k", ls=ls[0]),
                    Line2D([0], [0], color="k", ls=ls[1]),
                    Line2D([0], [0], color="k", ls=ls[2])]
    nsampe = 100
    with tqdm(total=nsampe*len(modelnames)*len(jetdefs)) as pbar:
        for i, modelname in enumerate(modelnames):
            for j, jetdef in enumerate(jetdefs):
                x = np.load(f"Data/XGB_Predict/y_pred_BDT_{modelname}_{jetdef}.npy")
                ytrue = np.load(f"Data/XGB_Predict/y_truth_BDT_{modelname}_{jetdef}.npy")
                xbg_c, xsi_c = x[ytrue == 0], x[ytrue == 1]
                sics = []
                for _ in range(nsampe):
                    xbg = np.random.choice(xbg_c, xbg_c.shape[0], replace=True)
                    xsi = np.random.choice(xsi_c, xsi_c.shape[0], replace=True)
                    sics.append(sic(xbg, xsi))
                    pbar.update()
                sic_eps = np.mean(np.vstack(sics), axis=0)
                std_eps = np.std(np.vstack(sics), axis=0)
                ax.grid(True)
                ax.plot(tpr, sic_eps, linestyle=ls[j], c=cols[i], label=f"{jetdef} {modelname}")
                ax.fill_between(tpr, sic_eps-std_eps, sic_eps+std_eps, color=cols[i], alpha=0.2)
    ax.set_ylabel(r"$\epsilon_S/\sqrt{\epsilon_B}$")
    ax.set_xlabel(r"$\epsilon_S$")
    ax.legend(custom_lines, ['CoAtNet', 'CNN', 'MaxViT', "AK08", "AK10", "AK14"], ncols=2, columnspacing=0.5, handletextpad=0.05, loc='upper right', frameon=False, borderaxespad=0.1)

    plt.savefig("plots/base_classifiers_sic_BDT.pdf", bbox_inches="tight")


def plots_sic_mneu_single_model_jetdef():
    """Plot sics vs mneu"""
    _, ax = plt.subplots(figsize=(8, 5), nrows=1, ncols=1)
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    ax.set_ylim(4, 12)
    cols = ["tab:blue", "tab:orange", "tab:red"]
    ls = [":", "-.", "-"]

    custom_lines = [Line2D([0], [0], marker='o', lw=0, color=cols[0]),
                    Line2D([0], [0], marker='o', lw=0, color=cols[1]),
                    Line2D([0], [0], marker='o', lw=0, color=cols[2]),
                    Line2D([0], [0], color="k", ls=ls[0]),
                    Line2D([0], [0], color="k", ls=ls[1]),
                    Line2D([0], [0], color="k", ls=ls[2])]
    m_test = np.vstack([np.stack((np.ones(1000)*mstop, np.ones(1000)*menu)).T for mstop in range(700, 1225, 25)
                        for menu in range(100, 510, 10)])
    m = m_test[:, 1]
    mneus = range(100, 510, 10)
    nsampe = 20
    with tqdm(total=nsampe*len(modelnames)*len(jetdefs)*len(mneus)) as pbar:
        for i, modelname in enumerate(modelnames):
            for j, jetdef in enumerate(jetdefs):
                sics = []
                sics_std = []
                y_pred = np.load(f"Data/XGB_Predict/y_pred_BDT_{modelname}_{jetdef}.npy")
                y_true = np.load(f"Data/XGB_Predict/y_truth_BDT_{modelname}_{jetdef}.npy")
                xbg_c = y_pred[y_true == 0]

                for mass in mneus:
                    # xsi_c=y_si[imass]
                    xsi_c = y_pred[y_true == 1][m == mass]
                    sic_m = []
                    for _ in range(nsampe):
                        xbg = np.random.choice(xbg_c, xbg_c.shape[0], replace=True)
                        xsi = np.random.choice(xsi_c, xsi_c.shape[0], replace=True)
                        sic_m.append(0.3/np.sqrt(np.sum(xbg > np.quantile(xsi, 0.7))/xbg.shape[0]))
                        pbar.update()
                    sics.append(np.mean(sic_m))
                    sics_std.append(np.std(sic_m))
                ax.grid(True)
                sics = np.array(sics)
                sics_std = np.array(sics_std)
                ax.plot(mneus, sics, linestyle=ls[j], c=cols[i], label=f"{jetdef} {modelname}")
                ax.fill_between(mneus, sics-sics_std, sics+sics_std, facecolor=cols[i], alpha=0.2)
    ax.set_ylabel(r"$\epsilon_S/\sqrt{\epsilon_B}$")
    ax.set_xlabel(r"$m_{\widetilde{\chi}}/{\rm GeV}$")
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.legend(custom_lines, ['CoAtNet', 'CNN', 'MaxViT', "AK08", "AK10", "AK14"], ncols=2, columnspacing=0.5, handletextpad=0.05, loc=(0.35, -0.00), frameon=False, borderaxespad=0.1)
    plt.savefig("plots/classifiers_mass.pdf", bbox_inches="tight")


def sic_mneu_gbdt_model_combinations():
    """Plot sics vs mneu for model combinations"""
    _, ax = plt.subplots(figsize=(8, 5), nrows=1, ncols=1)
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    ax.set_ylim(5, 14)

    cols = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple", "tab:brown"]

    names = ['CoAtNet', 'CNN', 'MaxViT', "Combined AK14", "CoAtNet+MaxViT", "All"]

    m_test = np.vstack([np.stack((np.ones(1000)*mstop, np.ones(1000)*menu)).T for mstop in range(700, 1225, 25)
                        for menu in range(100, 510, 10)])

    mneus = range(100, 500, 10)
    nsampe = 50
    modelcombinations = ["CoAtNet", "CNN", "MaxVitpt", "AK14", "OnlyTransformer", "Complete"]
    with tqdm(total=nsampe*len(modelcombinations)*len(mneus)) as pbar:
        for i, modelname in enumerate(modelcombinations):
            sics = []
            sics_std = []
            y_pred = np.load(f"Data/XGB_Predict/y_pred_BDT_{modelname}.npy")
            y_true = np.load(f"Data/XGB_Predict/y_truth_BDT_{modelname}.npy")
            xbg_c = y_pred[y_true == 0]

            for mass in mneus:
                xsi_c = y_pred[y_true == 1][np.logical_and(np.logical_and(m_test[:, 0] > 0, m_test[:, 0] < 1225), m_test[:, 1] == mass)]
                sic_m = []
                for _ in range(nsampe):  # 100
                    xbg = np.random.choice(xbg_c, xbg_c.shape[0], replace=True)
                    xsi = np.random.choice(xsi_c, xsi_c.shape[0], replace=True)
                    sic_m.append(0.3/np.sqrt(np.sum(xbg > np.quantile(xsi, 0.7))/xbg.shape[0]))
                    pbar.update()
                sics.append(np.mean(sic_m))
                sics_std.append(np.std(sic_m))

            ax.grid(True)
            sics = np.array(sics)
            sics_std = np.array(sics_std)
            ax.plot(mneus, sics, label=names[i], c=cols[i])
            ax.fill_between(mneus, sics-sics_std, sics+sics_std, alpha=0.2, color=cols[i])
    ax.set_ylabel(r"$\epsilon_S/\sqrt{\epsilon_B}$")
    ax.set_xlabel(r"$m_{\widetilde{\chi}}/{\rm GeV}$")
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.legend(ncols=2, frameon=False)
    plt.savefig("plots/classifiers_combined_all_mass.pdf", bbox_inches="tight")


def plot_sic_mneu_gbdt_extended_features():
    """Plot sics vs mneu with extended feature sets"""
    _, ax = plt.subplots(figsize=(8, 5), nrows=1, ncols=1)
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    ax.set_ylim(5, 22)

    cols = ["tab:blue", "tab:orange", "tab:red", "tab:cyan", "tab:green"]

    m_test = np.vstack([np.stack((np.ones(3000)*mstop, np.ones(3000)*menu)).T for mstop in range(700, 1225, 25) for menu in range(100, 510, 10)])
    m_neu = m_test[:, 1]
    mneus = range(100, 510, 10)

    labels = ["MaxViT+DS1", "MaxViT+DS2", "MaxVit", "DS1", "DS2"]
    nsample = 50
    modelcombinations = ["MaxViT_SNN_Small", "MaxViT_SNN_Large", "Vis_only", "SNN_Small", "SNN_Large"]
    with tqdm(total=nsample*len(modelcombinations)*len(mneus)) as pbar:
        for i, dataversion in enumerate(modelcombinations):
            sics = []
            sics_std = []
            y_pred = np.load(f"Data/XGB_Predict/y_pred_BDT_{dataversion}.npy")
            y_true = np.load(f"Data/XGB_Predict/y_truth_BDT_{dataversion}.npy")
            xbg_c = y_pred[y_true == 0]

            for mass in mneus:
                xsi_c = y_pred[y_true == 1][m_neu == mass]
                sic_m = []
                for _ in range(nsample):
                    xbg = np.random.choice(xbg_c, xbg_c.shape[0], replace=True)
                    xsi = np.random.choice(xsi_c, xsi_c.shape[0], replace=True)
                    sic_m.append(0.3/np.sqrt(np.sum(xbg > np.quantile(xsi, 0.7))/xbg.shape[0]))
                    pbar.update()
                sics.append(np.mean(sic_m))
                sics_std.append(np.std(sic_m))

            ax.grid(True)
            sics = np.array(sics)
            sics_std = np.array(sics_std)
            ax.plot(mneus, sics, c=cols[i], label=labels[i])
            ax.fill_between(mneus, sics-sics_std, sics+sics_std, alpha=0.2, color=cols[i])
    ax.set_ylabel(r"$\epsilon_S/\sqrt{\epsilon_B}$")
    ax.set_xlabel(r"$m_{\widetilde{\chi}}/{\rm GeV}$")
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.legend(ncols=2, frameon=False, columnspacing=0.7)
    plt.savefig("plots/classifiers_combined_all_mass_additional_data.pdf", bbox_inches="tight")


def plot_exclusion_bounds():
    """Plot exclusion bounds"""
    res = []
    jetdef = "AK14"
    xsect = []
    with open("mass_crosssection.dat", encoding="utf-8") as f:
        for line in f.readlines():
            xsect.append(np.array(line.strip().split("\t"), dtype=np.float32))
    xsect = np.stack(xsect)
    n_bg_exp = 273084.772290251
    mstops = range(700, 1225, 25)
    mneus = range(100, 510, 10)
    modelnames_snn = ["CoAtNet_SNN_Large", "MaxViT_SNN_Large", "CNN_SNN_Large", "SNN_Large"]
    with tqdm(total=len(modelnames_snn)*len(mneus)*len(mstops)) as pbar:
        for modelname in modelnames_snn:
            xbg_c = np.load(f"Data/MStop/y_pred_BDT_{modelname}_{jetdef}_BG.npy")
            sics = np.zeros((len(mstops), len(mneus)))
            for imstop, mstop in enumerate(mstops):
                for ineu, mneu in enumerate(mneus):
                    xsi_c = np.load(f"Data/MStop/y_pred_BDT_{modelname}_{jetdef}_{mstop}_{mneu}.npy")
                    quant = np.quantile(xsi_c, 0.7)
                    dat = xsect[np.logical_and(xsect[:, 1] == mstop, xsect[:, 2] == mneu)][0]
                    n_si_exp = dat[0]*137000*dat[3]/dat[4]
                    s = 0.3*n_si_exp
                    b = np.sum(xbg_c > quant)/xbg_c.shape[0]*n_bg_exp
                    sics[imstop, ineu] = significance(s, b)
                    pbar.update()
            res.append(sics)
    figsize = (8, 7)
    vmin = 0.1
    vmax = np.max(res)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    ls = ["solid", "dashed", "dotted"]
    std = 1
    levels = np.array([np.log(1), np.log(1.28155), np.log(1.64485)])
    contours=[]
    for imodelname, modelname in enumerate(modelnames_snn):
        fig, ax = plt.subplots(figsize=(6,7), nrows=1, ncols=1)
        plt.rcParams['text.usetex'] = True
        plt.rcParams.update({'font.size': 20})
        c = ax.pcolormesh(mneus, mstops, res[imodelname], norm=norm)
        contour=ax.contour(mneus, mstops, gaussian_filter(np.log(res[imodelname]+1e-6), std), levels=levels, colors='k', interpolation='none', linestyles=ls)
        contours.append(contour)
        ax.set_ylabel(r"$m_{\widetilde{t}}/{\rm GeV}$")
        ax.set_xlabel(r"$m_{\widetilde{\chi}}/{\rm GeV}$")
        ax.grid(True)
        ax.set_ylim(700,1200)
        ax.set_xlim(100,500)

        fig.colorbar(c, ax=ax)
        plt.savefig(f"plots/{modelname}.pdf", bbox_inches="tight")
    excl_mneu = []

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    mtest = np.linspace(100, 500, 1000)
    funcfits=[]
    for i in range(4):
        x=contours[i].allsegs[2][0][:,0]
        y=contours[i].allsegs[2][0][:,1]
        srt=np.argsort(x)
        y=y[srt]
        x=x[srt]
        funcfits.append(np.interp(mtest,x,y,left=np.nan, right=np.nan))
    cols=["C0","C1","C2"]
    for i in range(3):
        improvement=funcfits[i]-funcfits[3]
        ax.plot(mtest, improvement, label=modelnames_snn[i].replace("_SNN_Large",""),c=cols[i])
        mask=np.isnan(funcfits[i])
        ax.plot(mtest[mask],1200-funcfits[3][mask], ls=":", c=cols[i])
    ax.grid(True)
    ax.legend()
    ax.set_xlim(100, 500)
    ax.set_xlabel(r"$m_{\widetilde{\chi}}/{\rm GeV}$")
    ax.set_ylabel(r"$\Delta m_{\widetilde{t}}/{\rm GeV}$")
    ax.grid(which="minor", visible=True, ls=":", alpha=0.6)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(10))
    plt.savefig("plots/excl_mstop.pdf", bbox_inches="tight")



def plot_extended_distribution():
    """Plot extended feature set distributions"""
    for ifeature in tqdm(range(9)):
        _, ax = plt.subplots(figsize=(8, 5), nrows=1, ncols=1)

        
        ls = [':', '-']
        c = ["C1", "C2"]
        data_selection = [29, 30, 31, 32, 33, 37, 38, 40, 42]
        data_names = [r"$H_1$", r"$H_2$", r"$H_3$", r"$H_4$", r"$H_5$", r"$p_T^{\rm miss}/{\rm GeV}$", r"$H_T/{\rm GeV}$", r"$M_J/{\rm GeV}$", r"$N_j$"]
        filenames = ["H1", "H2", "H3", "H4", "H5", "p_t_miss", "H_t", "M_j", "N_j"]
        i = ifeature
        bins = 50 if i != 8 else range(7, 20)
        ax.set_ylabel("a.u.")
        ax.set_xlabel(data_names[i])
        i = data_selection[i]
        snn_bg = np.load("Data/SNN/SNN_BG.npy")
        ax.hist(snn_bg[:, i], density=True, bins=bins, histtype='step', color="C0", lw=2)
        for imstop, mstop in enumerate([700, 1100]):
            for imneu, mneu in enumerate([100, 300]):
                ax.hist(np.load(f"Data/SNN/SNN_SI_{mstop}_{mneu}.npy")[:9500, i], density=True, bins=bins, histtype='step', color=c[imstop], ls=ls[imneu], lw=2)

        ax.grid(True)
        if ifeature <5:
            ax.set_xlim(0,1)
        custom_lines = [Line2D([0], [0], marker='o', lw=0, color=c[0]),
                        Line2D([0], [0], marker='o', lw=0, color=c[1]),
                        Line2D([0], [0], marker='o', lw=0, color="C0"),
                        Line2D([0], [0], color="k", ls=ls[0]),
                        Line2D([0], [0], color="k", ls=ls[1]), ]
        ax.legend(custom_lines, [r"$m_{\widetilde{t}}=700{\rm GeV}$", r"$m_{\widetilde{t}}=1100{\rm GeV}$", 'Background', r"$m_{\widetilde{\chi}}=100{\rm GeV}$",
                                 r"$m_{\widetilde{\chi}}=300{\rm GeV}$"], ncols=2, columnspacing=0.5, handletextpad=0.05, loc='upper right', frameon=False, borderaxespad=0.1)
        plt.savefig(f"plots/{filenames[ifeature]}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 20})
    #plot_img()
    #plot_sic_single_jet()
    #plot_sig_gbdt_jets()
    #plots_sic_mneu_single_model_jetdef()
    sic_mneu_gbdt_model_combinations()
    #plot_sic_mneu_gbdt_extended_features()
    #plot_exclusion_bounds()
    #plot_extended_distribution()
