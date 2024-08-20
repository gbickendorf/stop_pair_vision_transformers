"""
Fit all XGBoost GBDTs to the data and predict labels for all plots
"""

import os
import itertools
import numpy as np
from xgboost import XGBClassifier
from tqdm import tqdm


def xgbdt_train(x_train, y_train, outname, datadir="Data", tree_method="exact", n_estimators=120):
    """Fit XGBoost GBDT to the data"""
    print("Training ", outname)
    model = XGBClassifier(objective="binary:logistic", tree_method=tree_method,
                          n_estimators=n_estimators, n_jobs=10, learning_rate=0.1, random_state=1)
    model.fit(x_train, y_train)
    model.save_model(os.path.join(datadir, f"XGB_Models/{outname}.json"))


def xgbdt_predict(x_test, y_test, outname, datadir="Data", inname=None, predictdir="XGB_Predict", tree_method="exact",):
    """Predict labels for data and save"""
    model = XGBClassifier(objective="binary:logistic", tree_method=tree_method, n_jobs=10, learning_rate=0.1, random_state=1)
    if inname is None:
        inname = outname
    print("Predict ", inname)
    model.load_model(os.path.join(datadir, f"XGB_Models/{inname}.json"))
    out = model.predict_proba(x_test)[:, 1]

    np.save(os.path.join(datadir, predictdir,
                         f"y_truth_BDT_{outname}.npy"), y_test)
    np.save(os.path.join(datadir, predictdir,
                         f"y_pred_BDT_{outname}.npy"), out)


def combine_three_jets():
    """Combine three hardest jets into one prediction"""
    modelnames = ["CoAtNet", "MaxVit", "CNN"]
    jetdefs = ["AK08", "AK10", "AK14"]
    for model, jetdef in itertools.product(modelnames, jetdefs):
        xsi_train, xsi_test = [np.vstack([np.load(f"Data/predict/x_SI_{mstop}_{menu}_{jetdef}_{model}.npy")[start:stop]
                                          for mstop in range(700, 1225, 25) for menu in range(100, 510, 10)]) for start, stop in [[0, 1000], [1000, 2000]]]
        xbg_train, xbg_test = [np.load(f"Data/predict/x_BG_{jetdef}_{model}.npy")[start:stop] for start, stop in [[0, xsi_train.shape[0]], [xsi_train.shape[0], xsi_train.shape[0]+xsi_test.shape[0]]]]
        x_train, x_test = np.vstack((xbg_train, xsi_train)), np.vstack((xbg_test, xsi_test))
        y_train, y_test = np.hstack((np.zeros(xbg_train.shape[0]), np.ones(xsi_train.shape[0]))), np.hstack((np.zeros(xbg_test.shape[0]), np.ones(xsi_test.shape[0])))
        xgbdt_train(x_train, y_train, outname=f"{model}_{jetdef}", tree_method="hist")
        xgbdt_predict(x_test, y_test, outname=f"{model}_{jetdef}", tree_method="hist")


def combine_three_jets_three_jetdef():
    """Combine three hardest jets and all jetdefs into one prediction"""
    modelnames = ["CoAtNet", "MaxVit", "CNN"]
    jetdefs = ["AK08", "AK10", "AK14"]
    for model in modelnames:
        xsi_train, xsi_test = [np.vstack([np.hstack([np.load(f"Data/predict/x_SI_{mstop}_{menu}_{jetdef}_{model}.npy") for jetdef in jetdefs])[start:stop]
                                          for mstop in range(700, 1225, 25) for menu in range(100, 510, 10)]) for start, stop in [[0, 1000], [1000, 2000]]]
        xbg_train, xbg_test = [np.hstack([np.load(f"Data/predict/x_BG_{jetdef}_{model}.npy")[start:stop] for jetdef in jetdefs])
                               for start, stop in [[0, xsi_train.shape[0]], [xsi_train.shape[0], xsi_train.shape[0]+xsi_test.shape[0]]]]

        x_train, x_test = np.vstack((xbg_train, xsi_train)), np.vstack((xbg_test, xsi_test))
        y_train, y_test = np.hstack((np.zeros(xbg_train.shape[0]), np.ones(xsi_train.shape[0]))), np.hstack((np.zeros(xbg_test.shape[0]), np.ones(xsi_test.shape[0])))
        xgbdt_train(x_train, y_train, outname=model, tree_method="hist")
        xgbdt_predict(x_test, y_test, outname=model, tree_method="hist")


def combine_three_jets_three_arch():
    """Combine three hardest jets and all architectures into one prediction"""
    modelnames = ["CoAtNet", "MaxVit", "CNN"]
    jetdefs = ["AK08", "AK10", "AK14"]
    for jetdef in jetdefs:
        print(jetdef)
        xsi_train, xsi_test = [np.vstack([np.hstack([np.load(f"Data/predict/x_SI_{mstop}_{menu}_{jetdef}_{model}.npy") for model in modelnames])[start:stop]
                                          for mstop in range(700, 1225, 25) for menu in range(100, 510, 10)]) for start, stop in [[0, 1000], [1000, 2000]]]
        xbg_train, xbg_test = [np.hstack([np.load(f"Data/predict/x_BG_{jetdef}_{model}.npy")[start:stop] for model in modelnames])
                               for start, stop in [[0, xsi_train.shape[0]], [xsi_train.shape[0], xsi_train.shape[0]+xsi_test.shape[0]]]]
        x_train, x_test = np.vstack((xbg_train, xsi_train)), np.vstack((xbg_test, xsi_test))
        y_train, y_test = np.hstack((np.zeros(xbg_train.shape[0]), np.ones(xsi_train.shape[0]))), np.hstack((np.zeros(xbg_test.shape[0]), np.ones(xsi_test.shape[0])))
        xgbdt_train(x_train, y_train, outname=jetdef, tree_method="hist")
        xgbdt_predict(x_test, y_test, outname=jetdef, tree_method="hist")


def combine_three_jets_three_jetdef_three_arch():
    """Combine three hardest jets, all jetdefs and all architectures into one prediction"""
    modelnames = ["CoAtNet", "MaxVit", "CNN"]
    jetdefs = ["AK08", "AK10", "AK14"]
    xsi_train, xsi_test = [np.vstack([np.hstack([np.load(f"Data/predict/x_SI_{mstop}_{menu}_{jetdef}_{model}.npy") for model in modelnames for jetdef in jetdefs])[
                                     start:stop] for mstop in range(700, 1225, 25) for menu in range(100, 510, 10)]) for start, stop in [[0, 1000], [1000, 2000]]]
    xbg_train, xbg_test = [np.hstack([np.load(f"Data/predict/x_BG_{jetdef}_{model}.npy")[start:stop] for model in modelnames for jetdef in jetdefs])
                           for start, stop in [[0, xsi_train.shape[0]], [xsi_train.shape[0], xsi_train.shape[0]+xsi_test.shape[0]]]]
    x_train, x_test = np.vstack((xbg_train, xsi_train)), np.vstack((xbg_test, xsi_test))
    y_train, y_test = np.hstack((np.zeros(xbg_train.shape[0]), np.ones(xsi_train.shape[0]))), np.hstack((np.zeros(xbg_test.shape[0]), np.ones(xsi_test.shape[0])))
    xgbdt_train(x_train, y_train, outname="Complete", tree_method="hist")
    xgbdt_predict(x_test, y_test, outname="Complete", tree_method="hist")


def combine_three_jets_three_jetdef_transformer_only():
    """Combine three hardest jets, all jetdefs and all transformer architectures into one prediction"""
    modelnames = ["CoAtNet", "MaxVit"]
    jetdefs = ["AK08", "AK10", "AK14"]
    xsi_train, xsi_test = [np.vstack([np.hstack([np.load(f"Data/predict/x_SI_{mstop}_{menu}_{jetdef}_{model}.npy") for model in modelnames for jetdef in jetdefs])[
                                     start:stop] for mstop in range(700, 1225, 25) for menu in range(100, 510, 10)]) for start, stop in [[0, 1000], [1000, 2000]]]
    xbg_train, xbg_test = [np.hstack([np.load(f"Data/predict/x_BG_{jetdef}_{model}.npy")[start:stop] for model in modelnames for jetdef in jetdefs])
                           for start, stop in [[0, xsi_train.shape[0]], [xsi_train.shape[0], xsi_train.shape[0]+xsi_test.shape[0]]]]
    x_train, x_test = np.vstack((xbg_train, xsi_train)), np.vstack((xbg_test, xsi_test))
    y_train, y_test = np.hstack((np.zeros(xbg_train.shape[0]), np.ones(xsi_train.shape[0]))), np.hstack((np.zeros(xbg_test.shape[0]), np.ones(xsi_test.shape[0])))
    xgbdt_train(x_train, y_train, outname="OnlyTransformer", tree_method="hist")
    xgbdt_predict(x_test, y_test, outname="OnlyTransformer", tree_method="hist")


def combine_additional_features():
    """Combine computer vision redictions with classical high energy variables"""
    mode = "hist"
    modelnames = ["CoAtNet", "MaxVit", "CNN"]
    jetdefs = ["AK08", "AK10", "AK14"]
    jetdef = jetdefs[2]
    model = modelnames[1]

    xsi_vis_train, xsi_vis_test = [np.vstack([np.load(f"Data/predict/x_SI_{mstop}_{menu}_{jetdef}_{model}.npy")[start:stop]
                                              for mstop in range(700, 1225, 25) for menu in range(100, 510, 10)]) for start, stop in [[0, 1000], [1000, 2000]]]
    nsi_train, nsi_test = xsi_vis_train.shape[0], xsi_vis_test.shape[0]
    xbg_vis_train, xbg_vis_test = [np.load(f"Data/predict/x_BG_{jetdef}_{model}.npy")[start:stop] for start, stop in [[0, nsi_train], [nsi_train, nsi_train+nsi_test]]]

    xsi_snn_train, xsi_snn_test = [np.vstack([np.load(f"Data/SNN/SNN_SI_{mstop}_{mneu}.npy")[start:stop] for mstop in range(700, 1225, 25)
                                              for mneu in range(100, 510, 10)]) for start, stop in [[0, 1000], [1000, 2000]]]
    xbg_snn_train, xbg_snn_test = [np.load("Data/SNN/SNN_BG.npy")[start:stop] for start, stop in [[0, nsi_train], [nsi_train, nsi_train+nsi_test]]]

    y_train = np.hstack((np.zeros(nsi_train), np.ones(nsi_train)))
    y_test = np.hstack((np.zeros(nsi_test), np.ones(nsi_test)))

    data_selection = [37, 38, 40, 42]
    x_train = np.vstack((xbg_snn_train[:, data_selection], xsi_snn_train[:, data_selection]))
    x_test = np.vstack((xbg_snn_test[:, data_selection], xsi_snn_test[:, data_selection]))
    xgbdt_train(x_train, y_train, outname="SNN_Small", tree_method=mode)
    xgbdt_predict(x_test, y_test, outname="SNN_Small", tree_method=mode)

    data_selection = [37, 38, 40, 42]
    x_train = np.vstack((np.hstack((xbg_vis_train, xbg_snn_train[:, data_selection])), np.hstack((xsi_vis_train, xsi_snn_train[:, data_selection]))))
    x_test = np.vstack((np.hstack((xbg_vis_test, xbg_snn_test[:, data_selection])), np.hstack((xsi_vis_test, xsi_snn_test[:, data_selection]))))
    xgbdt_train(x_train, y_train, outname="MaxViT_SNN_Small", tree_method=mode)
    xgbdt_predict(x_test, y_test, outname="MaxViT_SNN_Small", tree_method=mode)

    data_selection = [29, 30, 31, 32, 33, 37, 38, 40, 42]
    x_train = np.vstack((xbg_snn_train[:, data_selection], xsi_snn_train[:, data_selection]))
    x_test = np.vstack((xbg_snn_test[:, data_selection], xsi_snn_test[:, data_selection]))
    xgbdt_train(x_train, y_train, outname="SNN_Large", tree_method=mode)
    xgbdt_predict(x_test, y_test, outname="SNN_Large", tree_method=mode)

    x_train = np.vstack((xbg_vis_train, xsi_vis_train))
    x_test = np.vstack((xbg_vis_test, xsi_vis_test))
    xgbdt_train(x_train, y_train, outname="Vis_only", tree_method=mode)
    xgbdt_predict(x_test, y_test, outname="Vis_only", tree_method=mode)

    data_selection = [29, 30, 31, 32, 33, 37, 38, 40, 42]
    x_train = np.vstack((np.hstack((xbg_vis_train, xbg_snn_train[:, data_selection])), np.hstack((xsi_vis_train, xsi_snn_train[:, data_selection]))))
    x_test = np.vstack((np.hstack((xbg_vis_test, xbg_snn_test[:, data_selection])), np.hstack((xsi_vis_test, xsi_snn_test[:, data_selection]))))
    xgbdt_train(x_train, y_train, outname="MaxViT_SNN_Large", tree_method=mode)
    xgbdt_predict(x_test, y_test, outname="MaxViT_SNN_Large", tree_method=mode)

    model = modelnames[0]

    xsi_vis_train, xsi_vis_test = [np.vstack([np.load(f"Data/predict/x_SI_{mstop}_{menu}_{jetdef}_{model}.npy")[start:stop]
                                              for mstop in range(700, 1225, 25) for menu in range(100, 510, 10)]) for start, stop in [[0, 1000], [1000, 2000]]]
    nsi_train, nsi_test = xsi_vis_train.shape[0], xsi_vis_test.shape[0]
    xbg_vis_train, xbg_vis_test = [np.load(f"Data/predict/x_BG_{jetdef}_{model}.npy")[start:stop] for start, stop in [[0, nsi_train], [nsi_train, nsi_train+nsi_test]]]

    y_train = np.hstack((np.zeros(nsi_train), np.ones(nsi_train)))
    y_test = np.hstack((np.zeros(nsi_test), np.ones(nsi_test)))

    data_selection = [29, 30, 31, 32, 33, 37, 38, 40, 42]
    x_train = np.vstack((np.hstack((xbg_vis_train, xbg_snn_train[:, data_selection])), np.hstack((xsi_vis_train, xsi_snn_train[:, data_selection]))))
    x_test = np.vstack((np.hstack((xbg_vis_test, xbg_snn_test[:, data_selection])), np.hstack((xsi_vis_test, xsi_snn_test[:, data_selection]))))
    xgbdt_train(x_train, y_train, outname="CoAtNet_SNN_Large", tree_method=mode)
    xgbdt_predict(x_test, y_test, outname="CoAtNet_SNN_Large", tree_method=mode)

    model = modelnames[2]

    xsi_vis_train, xsi_vis_test = [np.vstack([np.load(f"Data/predict/x_SI_{mstop}_{menu}_{jetdef}_{model}.npy")[start:stop]
                                              for mstop in range(700, 1225, 25) for menu in range(100, 510, 10)]) for start, stop in [[0, 1000], [1000, 2000]]]
    nsi_train, nsi_test = xsi_vis_train.shape[0], xsi_vis_test.shape[0]
    xbg_vis_train, xbg_vis_test = [np.load(f"Data/predict/x_BG_{jetdef}_{model}.npy")[start:stop] for start, stop in [[0, nsi_train], [nsi_train, nsi_train+nsi_test]]]

    y_train = np.hstack((np.zeros(nsi_train), np.ones(nsi_train)))
    y_test = np.hstack((np.zeros(nsi_test), np.ones(nsi_test)))

    data_selection = [29, 30, 31, 32, 33, 37, 38, 40, 42]
    x_train = np.vstack((np.hstack((xbg_vis_train, xbg_snn_train[:, data_selection])), np.hstack((xsi_vis_train, xsi_snn_train[:, data_selection]))))
    x_test = np.vstack((np.hstack((xbg_vis_test, xbg_snn_test[:, data_selection])), np.hstack((xsi_vis_test, xsi_snn_test[:, data_selection]))))
    xgbdt_train(x_train, y_train, outname="CNN_SNN_Large", tree_method=mode)
    xgbdt_predict(x_test, y_test, outname="CNN_SNN_Large", tree_method=mode)


def predict_large_ds_snn_vis():
    """Predict large ds with vision and classical high energy variables"""
    jetdef = "AK14"
    data_selection = [29, 30, 31, 32, 33, 37, 38, 40, 42]
    modelnames = ["CoAtNet", "MaxVit", "CNN"]
    gbdt_modelnames = ["CoAtNet_SNN_Large", "MaxViT_SNN_Large", "CNN_SNN_Large"]

    xbg_snn = np.load("Data/SNN/SNN_BG.npy")[1000*861:]
    for imodel, vis_model in enumerate(modelnames):
        xbg_vis = np.load(f"Data/predict/x_BG_{jetdef}_{vis_model}.npy")[1000*861:]
        x_mstop = np.hstack((xbg_vis, xbg_snn[:, data_selection]))
        y_mstop = np.zeros(x_mstop.shape[0])
        xgbdt_predict(x_mstop, y_mstop, inname=gbdt_modelnames[imodel], outname=f"{gbdt_modelnames[imodel]}_{jetdef}_BG", predictdir="MStop")
        for mstop, mneu in tqdm(itertools.product(range(700, 1225, 25), range(100, 510, 10))):
            xsi_vis = np.load(f"Data/predict/x_SI_{mstop}_{mneu}_{jetdef}_{vis_model}.npy")[1000:]
            xsi_snn = np.load(f"Data/SNN/SNN_SI_{mstop}_{mneu}.npy")[1000:]
            x_mstop = np.hstack((xsi_vis, xsi_snn[:, data_selection]))
            y_mstop = np.ones(x_mstop.shape[0])
            xgbdt_predict(x_mstop, y_mstop, inname=gbdt_modelnames[imodel], outname=f"{gbdt_modelnames[imodel]}_{jetdef}_{mstop}_{mneu}", predictdir="MStop")

    model = "SNN_Large"  # ,"SNN_Large"]
    x_mstop = xbg_snn[:, data_selection]
    y_mstop = np.zeros(x_mstop.shape[0])
    xgbdt_predict(x_mstop, y_mstop, inname=model, outname=f"{model}_{jetdef}_BG", predictdir="MStop")
    for mstop, mneu in tqdm(itertools.product(range(700, 1225, 25), range(100, 510, 10))):
        xsi_snn = np.load(f"Data/SNN/SNN_SI_{mstop}_{mneu}.npy")[1000:]
        x_mstop = xsi_snn[:, data_selection]
        y_mstop = np.ones(x_mstop.shape[0])
        xgbdt_predict(x_mstop, y_mstop, inname=model, outname=f"{model}_{jetdef}_{mstop}_{mneu}", predictdir="MStop")


if __name__ == "__main__":

    #combine_three_jets()

    combine_three_jets_three_jetdef()
    exit()
    combine_three_jets_three_arch()

    combine_three_jets_three_jetdef_three_arch()

    combine_three_jets_three_jetdef_transformer_only()

    combine_additional_features()

    predict_large_ds_snn_vis()
