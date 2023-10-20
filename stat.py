# python stat.py '{"Func": "Levene", "Samples": [[7, 14, 14, 13, 12, 9, 6, 14, 12, 8], [15, 17, 13, 15, 15, 13, 9, 12, 10, 8], [6, 8, 8, 9, 5, 14, 13, 8, 10, 9]]}'
import collections
import logging
import json
import sys

import numpy as np
import scipy.stats as stats


def pingouin_welch(samples):
    stacked = []
    for i, sample in enumerate(samples):
        for s in sample:
            stacked.append([i, s])
    stacked = np.array(stacked)
    df = pd.DataFrame(stacked, columns=["sample", "v"])

    aov = pingouin.welch_anova(dv="v", between="sample", data=df)

    return aov.iloc[0]


def welch(samples):
    stacked = []
    for i, sample in enumerate(samples):
        for s in sample:
            stacked.append([i, s])
    stacked = np.array(stacked)
    df = pd.DataFrame(stacked, columns=["sample", "v"])

    data = df
    dv = "v"
    between = "sample"

    # Number of groups
    r = data[between].nunique()
    ddof1 = r - 1

    # Compute weights and ajusted means
    grp = data.groupby(between, observed=True, group_keys=False)[dv]
    weights = grp.count() / grp.var()
    adj_grandmean = (weights * grp.mean()).sum() / weights.sum()
    logging.info("weights: %s, adj_grandmean: %s", weights, adj_grandmean)

    # Sums of squares (regular and adjusted)
    ss_res = grp.apply(lambda x: (x - x.mean()) ** 2).sum()
    ss_bet = ((grp.mean(numeric_only=True) - data[dv].mean()) ** 2 * grp.count()).sum()
    logging.info("grp.mean: %s, data.mean: %s, grp.count: %s", grp.mean(numeric_only=True), data[dv].mean(), grp.count())
    ss_betadj = np.sum(weights * np.square(grp.mean(numeric_only=True) - adj_grandmean))
    ms_betadj = ss_betadj / ddof1
    logging.info("ss_res: %s, ss_bet: %s, ss_betadj: %s, ms_betadj: %s", ss_res, ss_bet, ss_betadj, ms_betadj)

    # Calculate lambda, F-value, p-value and np2
    lamb = (3 * np.sum((1 / (grp.count() - 1)) * (1 - (weights / weights.sum())) ** 2)) / (
        r**2 - 1
    )
    logging.info("lamb0: %s, cnt: %s, weighs: %s, weightsSum: %s", np.sum((1 / (grp.count() - 1)) * (1 - (weights / weights.sum())) ** 2), grp.count(), weights, weights.sum())
    fval = ms_betadj / (1 + (2 * lamb * (r - 2)) / 3)
    logging.info("lamb: %s, fval: %s", lamb, fval)
    pval = stats.f.sf(fval, ddof1, 1 / lamb)
    np2 = ss_bet / (ss_bet + ss_res)

    # Create output dataframe
    aov = pd.DataFrame(
        {
            "Source": between,
            "ddof1": ddof1,
            "ddof2": 1 / lamb,
            "F": fval,
            "p-unc": pval,
            "np2": np2,
        },
        index=[0],
    )

    return aov.iloc[0]


LeveneResult = collections.namedtuple('LeveneResult', ('statistic', 'pvalue'))


def levene(*samples, center='median', proportiontocut=0.05):
    k = len(samples)
    Ni = np.empty(k)
    Yci = np.empty(k, 'd')
    def func(x):
        return np.median(x, axis=0)

    for j in range(k):
        Ni[j] = len(samples[j])
        Yci[j] = func(samples[j])
    Ntot = np.sum(Ni, axis=0)
    logging.info("k: %s, Ni: %s, Yci: %s, Ntot %s", k, Ni, Yci, Ntot)

    # compute Zij's
    Zij = [None] * k
    for i in range(k):
        Zij[i] = abs(np.asarray(samples[i]) - Yci[i])

    # compute Zbari
    Zbari = np.empty(k, 'd')
    Zbar = 0.0
    for i in range(k):
        Zbari[i] = np.mean(Zij[i], axis=0)
        Zbar += Zbari[i] * Ni[i]

    Zbar /= Ntot
    numer = (Ntot - k) * np.sum(Ni * (Zbari - Zbar)**2, axis=0)
    logging.info("ZBari: %s, Zbar: %s, numer: %s", Zbari, Zbar, numer)

    # compute denom_variance
    dvar = 0.0
    for i in range(k):
        dvar += np.sum((Zij[i] - Zbari[i])**2, axis=0)

    denom = (k - 1.0) * dvar
    logging.info("dvar: %s, denom: %s", dvar, denom)

    W = numer / denom
    pval = stats.distributions.f.sf(W, k-1, Ntot-k)  # 1 - cdf
    return LeveneResult(W, pval)


def main():
    logging.basicConfig()
    lg = logging.getLogger()
    lg.setLevel(logging.INFO)
    lg.handlers[0].setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(pathname)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    cfg = json.loads(sys.argv[1])
    if cfg["Func"] == "Levene":
        scipyOut = stats.levene(*cfg["Samples"])
        # scipyOut = levene(*cfg["Samples"])
        out = {"Levene": {"Statistic": scipyOut.statistic, "PValue": scipyOut.pvalue}}
    elif cfg["Func"] == "Welch":
        global pd, pingouin
        import pandas as pd
        import pingouin
        pingouinOut = pingouin_welch(cfg["Samples"])
        # pingouinOut = welch(cfg["Samples"])
        out = {"Welch": {"F": pingouinOut["F"], "PValue": pingouinOut["p-unc"]}}
    elif cfg["Func"] == "Holm":
        global pd, pingouin
        import pandas as pd
        import pingouin
        reject, pvalsCorr = pingouin.multicomp(cfg["Samples"][0], method="holm")
        out = {"Holm": {"Reject": reject.tolist(), "PValuesCorrected": pvalsCorr.tolist()}}
    else:
        logging.info("unknown function \"%s\"", cfg["Func"])
    print(json.dumps(out))


if __name__ == "__main__":
    main()
