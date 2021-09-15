import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
import SS_learning as st
import multiprocessing as mp
import os

############################################################################################################
scenario = 12
m = 3
d = 10  ##dimension of features
nindpt = 1000

cost_vec = np.array([1/128,1/64,1/32,1/16,1/8,1/4,1/2,1.0])
#gamma_vec = np.array([1/128,1/64,1/32,1/16,1/8,1/4,1/2,1.0])
#param_grid = dict(gamma=gamma_vec, C=cost_vec)
tuned_paras = [{'C': cost_vec}]
lam2_vec = np.arange(0, 1.1, 0.1)
lam3_vec = np.arange(0, 1.1, 0.1)

itermax = 50
itertol = 1e-4

dist = "uniform"
studySetting = "observational"
weightingSetting = "IPW"
kernel = "linear"
tuneModel = "linear"

############################################################################################################
def simulation(seed_base):

    np.random.seed(1234+seed_base)

    ########################################################################################################
    ## matrix to store the final results
    cost_tune = np.zeros(scenario)
    #gamma_tune = np.zeros(scenario)
    lam2_tune = np.zeros(scenario)
    lam3_tune = np.zeros(scenario)
    cost_opt = np.zeros(scenario)
    #gamma_opt = np.zeros(scenario)
    lam2_opt = np.zeros(scenario)
    lam3_opt = np.zeros(scenario)
    acc_all_1CV = np.zeros(scenario)
    acc_all_allCV = np.zeros(scenario)
    acc_all_tune = np.zeros(scenario)
    acc_all_opt = np.zeros(scenario)
    acc_indpt_1CV = np.zeros(scenario)
    acc_indpt_allCV = np.zeros(scenario)
    acc_indpt_tune = np.zeros(scenario)
    acc_indpt_opt = np.zeros(scenario)
    evf_1CV = np.zeros(m*scenario)
    evf_allCV = np.zeros(m*scenario)
    evf_tune = np.zeros(m*scenario)
    evf_opt = np.zeros(m*scenario)
    ts_1CV = np.zeros(scenario)
    ts_allCV = np.zeros(scenario)
    ts_tune = np.zeros(scenario)
    ts_opt = np.zeros(scenario)
    time_1CV = np.zeros(scenario)
    time_allCV = np.zeros(scenario)
    time_tune = np.zeros(scenario)

    for ss in range(scenario):

        sce = st.parScenario()
        sce.generatePar(ss)
        n1, n2, n3 = sce.n1, sce.n2, sce.n3
        coef1, coef2, coef3 = sce.coef1, sce.coef2, sce.coef3
        sd1, sd2, sd3 = sce.sd1, sce.sd2, sce.sd3

        dataLabel = np.repeat(range(m), (n1,n2,n3))
        ########################################################################################################
        ## generate the simulation datasets ####################################################################
        data = st.dataGeneration()

        data.simIndpt(nindpt, d, dist)
        Xindpt, Tindpt = data.Xindpt, data.Tindpt

        data.sim3(n1, n2, n3, d, coef1, sd1, coef2, sd2, dist, studySetting)
        X1, A1, T1, B1 = data.X1, data.A1, data.T1, data.B1
        X2, A2, T2, B2 = data.X2, data.A2, data.T2, data.B2
        X3, A3, T3 = data.X3, data.A3, data.T3

        data.simB3(coef3, sd3)
        B3 = data.B3

        ######################################################################################################
        Xall = np.concatenate((X1, X2, X3), axis=0)
        Aall = np.concatenate((A1, A2, A3))
        Tall = np.concatenate((T1, T2, T3))
        Ball = np.concatenate((B1, B2, B3))

        #########################################################################################################
        ## determine propensity score ###########################################################################
        propenScore = st.propensityScore(Xall, Aall, m, dataLabel)
        pall = propenScore.p(studySetting)

        ######################################################################################################
        ## 5-fold CV on S_B1 #################################################################################
        start_time_1CV = time.time()

        if weightingSetting == "IPW":
            sample_weight1 = B1/pall[:n1]
        elif weightingSetting == "overlap":
            sample_weight1 = B1*(1-pall[:n1])

        model1 = GridSearchCV(svm.SVC(kernel='linear'), tuned_paras, cv=5, scoring='accuracy',
                              fit_params={'sample_weight': sample_weight1})
        model1.fit(X1, A1)

        time_1CV[ss] = time.time()-start_time_1CV

        predAll_model1 = model1.best_estimator_.predict(Xall)
        acc_all_1CV[ss] = st.evalPred(predAll_model1, Tall).acc()
        predIndpt_model1 = model1.best_estimator_.predict(Xindpt)
        acc_indpt_1CV[ss] = st.evalPred(predIndpt_model1, Tindpt).acc()

        ts1 = st.tuneStat(Xall, Aall, Ball, m, dataLabel, model1)
        ts_1CV[ss] = ts1.tsSSE(model=tuneModel)

        evf1 = st.EVF()
        evf1.evfCal(Xall, Aall, Ball, m, dataLabel, model1)
        evf_1CV[(m*ss):(m*(ss+1))] = evf1.evfSeq

        ############################################################################################################
        ## 5-fold CV on S ##########################################################################################
        start_time_allCV = time.time()

        if weightingSetting == "IPW":
            sample_weightAll = Ball/pall
        elif weightingSetting == "overlap":
            sample_weightAll = Ball*(1-pall)

        modelAll = GridSearchCV(svm.SVC(kernel='linear'), tuned_paras, cv=5, scoring='accuracy',
                                fit_params={'sample_weight': sample_weightAll})
        modelAll.fit(Xall, Aall)

        time_allCV[ss] = time.time()-start_time_allCV

        predAll_modelAll = modelAll.best_estimator_.predict(Xall)
        acc_all_allCV[ss] = st.evalPred(predAll_modelAll, Tall).acc()
        predIndpt_modelAll = modelAll.best_estimator_.predict(Xindpt)
        acc_indpt_allCV[ss] = st.evalPred(predIndpt_modelAll, Tindpt).acc()

        tsAll = st.tuneStat(Xall, Aall, Ball, m, dataLabel, modelAll)
        ts_allCV[ss] = tsAll.tsSSE(model=tuneModel)

        evfAll = st.EVF()
        evfAll.evfCal(Xall, Aall, Ball, m, dataLabel, modelAll)
        evf_allCV[(m*ss):(m*(ss+1))] = evfAll.evfSeq

        ############################################################################################################
        ## SS-learning
        ## store the results at convergence for each cost and lambda pair

        conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan)  ##matrix
        obj_conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan)  ##matrix
        ts_conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan)  ##matrix
        evf_conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0], m], np.nan)  ##matrix
        acc_all_conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan) ##matrix
        acc_indpt_conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan) ##matrix

        start_time_tune = time.time()

        out = st.SS_learning(Xall, Aall, Ball, m, dataLabel, pall, weightingSetting, kernel)

        for ii in np.arange(cost_vec.shape[0]):
            # initial fitting
            out.iniFit(cost_vec[ii])

            for kk in np.arange(lam2_vec.shape[0]):
                for ll in np.arange(lam3_vec.shape[0]):

                    lam = [lam2_vec[kk], lam3_vec[ll]]

                    out.fit(lam, itermax, itertol, track=True, tuning='SSE', model=tuneModel)

                    conv[ii, kk, ll] = out.conv

                    if (out.conv != 99):
                        obj_conv[ii, kk, ll] = out.objConv
                        ts_conv[ii, kk, ll] = out.tsConv
                        acc_all_conv[ii, kk, ll] = st.evalPred(out.predConv, Tall).acc()
                        predIndpt = out.predict(Xindpt, track=False)
                        acc_indpt_conv[ii, kk, ll] = st.evalPred(predIndpt, Tindpt).acc()
                        evfConv = st.EVF()
                        evfConv.evfCal(Xall, Aall, Ball, m, dataLabel, out.modelConv)
                        evf_conv[ii, kk, ll, :] = evfConv.evfSeq
                    else:
                        obj_conv[ii, kk, ll] = 1e5
                        ts_conv[ii, kk, ll] = 0
                        acc_all_conv[ii, kk, ll] = 0
                        acc_indpt_conv[ii, kk, ll] = 0
                        evf_conv[ii, kk, ll, :] = 0

        time_tune[ss] = time.time()-start_time_tune

        opt_index = st.evalTune()
        opt_index.maxTune(ts_conv, acc_all_conv, method='min')
        cost_tune_idx = opt_index.tune_idx[0]
        cost_opt_idx = opt_index.opt_idx[0]
        #gamma_tune_idx = opt_index.tune_idx[1]
        #gamma_opt_idx = opt_index.opt_idx[1]
        lam2_tune_idx = opt_index.tune_idx[1]
        lam2_opt_idx = opt_index.opt_idx[1]
        lam3_tune_idx = opt_index.tune_idx[2]
        lam3_opt_idx = opt_index.opt_idx[2]

        cost_tune[ss] = cost_vec[cost_tune_idx]
        #gamma_tune[ss] = gamma_vec[gamma_tune_idx]
        lam2_tune[ss] = lam2_vec[lam2_tune_idx]
        lam3_tune[ss] = lam3_vec[lam3_tune_idx]
        cost_opt[ss] = cost_vec[cost_opt_idx]
        #gamma_opt[ss] = gamma_vec[gamma_opt_idx]
        lam2_opt[ss] = lam2_vec[lam2_opt_idx]
        lam3_opt[ss] = lam3_vec[lam3_opt_idx]
        acc_all_tune[ss] = acc_all_conv[cost_tune_idx, lam2_tune_idx, lam3_tune_idx]
        acc_all_opt[ss] = acc_all_conv[cost_opt_idx, lam2_opt_idx, lam3_opt_idx]
        acc_indpt_tune[ss] = acc_indpt_conv[cost_tune_idx, lam2_tune_idx, lam3_tune_idx]
        acc_indpt_opt[ss] = acc_indpt_conv[cost_opt_idx, lam2_opt_idx, lam3_opt_idx]
        ts_tune[ss] = ts_conv[cost_tune_idx, lam2_tune_idx, lam3_tune_idx]
        ts_opt[ss] = ts_conv[cost_opt_idx, lam2_opt_idx, lam3_opt_idx]
        evf_tune[(m*ss):(m*(ss+1))] = evf_conv[cost_tune_idx, lam2_tune_idx, lam3_tune_idx,:]
        evf_opt[(m*ss):(m*(ss+1))] = evf_conv[cost_opt_idx, lam2_opt_idx, lam3_opt_idx,:]

    dresult = dict()
    dresult['cost_tune'] = cost_tune
    #dresult['gamma_tune'] = gamma_tune
    dresult['lam2_tune'] = lam2_tune
    dresult['lam3_tune'] = lam3_tune
    dresult['cost_opt'] = cost_opt
    #dresult['gamma_opt'] = gamma_opt
    dresult['lam2_opt'] = lam2_opt
    dresult['lam3_opt'] = lam3_opt
    dresult['acc_all_1CV'] = acc_all_1CV
    dresult['acc_all_allCV'] = acc_all_allCV
    dresult['acc_all_tune'] = acc_all_tune
    dresult['acc_all_opt'] = acc_all_opt
    dresult['acc_indpt_1CV'] = acc_indpt_1CV
    dresult['acc_indpt_allCV'] = acc_indpt_allCV
    dresult['acc_indpt_tune'] = acc_indpt_tune
    dresult['acc_indpt_opt'] = acc_indpt_opt
    dresult['evf_1CV'] = evf_1CV
    dresult['evf_allCV'] = evf_allCV
    dresult['evf_tune'] = evf_tune
    dresult['evf_opt'] = evf_opt
    dresult['ts_1CV'] = ts_1CV
    dresult['ts_allCV'] = ts_allCV
    dresult['ts_tune'] = ts_tune
    dresult['ts_opt'] = ts_opt
    dresult['time_1CV'] = time_1CV
    dresult['time_allCV'] = time_allCV
    dresult['time_tune'] = time_tune
    return(dresult)

############################################################################################################
if __name__ == '__main__':
    '''
    ncpus = 1
    pool = mp.Pool(processes=ncpus)
    len_replicate = ncpus
    slurm_index = int(os.environ["PBS_ARRAYID"])
    slurm_index_str = str(slurm_index)
    results = pool.map(simulation, range(ncpus*(slurm_index-1), ncpus*slurm_index))
    '''

    ncpus = 1
    pool = mp.Pool(processes=ncpus)
    len_replicate = ncpus
    slurm_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    slurm_index_str = str(slurm_index)
    results = pool.map(simulation, range(ncpus*(slurm_index - 1), ncpus*slurm_index))

    '''
    pool = mp.Pool(processes=4)
    len_replicate = 1
    results = pool.map(simulation, range(len_replicate))
    '''

    cost_tune = np.row_stack(results[i]['cost_tune'] for i in range(len_replicate))
    #gamma_tune = np.row_stack(results[i]['gamma_tune'] for i in range(len_replicate))
    lam2_tune = np.row_stack(results[i]['lam2_tune'] for i in range(len_replicate))
    lam3_tune = np.row_stack(results[i]['lam3_tune'] for i in range(len_replicate))
    cost_opt = np.row_stack(results[i]['cost_opt'] for i in range(len_replicate))
    #gamma_opt = np.row_stack(results[i]['gamma_opt'] for i in range(len_replicate))
    lam2_opt = np.row_stack(results[i]['lam2_opt'] for i in range(len_replicate))
    lam3_opt = np.row_stack(results[i]['lam3_opt'] for i in range(len_replicate))
    acc_all_1CV = np.row_stack(results[i]['acc_all_1CV'] for i in range(len_replicate))
    acc_all_allCV = np.row_stack(results[i]['acc_all_allCV'] for i in range(len_replicate))
    acc_all_tune = np.row_stack(results[i]['acc_all_tune'] for i in range(len_replicate))
    acc_all_opt = np.row_stack(results[i]['acc_all_opt'] for i in range(len_replicate))
    acc_indpt_1CV = np.row_stack(results[i]['acc_indpt_1CV'] for i in range(len_replicate))
    acc_indpt_allCV = np.row_stack(results[i]['acc_indpt_allCV'] for i in range(len_replicate))
    acc_indpt_tune = np.row_stack(results[i]['acc_indpt_tune'] for i in range(len_replicate))
    acc_indpt_opt = np.row_stack(results[i]['acc_indpt_opt'] for i in range(len_replicate))
    evf_1CV = np.row_stack(results[i]['evf_1CV'] for i in range(len_replicate))
    evf_allCV = np.row_stack(results[i]['evf_allCV'] for i in range(len_replicate))
    evf_tune = np.row_stack(results[i]['evf_tune'] for i in range(len_replicate))
    evf_opt = np.row_stack(results[i]['evf_opt'] for i in range(len_replicate))
    ts_1CV = np.row_stack(results[i]['ts_1CV'] for i in range(len_replicate))
    ts_allCV = np.row_stack(results[i]['ts_allCV'] for i in range(len_replicate))
    ts_tune = np.row_stack(results[i]['ts_tune'] for i in range(len_replicate))
    ts_opt = np.row_stack(results[i]['ts_opt'] for i in range(len_replicate))
    time_1CV = np.row_stack(results[i]['time_1CV'] for i in range(len_replicate))
    time_allCV = np.row_stack(results[i]['time_allCV'] for i in range(len_replicate))
    time_tune = np.row_stack(results[i]['time_tune'] for i in range(len_replicate))

    ### save files ###############################################################
    np.savetxt("cost_tune_"+slurm_index_str+".txt", cost_tune, delimiter=",")
    #np.savetxt("gamma_tune_"+slurm_index_str+".txt", gamma_tune, delimiter=",")
    np.savetxt("lam2_tune_"+slurm_index_str+".txt", lam2_tune, delimiter=",")
    np.savetxt("lam3_tune_"+slurm_index_str+".txt", lam3_tune, delimiter=",")
    np.savetxt("cost_opt_"+slurm_index_str+".txt", cost_opt, delimiter=",")
    #np.savetxt("gamma_opt_"+slurm_index_str+".txt", gamma_opt, delimiter=",")
    np.savetxt("lam2_opt_"+slurm_index_str+".txt", lam2_opt, delimiter=",")
    np.savetxt("lam3_opt_"+slurm_index_str+".txt", lam3_opt, delimiter=",")
    np.savetxt("acc_all_1CV_" + slurm_index_str + ".txt", acc_all_1CV, delimiter=",")
    np.savetxt("acc_all_allCV_" + slurm_index_str + ".txt", acc_all_allCV, delimiter=",")
    np.savetxt("acc_all_tune_"+slurm_index_str+".txt", acc_all_tune, delimiter=",")
    np.savetxt("acc_all_opt_"+slurm_index_str+".txt", acc_all_opt, delimiter=",")
    np.savetxt("acc_indpt_1CV_" + slurm_index_str + ".txt", acc_indpt_1CV, delimiter=",")
    np.savetxt("acc_indpt_allCV_" + slurm_index_str + ".txt", acc_indpt_allCV, delimiter=",")
    np.savetxt("acc_indpt_tune_"+slurm_index_str+".txt", acc_indpt_tune, delimiter=",")
    np.savetxt("acc_indpt_opt_"+slurm_index_str+".txt", acc_indpt_opt, delimiter=",")
    np.savetxt("evf_1CV_" + slurm_index_str + ".txt", evf_1CV, delimiter=",")
    np.savetxt("evf_allCV_" + slurm_index_str + ".txt", evf_allCV, delimiter=",")
    np.savetxt("evf_tune_" + slurm_index_str + ".txt", evf_tune, delimiter=",")
    np.savetxt("evf_opt_" + slurm_index_str + ".txt", evf_opt, delimiter=",")
    np.savetxt("ts_1CV_" + slurm_index_str + ".txt", ts_1CV, delimiter=",")
    np.savetxt("ts_allCV_" + slurm_index_str + ".txt", ts_allCV, delimiter=",")
    np.savetxt("ts_tune_"+slurm_index_str+".txt", ts_tune, delimiter=",")
    np.savetxt("ts_opt_"+slurm_index_str+".txt", ts_opt, delimiter=",")
    np.savetxt("time_1CV_"+slurm_index_str+".txt", time_1CV, delimiter=",")
    np.savetxt("time_allCV_"+slurm_index_str+".txt", time_allCV, delimiter=",")
    np.savetxt("time_tune_"+slurm_index_str+".txt", time_tune, delimiter=",")

    '''
    np.savetxt("cost_tune.txt", cost_tune, delimiter=",")
    #np.savetxt("gamma_tune.txt", gamma_tune, delimiter=",")
    np.savetxt("lam2_tune.txt", lam2_tune, delimiter=",")
    np.savetxt("lam3_tune.txt", lam3_tune, delimiter=",")
    np.savetxt("cost_opt.txt", cost_opt, delimiter=",")
    #np.savetxt("gamma_opt.txt", gamma_opt, delimiter=",")
    np.savetxt("lam2_opt.txt", lam2_opt, delimiter=",")
    np.savetxt("lam3_opt.txt", lam3_opt, delimiter=",")
    np.savetxt("acc_all_1CV.txt", acc_all_1CV, delimiter=",")
    np.savetxt("acc_all_allCV.txt", acc_all_allCV, delimiter=",")
    np.savetxt("acc_all_tune.txt", acc_all_tune, delimiter=",")
    np.savetxt("acc_all_opt.txt", acc_all_opt, delimiter=",")
    np.savetxt("acc_indpt_1CV.txt", acc_indpt_1CV, delimiter=",")
    np.savetxt("acc_indpt_allCV.txt", acc_indpt_allCV, delimiter=",")
    np.savetxt("acc_indpt_tune.txt", acc_indpt_tune, delimiter=",")
    np.savetxt("acc_indpt_opt.txt", acc_indpt_opt, delimiter=",")
    np.savetxt("evf_1CV.txt", evf_1CV, delimiter=",")
    np.savetxt("evf_allCV.txt", evf_allCV, delimiter=",")
    np.savetxt("evf_tune.txt", evf_tune, delimiter=",")
    np.savetxt("evf_opt.txt", evf_opt, delimiter=",")
    np.savetxt("ts_1CV.txt", ts_1CV, delimiter=",")
    np.savetxt("ts_allCV.txt", ts_allCV, delimiter=",")
    np.savetxt("ts_tune.txt", ts_tune, delimiter=",")
    np.savetxt("ts_opt.txt", ts_opt, delimiter=",")
    np.savetxt("time_1CV.txt", time_1CV, delimiter=",")
    np.savetxt("time_allCV.txt", time_allCV, delimiter=",")
    np.savetxt("time_tune.txt", time_tune, delimiter=",")
    '''