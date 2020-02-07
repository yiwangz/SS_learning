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
m = 3  ##number of total datasets

n1 = 20
n2 = 100
n3 = 380
nindpt = 1000

dataLabel = np.repeat(range(m), (n1,n2,n3))  ##label of subjects from S1, S2, S3

d = 10  ##dimension of features
coef1 = 3.0
coef2 = 1.0
coef3 = 0.5
sd1 = 0.1
sd2 = 0.5
sd3_vec = [0.1,0.5,1.0,2.0] ##varying B3 sd

cost_vec = np.array([1/128,1/64,1/32,1/16,1/8,1/4,1/2,1.0])
tuned_paras = [{'C': cost_vec}]
lam2_vec = np.arange(0, 1.1, 0.1)
lam3_vec = np.arange(0, 1.1, 0.1)

itermax = 50
itertol = 1e-4

studySetting = "trial"

############################################################################################################
def simulation(seed_base):

    np.random.seed(1234+seed_base)

    ########################################################################################################
    ## generate the simulation datasets ####################################################################
    data = st.dataGeneration()
    data.simIndpt(nindpt, d, dist="uniform")
    Xindpt, Tindpt = data.Xindpt, data.Tindpt
    data.sim3(n1, n2, n3, d, coef1, sd1, coef2, sd2, dist='uniform')
    X1, A1, T1, B1 = data.X1, data.A1, data.T1, data.B1
    X2, A2, T2, B2 = data.X2, data.A2, data.T2, data.B2
    X3, A3, T3 = data.X3, data.A3, data.T3

    B3_mat = np.full([len(sd3_vec), n3], np.nan)
    for i in range(len(sd3_vec)):
        data.simB3(coef3, sd3_vec[i])
        B3_mat[i,:] = data.B3

    ######################################################################################################
    Xall = np.concatenate((X1, X2, X3), axis=0)
    Aall = np.concatenate((A1, A2, A3))
    Tall = np.concatenate((T1, T2, T3))

    #########################################################################################################
    ## determine propensity score ###########################################################################
    propenScore = st.propensityScore(Xall, Aall, m, dataLabel)
    pall = propenScore.p(studySetting=studySetting)

    #########################################################################################################
    ## vectors to save the accuracy results #################################################################
    cost_tune = np.zeros(len(sd3_vec))
    lam2_tune = np.zeros(len(sd3_vec))
    lam3_tune = np.zeros(len(sd3_vec))
    cost_opt = np.zeros(len(sd3_vec))
    lam2_opt = np.zeros(len(sd3_vec))
    lam3_opt = np.zeros(len(sd3_vec))
    acc_all_tune = np.zeros(len(sd3_vec))
    acc_all_opt = np.zeros(len(sd3_vec))
    acc_all_1CV = np.zeros(len(sd3_vec))
    acc_all_allCV = np.zeros(len(sd3_vec))
    acc_indpt_tune = np.zeros(len(sd3_vec))
    acc_indpt_opt = np.zeros(len(sd3_vec))
    acc_indpt_1CV = np.zeros(len(sd3_vec))
    acc_indpt_allCV = np.zeros(len(sd3_vec))
    evf_B1_tune = np.zeros(len(sd3_vec))
    evf_B1_opt = np.zeros(len(sd3_vec))
    evf_B1_1CV = np.zeros(len(sd3_vec))
    evf_B1_allCV = np.zeros(len(sd3_vec))
    evf_B2_tune = np.zeros(len(sd3_vec))
    evf_B2_opt = np.zeros(len(sd3_vec))
    evf_B2_1CV = np.zeros(len(sd3_vec))
    evf_B2_allCV = np.zeros(len(sd3_vec))
    evf_B3_tune = np.zeros(len(sd3_vec))
    evf_B3_opt = np.zeros(len(sd3_vec))
    evf_B3_1CV = np.zeros(len(sd3_vec))
    evf_B3_allCV = np.zeros(len(sd3_vec))
    ts_tune = np.zeros(len(sd3_vec))
    ts_opt = np.zeros(len(sd3_vec))
    ts_1CV = np.zeros(len(sd3_vec))
    ts_allCV = np.zeros(len(sd3_vec))
    time_1CV = np.zeros(len(sd3_vec))
    time_allCV = np.zeros(len(sd3_vec))
    time_tune = np.zeros(len(sd3_vec))

    for index in range(len(sd3_vec)):

        ######################################################################################################
        Ball = np.concatenate((B1, B2, B3_mat[index,:]))

        ######################################################################################################
        ## 5-fold CV on S_B1 only ############################################################################
        start_time_1CV = time.time()

        model1 = GridSearchCV(svm.SVC(kernel='linear'), tuned_paras, cv=5, scoring='accuracy', fit_params={'sample_weight': B1/pall[:n1]})
        model1.fit(X1, A1)

        time_1CV[index] = time.time()-start_time_1CV

        predAll_model1 = model1.best_estimator_.predict(Xall)
        acc_all_model1 = st.evalPred(predAll_model1, Tall).acc()
        predIndpt_model1 = model1.best_estimator_.predict(Xindpt)
        acc_indpt_model1 = st.evalPred(predIndpt_model1, Tindpt).acc()

        acc_all_1CV[index] = acc_all_model1
        acc_indpt_1CV[index] = acc_indpt_model1

        ts1 = st.tuneStat(Xall, Aall, Ball, m, dataLabel, model1)
        ts_1CV[index] = ts1.tsSSE(model='linear')

        evf1 = st.EVF()
        evf1.evfCal(Xall, Aall, Ball, m, dataLabel, model1)
        evf_B1_1CV[index] = evf1.evfSeq[0]
        evf_B2_1CV[index] = evf1.evfSeq[1]
        evf_B3_1CV[index] = evf1.evfSeq[2]

        ############################################################################################################
        ## 5-fold CV on S_B1 and S_B2 ##############################################################################
        start_time_allCV = time.time()

        model123 = GridSearchCV(svm.SVC(kernel='linear'), tuned_paras, cv=5, scoring='accuracy',fit_params={'sample_weight': Ball/pall})
        model123.fit(Xall, Aall)

        time_allCV[index] = time.time()-start_time_allCV

        predAll_model123 = model123.best_estimator_.predict(Xall)
        acc_all_model123 = st.evalPred(predAll_model123, Tall).acc()
        predIndpt_model123 = model123.best_estimator_.predict(Xindpt)
        acc_indpt_model123 = st.evalPred(predIndpt_model123, Tindpt).acc()

        acc_all_allCV[index] = acc_all_model123
        acc_indpt_allCV[index] = acc_indpt_model123

        ts123 = st.tuneStat(Xall, Aall, Ball, m, dataLabel, model123)
        ts_allCV[index] = ts123.tsSSE(model='linear')

        evf123 = st.EVF()
        evf123.evfCal(Xall, Aall, Ball, m, dataLabel, model123)
        evf_B1_allCV[index] = evf123.evfSeq[0]
        evf_B2_allCV[index] = evf123.evfSeq[1]
        evf_B3_allCV[index] = evf123.evfSeq[2]

        ############################################################################################################
        ## proposed method
        ## store the results at convergence for each cost and lambda pair

        conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan)  ##matrix
        obj_conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan)  ##matrix
        ts_conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan)  ##matrix
        evf_B1_conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan)  ##matrix
        evf_B2_conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan)  ##matrix
        evf_B3_conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan)  ##matrix
        acc_all_conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan) ##matrix
        acc_indpt_conv = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0]], np.nan) ##matrix

        start_time_tune = time.time()

        out = st.STowlLinear(Xall, Aall, Ball, m, dataLabel, pall)

        for ii in np.arange(cost_vec.shape[0]):
            # initial fitting
            out.iniFit(cost_vec[ii])

            for jj in np.arange(lam2_vec.shape[0]):

                for kk in np.arange(lam3_vec.shape[0]):

                    lam = [lam2_vec[jj], lam3_vec[kk]]

                    out.fit(lam, itermax, itertol, track=True, tuning='SSE', model='linear')

                    conv[ii,jj,kk] = out.conv

                    if(out.conv != 99):
                        obj_conv[ii,jj,kk] = out.objConv
                        ts_conv[ii,jj,kk] = out.tsConv
                        acc_all_conv[ii,jj,kk] = st.evalPred(out.predConv, Tall).acc()
                        predIndpt = out.predict(Xindpt, track=False)
                        acc_indpt_conv[ii,jj,kk] = st.evalPred(predIndpt, Tindpt).acc()
                        evfConv = st.EVF()
                        evfConv.evfCal(Xall, Aall, Ball, m, dataLabel, out.modelConv)
                        evf_B1_conv[ii,jj,kk] = evfConv.evfSeq[0]
                        evf_B2_conv[ii,jj,kk] = evfConv.evfSeq[1]
                        evf_B3_conv[ii,jj,kk] = evfConv.evfSeq[2]
                    else:
                        obj_conv[ii,jj,kk] = 1e5
                        ts_conv[ii,jj,kk] = 0
                        acc_all_conv[ii,jj,kk] = 0
                        acc_indpt_conv[ii,jj,kk] = 0
                        evf_B1_conv[ii,jj,kk] = 0
                        evf_B2_conv[ii,jj,kk] = 0
                        evf_B3_conv[ii,jj,kk] = 0

        time_tune[index] = time.time()-start_time_tune

        opt_index = st.evalTune()
        opt_index.maxTune(ts_conv, acc_all_conv, method='min')
        cost_tune_idx = opt_index.tune_idx[0]
        cost_opt_idx = opt_index.opt_idx[0]
        lam2_tune_idx = opt_index.tune_idx[1]
        lam2_opt_idx = opt_index.opt_idx[1]
        lam3_tune_idx = opt_index.tune_idx[2]
        lam3_opt_idx = opt_index.opt_idx[2]

        cost_tune[index] = cost_vec[cost_tune_idx]
        lam2_tune[index] = lam2_vec[lam2_tune_idx]
        lam3_tune[index] = lam3_vec[lam3_tune_idx]
        cost_opt[index] = cost_vec[cost_opt_idx]
        lam2_opt[index] = lam2_vec[lam2_opt_idx]
        lam3_opt[index] = lam3_vec[lam3_opt_idx]
        acc_all_tune[index] = acc_all_conv[cost_tune_idx, lam2_tune_idx, lam3_tune_idx]
        acc_all_opt[index] = acc_all_conv[cost_opt_idx, lam2_opt_idx, lam3_opt_idx]
        acc_indpt_tune[index] = acc_indpt_conv[cost_tune_idx, lam2_tune_idx, lam3_tune_idx]
        acc_indpt_opt[index] = acc_indpt_conv[cost_opt_idx, lam2_opt_idx, lam3_opt_idx]
        ts_tune[index] = ts_conv[cost_tune_idx, lam2_tune_idx, lam3_tune_idx]
        ts_opt[index] = ts_conv[cost_opt_idx, lam2_opt_idx, lam3_opt_idx]
        evf_B1_tune[index] = evf_B1_conv[cost_tune_idx, lam2_tune_idx, lam3_tune_idx]
        evf_B1_opt[index] = evf_B1_conv[cost_opt_idx, lam2_opt_idx, lam3_opt_idx]
        evf_B2_tune[index] = evf_B2_conv[cost_tune_idx, lam2_tune_idx, lam3_tune_idx]
        evf_B2_opt[index] = evf_B2_conv[cost_opt_idx, lam2_opt_idx, lam3_opt_idx]
        evf_B3_tune[index] = evf_B3_conv[cost_tune_idx, lam2_tune_idx, lam3_tune_idx]
        evf_B3_opt[index] = evf_B3_conv[cost_opt_idx, lam2_opt_idx, lam3_opt_idx]

    dresult = dict()
    dresult['cost_tune'] = cost_tune
    dresult['lam2_tune'] = lam2_tune
    dresult['lam3_tune'] = lam3_tune
    dresult['cost_opt'] = cost_opt
    dresult['lam2_opt'] = lam2_opt
    dresult['lam3_opt'] = lam3_opt
    dresult['acc_all_tune'] = acc_all_tune
    dresult['acc_all_opt'] = acc_all_opt
    dresult['acc_all_1CV'] = acc_all_1CV
    dresult['acc_all_allCV'] = acc_all_allCV
    dresult['acc_indpt_tune'] = acc_indpt_tune
    dresult['acc_indpt_opt'] = acc_indpt_opt
    dresult['acc_indpt_1CV'] = acc_indpt_1CV
    dresult['acc_indpt_allCV'] = acc_indpt_allCV
    dresult['evf_B1_tune'] = evf_B1_tune
    dresult['evf_B1_opt'] = evf_B1_opt
    dresult['evf_B1_1CV'] = evf_B1_1CV
    dresult['evf_B1_allCV'] = evf_B1_allCV
    dresult['evf_B2_tune'] = evf_B2_tune
    dresult['evf_B2_opt'] = evf_B2_opt
    dresult['evf_B2_1CV'] = evf_B2_1CV
    dresult['evf_B2_allCV'] = evf_B2_allCV
    dresult['evf_B3_tune'] = evf_B3_tune
    dresult['evf_B3_opt'] = evf_B3_opt
    dresult['evf_B3_1CV'] = evf_B3_1CV
    dresult['evf_B3_allCV'] = evf_B3_allCV
    dresult['ts_tune'] = ts_tune
    dresult['ts_opt'] = ts_opt
    dresult['ts_1CV'] = ts_1CV
    dresult['ts_allCV'] = ts_allCV
    dresult['time_1CV'] = time_1CV
    dresult['time_allCV'] = time_allCV
    dresult['time_tune'] = time_tune
    return(dresult)

############################################################################################################
if __name__ == '__main__':

    ############################################################################################################
    ## run script in cluster
    '''ncpus = 1
    pool = mp.Pool(processes=ncpus)
    len_replicate = ncpus
    slurm_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    slurm_index_str = str(slurm_index)
    results = pool.map(simulation, range(ncpus*(slurm_index - 1), ncpus*slurm_index))'''

    ############################################################################################################
    ## run script in local computer
    pool = mp.Pool(processes=4)
    len_replicate = 1
    results = pool.map(simulation, range(len_replicate))

    cost_tune = np.row_stack(results[i]['cost_tune'] for i in range(len_replicate))
    lam2_tune = np.row_stack(results[i]['lam2_tune'] for i in range(len_replicate))
    lam3_tune = np.row_stack(results[i]['lam3_tune'] for i in range(len_replicate))
    cost_opt = np.row_stack(results[i]['cost_opt'] for i in range(len_replicate))
    lam2_opt = np.row_stack(results[i]['lam2_opt'] for i in range(len_replicate))
    lam3_opt = np.row_stack(results[i]['lam3_opt'] for i in range(len_replicate))
    acc_all_tune = np.row_stack(results[i]['acc_all_tune'] for i in range(len_replicate))
    acc_all_opt = np.row_stack(results[i]['acc_all_opt'] for i in range(len_replicate))
    acc_all_1CV = np.row_stack(results[i]['acc_all_1CV'] for i in range(len_replicate))
    acc_all_allCV = np.row_stack(results[i]['acc_all_allCV'] for i in range(len_replicate))
    acc_indpt_tune = np.row_stack(results[i]['acc_indpt_tune'] for i in range(len_replicate))
    acc_indpt_opt = np.row_stack(results[i]['acc_indpt_opt'] for i in range(len_replicate))
    acc_indpt_1CV = np.row_stack(results[i]['acc_indpt_1CV'] for i in range(len_replicate))
    acc_indpt_allCV = np.row_stack(results[i]['acc_indpt_allCV'] for i in range(len_replicate))
    evf_B1_tune = np.row_stack(results[i]['evf_B1_tune'] for i in range(len_replicate))
    evf_B1_opt = np.row_stack(results[i]['evf_B1_opt'] for i in range(len_replicate))
    evf_B1_1CV = np.row_stack(results[i]['evf_B1_1CV'] for i in range(len_replicate))
    evf_B1_allCV = np.row_stack(results[i]['evf_B1_allCV'] for i in range(len_replicate))
    evf_B2_tune = np.row_stack(results[i]['evf_B2_tune'] for i in range(len_replicate))
    evf_B2_opt = np.row_stack(results[i]['evf_B2_opt'] for i in range(len_replicate))
    evf_B2_1CV = np.row_stack(results[i]['evf_B2_1CV'] for i in range(len_replicate))
    evf_B2_allCV = np.row_stack(results[i]['evf_B2_allCV'] for i in range(len_replicate))
    evf_B3_tune = np.row_stack(results[i]['evf_B3_tune'] for i in range(len_replicate))
    evf_B3_opt = np.row_stack(results[i]['evf_B3_opt'] for i in range(len_replicate))
    evf_B3_1CV = np.row_stack(results[i]['evf_B3_1CV'] for i in range(len_replicate))
    evf_B3_allCV = np.row_stack(results[i]['evf_B3_allCV'] for i in range(len_replicate))
    ts_tune = np.row_stack(results[i]['ts_tune'] for i in range(len_replicate))
    ts_opt = np.row_stack(results[i]['ts_opt'] for i in range(len_replicate))
    ts_1CV = np.row_stack(results[i]['ts_1CV'] for i in range(len_replicate))
    ts_allCV = np.row_stack(results[i]['ts_allCV'] for i in range(len_replicate))
    time_1CV = np.row_stack(results[i]['time_1CV'] for i in range(len_replicate))
    time_allCV = np.row_stack(results[i]['time_allCV'] for i in range(len_replicate))
    time_tune = np.row_stack(results[i]['time_tune'] for i in range(len_replicate))

    ############################################################################################################
    ## save files script in cluster
    '''np.savetxt("cost_tune_"+slurm_index_str+".txt", cost_tune, delimiter=",")
    np.savetxt("lam2_tune_"+slurm_index_str+".txt", lam2_tune, delimiter=",")
    np.savetxt("lam3_tune_"+slurm_index_str+".txt", lam3_tune, delimiter=",")
    np.savetxt("cost_opt_"+slurm_index_str+".txt", cost_opt, delimiter=",")
    np.savetxt("lam2_opt_"+slurm_index_str+".txt", lam2_opt, delimiter=",")
    np.savetxt("lam3_opt_"+slurm_index_str+".txt", lam3_opt, delimiter=",")
    np.savetxt("acc_all_tune_"+slurm_index_str+".txt", acc_all_tune, delimiter=",")
    np.savetxt("acc_all_opt_"+slurm_index_str+".txt", acc_all_opt, delimiter=",")
    np.savetxt("acc_all_1CV_"+slurm_index_str+".txt", acc_all_1CV, delimiter=",")
    np.savetxt("acc_all_allCV_"+slurm_index_str+".txt", acc_all_allCV, delimiter=",")
    np.savetxt("acc_indpt_tune_"+slurm_index_str+".txt", acc_indpt_tune, delimiter=",")
    np.savetxt("acc_indpt_opt_"+slurm_index_str+".txt", acc_indpt_opt, delimiter=",")
    np.savetxt("acc_indpt_1CV_"+slurm_index_str+".txt", acc_indpt_1CV, delimiter=",")
    np.savetxt("acc_indpt_allCV_"+slurm_index_str+".txt", acc_indpt_allCV, delimiter=",")
    np.savetxt("evf_B1_tune_" + slurm_index_str + ".txt", evf_B1_tune, delimiter=",")
    np.savetxt("evf_B1_opt_" + slurm_index_str + ".txt", evf_B1_opt, delimiter=",")
    np.savetxt("evf_B1_1CV_" + slurm_index_str + ".txt", evf_B1_1CV, delimiter=",")
    np.savetxt("evf_B1_allCV_" + slurm_index_str + ".txt", evf_B1_allCV, delimiter=",")
    np.savetxt("evf_B2_tune_" + slurm_index_str + ".txt", evf_B2_tune, delimiter=",")
    np.savetxt("evf_B2_opt_" + slurm_index_str + ".txt", evf_B2_opt, delimiter=",")
    np.savetxt("evf_B2_1CV_" + slurm_index_str + ".txt", evf_B2_1CV, delimiter=",")
    np.savetxt("evf_B2_allCV_" + slurm_index_str + ".txt", evf_B2_allCV, delimiter=",")
    np.savetxt("evf_B3_tune_" + slurm_index_str + ".txt", evf_B3_tune, delimiter=",")
    np.savetxt("evf_B3_opt_" + slurm_index_str + ".txt", evf_B3_opt, delimiter=",")
    np.savetxt("evf_B3_1CV_" + slurm_index_str + ".txt", evf_B3_1CV, delimiter=",")
    np.savetxt("evf_B3_allCV_" + slurm_index_str + ".txt", evf_B3_allCV, delimiter=",")
    np.savetxt("ts_tune_"+slurm_index_str+".txt", ts_tune, delimiter=",")
    np.savetxt("ts_opt_"+slurm_index_str+".txt", ts_opt, delimiter=",")
    np.savetxt("ts_1CV_" + slurm_index_str + ".txt", ts_1CV, delimiter=",")
    np.savetxt("ts_allCV_" + slurm_index_str + ".txt", ts_allCV, delimiter=",")
    np.savetxt("time_1CV_"+slurm_index_str+".txt", time_1CV, delimiter=",")
    np.savetxt("time_allCV_"+slurm_index_str+".txt", time_allCV, delimiter=",")
    np.savetxt("time_tune_"+slurm_index_str+".txt", time_tune, delimiter=",")'''

    ############################################################################################################
    ## save files in local computer
    np.savetxt("cost_tune.txt", cost_tune, delimiter=",")
    np.savetxt("lam2_tune.txt", lam2_tune, delimiter=",")
    np.savetxt("lam3_tune.txt", lam3_tune, delimiter=",")
    np.savetxt("cost_opt.txt", cost_opt, delimiter=",")
    np.savetxt("lam2_opt.txt", lam2_opt, delimiter=",")
    np.savetxt("lam3_opt.txt", lam3_opt, delimiter=",")
    np.savetxt("acc_all_tune.txt", acc_all_tune, delimiter=",")
    np.savetxt("acc_all_opt.txt", acc_all_opt, delimiter=",")
    np.savetxt("acc_all_1CV.txt", acc_all_1CV, delimiter=",")
    np.savetxt("acc_all_allCV.txt", acc_all_allCV, delimiter=",")
    np.savetxt("acc_indpt_tune.txt", acc_indpt_tune, delimiter=",")
    np.savetxt("acc_indpt_opt.txt", acc_indpt_opt, delimiter=",")
    np.savetxt("acc_indpt_1CV.txt", acc_indpt_1CV, delimiter=",")
    np.savetxt("acc_indpt_allCV.txt", acc_indpt_allCV, delimiter=",")
    np.savetxt("evf_B1_tune.txt", evf_B1_tune, delimiter=",")
    np.savetxt("evf_B1_opt.txt", evf_B1_opt, delimiter=",")
    np.savetxt("evf_B1_1CV.txt", evf_B1_1CV, delimiter=",")
    np.savetxt("evf_B1_allCV.txt", evf_B1_allCV, delimiter=",")
    np.savetxt("evf_B2_tune.txt", evf_B2_tune, delimiter=",")
    np.savetxt("evf_B2_opt.txt", evf_B2_opt, delimiter=",")
    np.savetxt("evf_B2_1CV.txt", evf_B2_1CV, delimiter=",")
    np.savetxt("evf_B2_allCV.txt", evf_B2_allCV, delimiter=",")
    np.savetxt("evf_B3_tune.txt", evf_B3_tune, delimiter=",")
    np.savetxt("evf_B3_opt.txt", evf_B3_opt, delimiter=",")
    np.savetxt("evf_B3_1CV.txt", evf_B3_1CV, delimiter=",")
    np.savetxt("evf_B3_allCV.txt", evf_B3_allCV, delimiter=",")
    np.savetxt("ts_tune.txt", ts_tune, delimiter=",")
    np.savetxt("ts_opt.txt", ts_opt, delimiter=",")
    np.savetxt("ts_1CV.txt", ts_1CV, delimiter=",")
    np.savetxt("ts_allCV.txt", ts_allCV, delimiter=",")
    np.savetxt("time_1CV.txt", time_1CV, delimiter=",")
    np.savetxt("time_allCV.txt", time_allCV, delimiter=",")
    np.savetxt("time_tune.txt", time_tune, delimiter=",")










