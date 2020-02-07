import numpy as np
import pandas as pd
from sklearn import svm
import SS_learning as st
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp
import os

############################################################################################################
m = 3  ##number of total datasets

n1_vec = [20,100,180,250]
n2 = 100
n3_vec = [380,300,220,150]
nindpt = 1000

d = 10  ##dimension of features
coef1 = 3.0
coef2 = 1.0
coef3 = 0.5
sd1 = 0.1
sd2 = 0.5
sd3 = 1.0

cost_vec = np.array([1/128,1/64,1/32,1/16,1/8,1/4,1/2,1.0])
tuned_paras = [{'C': cost_vec}]
lam2_vec = np.arange(0, 1.1, 0.1)
lam3_vec = np.arange(0, 1.1, 0.1)

K = 20

studySetting = "trial"

############################################################################################################
def simulation(seed_base):

    np.random.seed(1234+seed_base)

    ########################################################################################################
    ## generate the simulation indpt dataset (which should stay the same for different S1, S2)
    data = st.dataGeneration()
    data.simIndpt(nindpt, d, dist="uniform")
    Xindpt, Tindpt = data.Xindpt, data.Tindpt

    #########################################################################################################
    ## matrices to save the accuracy results ################################################################
    cost_tune = np.full(len(n1_vec), np.nan)
    lam2_tune = np.full(len(n1_vec), np.nan)
    lam3_tune = np.full(len(n1_vec), np.nan)
    cost_opt = np.full(len(n1_vec), np.nan)
    lam2_opt = np.full(len(n1_vec), np.nan)
    lam3_opt = np.full(len(n1_vec), np.nan)
    acc_all_tune = np.full(K*len(n1_vec), np.nan)
    acc_indpt_tune = np.full(K*len(n1_vec), np.nan)
    obj_tune = np.full(K*len(n1_vec), np.nan)
    ts_tune = np.full(K*len(n1_vec), np.nan)
    acc_all_opt = np.full(K*len(n1_vec), np.nan)
    acc_indpt_opt = np.full(K*len(n1_vec), np.nan)
    obj_opt = np.full(K*len(n1_vec), np.nan)
    ts_opt = np.full(K*len(n1_vec), np.nan)

    for index in range(len(n1_vec)):

        dataLabel = np.repeat(range(m), (n1_vec[index], n2, n3_vec[index]))  ##label of subjects from S1, S2, S3

        ########################################################################################################
        ## generate the simulation datasets ####################################################################
        data.sim3(n1_vec[index], n2, n3_vec[index], d, coef1, sd1, coef2, sd2, dist='uniform')
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
        pall = propenScore.p(studySetting=studySetting)

        ############################################################################################################
        ## proposed method
        ## store the results for each cost and lambda pair
        obj_path = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0], K], np.nan)  ##matrix
        ts_path = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0], K], np.nan)  ##matrix  ##this is not trustworthy
        acc_all_path = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0], K], np.nan)  ##array
        acc_indpt_path = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0], K], np.nan)  ##array

        out = st.STowlLinear(Xall, Aall, Ball, m, dataLabel, pall)

        for ii in np.arange(cost_vec.shape[0]):

            out.iniFit(cost_vec[ii])

            for jj in np.arange(lam2_vec.shape[0]):

                for kk in np.arange(lam3_vec.shape[0]):

                    lam = [lam2_vec[jj], lam3_vec[kk]]

                    out.fit(lam, itermax=K, itertol=0, track=True, tuning='SSE', model='linear')

                    obj_path[ii,jj,kk,:] = out.objPath
                    ts_path[ii,jj,kk,:] = out.tsPath
                    acc_all_path[ii,jj,kk,:] = st.evalPred(out.predPath, Tall).acc()
                    predIndpt = out.predict(Xindpt, track=True)
                    acc_indpt_path[ii,jj,kk,:] = st.evalPred(predIndpt, Tindpt).acc()

        opt_index = st.evalTune()
        opt_index.maxTune(ts_path[:,:,:,K-1], acc_all_path[:,:,:,K-1], method='min')
        cost_tune_idx = opt_index.tune_idx[0]
        lam2_tune_idx = opt_index.tune_idx[1]
        lam3_tune_idx = opt_index.tune_idx[2]
        cost_opt_idx = opt_index.opt_idx[0]
        lam2_opt_idx = opt_index.opt_idx[1]
        lam3_opt_idx = opt_index.opt_idx[2]

        cost_tune[index] = cost_vec[cost_tune_idx]
        lam2_tune[index] = lam2_vec[lam2_tune_idx]
        lam3_tune[index] = lam3_vec[lam3_tune_idx]
        cost_opt[index] = cost_vec[cost_opt_idx]
        lam2_opt[index] = lam2_vec[lam2_opt_idx]
        lam3_opt[index] = lam3_vec[lam3_opt_idx]

        for l in range(K):
            acc_all_tune[index*K+l] = acc_all_path[cost_tune_idx, lam2_tune_idx,lam3_tune_idx,l]
            acc_indpt_tune[index*K+l] = acc_indpt_path[cost_tune_idx, lam2_tune_idx,lam3_tune_idx,l]
            obj_tune[index*K+l] = obj_path[cost_tune_idx, lam2_tune_idx,lam3_tune_idx,l]
            ts_tune[index*K+l] = ts_path[cost_tune_idx, lam2_tune_idx,lam3_tune_idx,l]

            acc_all_opt[index*K+l] = acc_all_path[cost_opt_idx, lam2_opt_idx,lam3_opt_idx,l]
            acc_indpt_opt[index*K+l] = acc_indpt_path[cost_opt_idx, lam2_opt_idx,lam3_opt_idx,l]
            obj_opt[index*K+l] = obj_path[cost_opt_idx, lam2_opt_idx,lam3_opt_idx,l]
            ts_opt[index*K+l] = ts_path[cost_opt_idx, lam2_opt_idx,lam3_opt_idx,l]

    dresult = dict()
    dresult['cost_tune'] = cost_tune
    dresult['lam2_tune'] = lam2_tune
    dresult['lam3_tune'] = lam3_tune
    dresult['cost_opt'] = cost_opt
    dresult['lam2_opt'] = lam2_opt
    dresult['lam3_opt'] = lam3_opt
    dresult['acc_all_tune'] = acc_all_tune
    dresult['acc_indpt_tune'] = acc_indpt_tune
    dresult['obj_tune'] = obj_tune
    dresult['ts_tune'] = ts_tune
    dresult['acc_all_opt'] = acc_all_opt
    dresult['acc_indpt_opt'] = acc_indpt_opt
    dresult['obj_opt'] = obj_opt
    dresult['ts_opt'] = ts_opt
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
    results = pool.map(simulation, range(ncpus * (slurm_index - 1), ncpus * slurm_index))'''

    ############################################################################################################
    ## run script on local computer
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
    acc_indpt_tune = np.row_stack(results[i]['acc_indpt_tune'] for i in range(len_replicate))
    obj_tune = np.row_stack(results[i]['obj_tune'] for i in range(len_replicate))
    ts_tune = np.row_stack(results[i]['ts_tune'] for i in range(len_replicate))
    acc_all_opt = np.row_stack(results[i]['acc_all_opt'] for i in range(len_replicate))
    acc_indpt_opt = np.row_stack(results[i]['acc_indpt_opt'] for i in range(len_replicate))
    obj_opt = np.row_stack(results[i]['obj_opt'] for i in range(len_replicate))
    ts_opt = np.row_stack(results[i]['ts_opt'] for i in range(len_replicate))

    ############################################################################################################
    ## save files in cluster
    '''np.savetxt("cost_tune_"+slurm_index_str+".txt", cost_tune, delimiter=",")
    np.savetxt("lam2_tune_"+slurm_index_str+".txt", lam2_tune, delimiter=",")
    np.savetxt("lam3_tune_"+slurm_index_str+".txt", lam3_tune, delimiter=",")
    np.savetxt("cost_opt_"+slurm_index_str+".txt", cost_opt, delimiter=",")
    np.savetxt("lam2_opt_"+slurm_index_str+".txt", lam2_opt, delimiter=",")
    np.savetxt("lam3_opt_"+slurm_index_str+".txt", lam3_opt, delimiter=",")
    np.savetxt("acc_all_tune_"+slurm_index_str+".txt", acc_all_tune, delimiter=",")
    np.savetxt("acc_indpt_tune_"+slurm_index_str+".txt", acc_indpt_tune, delimiter=",")
    np.savetxt("obj_tune_"+slurm_index_str+".txt", obj_tune, delimiter=",")
    np.savetxt("ts_tune_"+slurm_index_str+".txt", ts_tune, delimiter=",")
    np.savetxt("acc_all_opt_"+slurm_index_str+".txt", acc_all_opt, delimiter=",")
    np.savetxt("acc_indpt_opt_"+slurm_index_str+".txt", acc_indpt_opt, delimiter=",")
    np.savetxt("obj_opt_"+slurm_index_str+".txt", obj_opt, delimiter=",")
    np.savetxt("ts_opt_"+slurm_index_str+".txt", ts_opt, delimiter=",")'''

    ############################################################################################################
    ## save files in local computer
    np.savetxt("cost_tune.txt", cost_tune, delimiter=",")
    np.savetxt("lam2_tune.txt", lam2_tune, delimiter=",")
    np.savetxt("lam3_tune.txt", lam3_tune, delimiter=",")
    np.savetxt("cost_opt.txt", cost_opt, delimiter=",")
    np.savetxt("lam2_opt.txt", lam2_opt, delimiter=",")
    np.savetxt("lam3_opt.txt", lam3_opt, delimiter=",")
    np.savetxt("acc_all_tune.txt", acc_all_tune, delimiter=",")
    np.savetxt("acc_indpt_tune.txt", acc_indpt_tune, delimiter=",")
    np.savetxt("obj_tune.txt", obj_tune, delimiter=",")
    np.savetxt("ts_tune.txt", ts_tune, delimiter=",")
    np.savetxt("acc_all_opt.txt", acc_all_opt, delimiter=",")
    np.savetxt("acc_indpt_opt.txt", acc_indpt_opt, delimiter=",")
    np.savetxt("obj_opt.txt", obj_opt, delimiter=",")
    np.savetxt("ts_opt.txt", ts_opt, delimiter=",")











