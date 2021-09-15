import numpy as np
import pandas as pd
from sklearn import svm
import SS_learning as st
from sklearn.linear_model import LogisticRegression
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

K = 20

dist = "uniform"
studySetting = "observational"
weightingSetting = "IPW"
kernel = "linear"
tuneModel = "linear"

############################################################################################################
def simulation(seed_base):

    np.random.seed(1234+seed_base)

    #########################################################################################################
    ## matrices to save the accuracy results ################################################################
    cost_tune = np.zeros(scenario)
    #gamma_tune = np.zeros(scenario)
    lam2_tune = np.zeros(scenario)
    lam3_tune = np.zeros(scenario)
    cost_opt = np.zeros(scenario)
    #gamma_opt = np.zeros(scenario)
    lam2_opt = np.zeros(scenario)
    lam3_opt = np.zeros(scenario)
    acc_all_tune = np.zeros(K*scenario)
    obj_tune = np.zeros(K*scenario)
    ts_tune = np.zeros(K*scenario)
    acc_all_opt = np.zeros(K*scenario)
    obj_opt = np.zeros(K*scenario)
    ts_opt = np.zeros(K*scenario)

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

        ############################################################################################################
        ## SS-learning
        ## store the results for each cost and lambda pair
        obj_path = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0], K], np.nan)  ##array
        ts_path = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0], K], np.nan)  ##array  ##this is not trustworthy
        acc_all_path = np.full([cost_vec.shape[0], lam2_vec.shape[0], lam3_vec.shape[0], K], np.nan)  ##array

        out = st.SS_learning(Xall, Aall, Ball, m, dataLabel, pall, weightingSetting, kernel)

        for ii in np.arange(cost_vec.shape[0]):

            out.iniFit(cost_vec[ii])

            for kk in np.arange(lam2_vec.shape[0]):
                for ll in np.arange(lam3_vec.shape[0]):
                    lam = [lam2_vec[kk], lam3_vec[ll]]

                    out.fit(lam, itermax=K, itertol=0, track=True, tuning='SSE', model=tuneModel)

                    obj_path[ii, kk, ll, :] = out.objPath
                    ts_path[ii, kk, ll, :] = out.tsPath
                    acc_all_path[ii, kk, ll, :] = st.evalPred(out.predPath, Tall).acc()

        opt_index = st.evalTune()
        opt_index.maxTune(ts_path[:,:,:,K-1], acc_all_path[:,:,:,K-1], method='min')
        cost_tune_idx = opt_index.tune_idx[0]
        #gamma_tune_idx = opt_index.tune_idx[1]
        lam2_tune_idx = opt_index.tune_idx[1]
        lam3_tune_idx = opt_index.tune_idx[2]
        cost_opt_idx = opt_index.opt_idx[0]
        #gamma_opt_idx = opt_index.opt_idx[1]
        lam2_opt_idx = opt_index.opt_idx[1]
        lam3_opt_idx = opt_index.opt_idx[2]

        cost_tune[ss] = cost_vec[cost_tune_idx]
        #gamma_tune[ss] = gamma_vec[gamma_tune_idx]
        lam2_tune[ss] = lam2_vec[lam2_tune_idx]
        lam3_tune[ss] = lam3_vec[lam3_tune_idx]
        cost_opt[ss] = cost_vec[cost_opt_idx]
        #gamma_opt[ss] = gamma_vec[gamma_opt_idx]
        lam2_opt[ss] = lam2_vec[lam2_opt_idx]
        lam3_opt[ss] = lam3_vec[lam3_opt_idx]

        for l in range(K):
            acc_all_tune[ss*K+l] = acc_all_path[cost_tune_idx, lam2_tune_idx, lam3_tune_idx,l]
            obj_tune[ss*K+l] = obj_path[cost_tune_idx, lam2_tune_idx, lam3_tune_idx,l]
            ts_tune[ss*K+l] = ts_path[cost_tune_idx, lam2_tune_idx, lam3_tune_idx,l]

            acc_all_opt[ss*K+l] = acc_all_path[cost_opt_idx, lam2_opt_idx, lam3_opt_idx,l]
            obj_opt[ss*K+l] = obj_path[cost_opt_idx, lam2_opt_idx, lam3_opt_idx,l]
            ts_opt[ss*K+l] = ts_path[cost_opt_idx, lam2_opt_idx, lam3_opt_idx,l]

    dresult = dict()
    dresult['cost_tune'] = cost_tune
    #dresult['gamma_tune'] = gamma_tune
    dresult['lam2_tune'] = lam2_tune
    dresult['lam3_tune'] = lam3_tune
    dresult['cost_opt'] = cost_opt
    #dresult['gamma_opt'] = gamma_opt
    dresult['lam2_opt'] = lam2_opt
    dresult['lam3_opt'] = lam3_opt
    dresult['acc_all_tune'] = acc_all_tune
    dresult['obj_tune'] = obj_tune
    dresult['ts_tune'] = ts_tune
    dresult['acc_all_opt'] = acc_all_opt
    dresult['obj_opt'] = obj_opt
    dresult['ts_opt'] = ts_opt
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
    results = pool.map(simulation, range(ncpus * (slurm_index - 1), ncpus * slurm_index))

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
    acc_all_tune = np.row_stack(results[i]['acc_all_tune'] for i in range(len_replicate))
    obj_tune = np.row_stack(results[i]['obj_tune'] for i in range(len_replicate))
    ts_tune = np.row_stack(results[i]['ts_tune'] for i in range(len_replicate))
    acc_all_opt = np.row_stack(results[i]['acc_all_opt'] for i in range(len_replicate))
    obj_opt = np.row_stack(results[i]['obj_opt'] for i in range(len_replicate))
    ts_opt = np.row_stack(results[i]['ts_opt'] for i in range(len_replicate))

    ### save files ###############################################################
    np.savetxt("cost_tune_"+slurm_index_str+".txt", cost_tune, delimiter=",")
    #np.savetxt("gamma_tune_"+slurm_index_str+".txt", gamma_tune, delimiter=",")
    np.savetxt("lam2_tune_"+slurm_index_str+".txt", lam2_tune, delimiter=",")
    np.savetxt("lam3_tune_"+slurm_index_str+".txt", lam3_tune, delimiter=",")
    np.savetxt("cost_opt_"+slurm_index_str+".txt", cost_opt, delimiter=",")
    #np.savetxt("gamma_opt_"+slurm_index_str+".txt", gamma_opt, delimiter=",")
    np.savetxt("lam2_opt_"+slurm_index_str+".txt", lam2_opt, delimiter=",")
    np.savetxt("lam3_opt_"+slurm_index_str+".txt", lam3_opt, delimiter=",")
    np.savetxt("acc_all_tune_"+slurm_index_str+".txt", acc_all_tune, delimiter=",")
    np.savetxt("obj_tune_"+slurm_index_str+".txt", obj_tune, delimiter=",")
    np.savetxt("ts_tune_"+slurm_index_str+".txt", ts_tune, delimiter=",")
    np.savetxt("acc_all_opt_"+slurm_index_str+".txt", acc_all_opt, delimiter=",")
    np.savetxt("obj_opt_"+slurm_index_str+".txt", obj_opt, delimiter=",")
    np.savetxt("ts_opt_"+slurm_index_str+".txt", ts_opt, delimiter=",")

    '''
    np.savetxt("cost_tune.txt", cost_tune, delimiter=",")
    #np.savetxt("gamma_tune.txt", gamma_tune, delimiter=",")
    np.savetxt("lam2_tune.txt", lam2_tune, delimiter=",")
    np.savetxt("lam3_tune.txt", lam3_tune, delimiter=",")
    np.savetxt("cost_opt.txt", cost_opt, delimiter=",")
    #np.savetxt("gamma_opt.txt", gamma_opt, delimiter=",")
    np.savetxt("lam2_opt.txt", lam2_opt, delimiter=",")
    np.savetxt("lam3_opt.txt", lam3_opt, delimiter=",")
    np.savetxt("acc_all_tune.txt", acc_all_tune, delimiter=",")
    np.savetxt("obj_tune.txt", obj_tune, delimiter=",")
    np.savetxt("ts_tune.txt", ts_tune, delimiter=",")
    np.savetxt("acc_all_opt.txt", acc_all_opt, delimiter=",")
    np.savetxt("obj_opt.txt", obj_opt, delimiter=",")
    np.savetxt("ts_opt.txt", ts_opt, delimiter=",")
    '''

