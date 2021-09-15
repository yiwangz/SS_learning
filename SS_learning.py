import numpy as np
from sklearn import svm
from sklearn import linear_model
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from pygam import LinearGAM

########################################################################################################################
########################################################################################################################
## change parameters in different scenarios
class parScenario:
    def generatePar(self, ss):
        if ss == 0:
            n1 = 20
            n2 = 100
            n3 = 380
            coef1 = 3.0
            coef2 = 1.0
            coef3 = 0.1
            sd1 = 0.1
            sd2 = 0.5
            sd3 = 1.0
        elif ss == 1:
            n1 = 20
            n2 = 100
            n3 = 380
            coef1 = 3.0
            coef2 = 1.0
            coef3 = 0.5
            sd1 = 0.1
            sd2 = 0.5
            sd3 = 1.0
        elif ss == 2:
            n1 = 20
            n2 = 100
            n3 = 380
            coef1 = 3.0
            coef2 = 1.0
            coef3 = 1.0
            sd1 = 0.1
            sd2 = 0.5
            sd3 = 1.0
        elif ss == 3:
            n1 = 20
            n2 = 100
            n3 = 380
            coef1 = 3.0
            coef2 = 1.0
            coef3 = 3.0
            sd1 = 0.1
            sd2 = 0.5
            sd3 = 1.0
        elif ss == 4:
            n1 = 20
            n2 = 100
            n3 = 380
            coef1 = 3.0
            coef2 = 1.0
            coef3 = 0.5
            sd1 = 0.1
            sd2 = 0.5
            sd3 = 0.1
        elif ss == 5:
            n1 = 20
            n2 = 100
            n3 = 380
            coef1 = 3.0
            coef2 = 1.0
            coef3 = 0.5
            sd1 = 0.1
            sd2 = 0.5
            sd3 = 0.5
        elif ss == 6:
            n1 = 20
            n2 = 100
            n3 = 380
            coef1 = 3.0
            coef2 = 1.0
            coef3 = 0.5
            sd1 = 0.1
            sd2 = 0.5
            sd3 = 1.0
        elif ss == 7:
            n1 = 20
            n2 = 100
            n3 = 380
            coef1 = 3.0
            coef2 = 1.0
            coef3 = 0.5
            sd1 = 0.1
            sd2 = 0.5
            sd3 = 2.0
        elif ss == 8:
            n1 = 20
            n2 = 100
            n3 = 380
            coef1 = 3.0
            coef2 = 1.0
            coef3 = 0.5
            sd1 = 0.1
            sd2 = 0.5
            sd3 = 1.0
        elif ss == 9:
            n1 = 100
            n2 = 100
            n3 = 300
            coef1 = 3.0
            coef2 = 1.0
            coef3 = 0.5
            sd1 = 0.1
            sd2 = 0.5
            sd3 = 1.0
        elif ss == 10:
            n1 = 180
            n2 = 100
            n3 = 220
            coef1 = 3.0
            coef2 = 1.0
            coef3 = 0.5
            sd1 = 0.1
            sd2 = 0.5
            sd3 = 1.0
        elif ss == 11:
            n1 = 250
            n2 = 100
            n3 = 150
            coef1 = 3.0
            coef2 = 1.0
            coef3 = 0.5
            sd1 = 0.1
            sd2 = 0.5
            sd3 = 1.0

        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.coef1 = coef1
        self.coef2 = coef2
        self.coef3 = coef3
        self.sd1 = sd1
        self.sd2 = sd2
        self.sd3 = sd3


########################################################################################################################
########################################################################################################################
## simulate datasets S1 S2 (S3) and Sindpt
class dataGeneration:

    def simIndpt(self, nindpt, d, dist="uniform"):
        if d < 4:
            print('Too small dimension of features.')
        else:
            if dist == 'uniform':
                Xindpt = np.random.uniform(low=0, high=1, size=(nindpt, d))
            elif dist == 'normal':
                Xindpt = np.random.normal(loc=0, scale=1, size=(nindpt, d))
            else:
                print('Distribution not supported.')

            Tindpt = np.repeat(-1, repeats=nindpt)
            for i in range(nindpt):
                if 1+Xindpt[i,0]+Xindpt[i,1]-1.8*Xindpt[i,2]-2.2*Xindpt[i,3] > 0:
                    Tindpt[i] = 1

            self.nindpt = nindpt
            self.Xindpt = Xindpt
            self.Tindpt = Tindpt

    def sim2(self, n1, n2, d, coef1, sd1, dist='uniform', studySetting="trial"):
        if d < 4:
            print('Too small dimension of features.')
        else:
            ## features X1, X2, Xindpt
            if dist == 'uniform':
                X1 = np.random.uniform(low=0, high=1, size=(n1, d))
                X2 = np.random.uniform(low=0, high=1, size=(n2, d))
            elif dist == 'normal':
                X1 = np.random.normal(loc=0, scale=1, size=(n1, d))
                X2 = np.random.normal(loc=0, scale=1, size=(n2, d))
            else:
                print('Distribution not supported.')

            if studySetting == "trial":
                ## treatment assignments A1, A2
                A1 = np.repeat((1,-1), repeats=(n1/2,n1/2))
                A2 = np.repeat((1,-1), repeats=(n2/2,n2/2))
            elif studySetting == "observational":
                ## calculate the propensity score for each individual
                pmu1 = 0.4+0.5*X1[:,0]+0.5*X1[:,2]-X1[:,4]-X1[:,6]+0.2*X1[:,8]
                pmu2 = 0.4+0.5*X2[:,0]+0.5*X2[:,2]-X2[:,4]-X2[:,6]+0.2*X2[:,8]

                p1 = np.exp(pmu1)/(1+np.exp(pmu1))
                p2 = np.exp(pmu2)/(1+np.exp(pmu2))

                A1 = np.random.binomial(1,p1)
                A2 = np.random.binomial(1,p2)

                for i in range(n1):
                    if A1[i] == 0:
                        A1[i] = -1

                for i in range(n2):
                    if A2[i] == 0:
                        A2[i] = -1

            ## true labels T1, T2, Tindpt
            T1 = np.repeat(-1, repeats=n1)
            for i in range(n1):
                if 1+X1[i,0]+X1[i,1]-1.8*X1[i,2]-2.2*X1[i,3] > 0:
                    T1[i] = 1

            T2 = np.repeat(-1, repeats=n2)
            for i in range(n2):
                if 1+X2[i,0]+X2[i,1]-1.8*X2[i,2]-2.2*X2[i,3] > 0:
                    T2[i] = 1

            ## benefit values B1, B2
            mu1 = 0.01+0.02*X1[:,3]+coef1*A1*(1+X1[:,0]+X1[:,1]-1.8*X1[:,2]-2.2*X1[:,3])

            B1 = np.random.normal(loc=mu1, scale=sd1, size=n1)
            if min(B1) < 0:
                B1 = B1+abs(min(B1))+0.001

            self.X1 = X1
            self.A1 = A1
            self.T1 = T1
            self.mu1 = mu1
            self.B1 = B1
            self.X2 = X2
            self.A2 = A2
            self.T2 = T2
            self.n1 = n1
            self.n2 = n2

    def simB2(self, coef2, sd2):
        mu2 = 0.1+0.2*self.X2[:,3]+coef2*self.A2*(1+self.X2[:,0]+self.X2[:,1]-1.8*self.X2[:,2]-2.2*self.X2[:,3])

        B2 = np.random.normal(loc=mu2, scale=sd2, size=self.n2)
        if min(B2) < 0:
            B2 = B2+abs(min(B2))+0.001

        self.mu2 = mu2
        self.B2 = B2

    def sim3(self, n1, n2, n3, d, coef1, coef2, sd1, sd2, dist='uniform', studySetting="trial"):
        if d < 4:
            print('Too small dimension of features.')
        else:
            ## features X1, X2, Xindpt
            if dist == 'uniform':
                X1 = np.random.uniform(low=0, high=1, size=(n1, d))
                X2 = np.random.uniform(low=0, high=1, size=(n2, d))
                X3 = np.random.uniform(low=0, high=1, size=(n3, d))
            elif dist == 'normal':
                X1 = np.random.normal(loc=0, scale=1, size=(n1, d))
                X2 = np.random.normal(loc=0, scale=1, size=(n2, d))
                X3 = np.random.normal(loc=0, scale=1, size=(n3, d))
            else:
                print('Distribution not supported.')

            if studySetting == "trial":
                ## treatment assignments A1, A2
                A1 = np.repeat((1,-1), repeats=(n1/2,n1/2))
                A2 = np.repeat((1,-1), repeats=(n2/2,n2/2))
                A3 = np.repeat((1,-1), repeats=(n3/2,n3/2))
            elif studySetting == "observational":
                ## calculate the propensity score for each individual
                pmu1 = 0.4+0.5*X1[:,0]+0.5*X1[:,2]-X1[:,4]-X1[:,6]+0.2*X1[:,8]
                pmu2 = 0.4+0.5*X2[:,0]+0.5*X2[:,2]-X2[:,4]-X2[:,6]+0.2*X2[:,8]
                pmu3 = 0.4+0.5*X3[:,0]+0.5*X3[:,2]-X3[:,4]-X3[:,6]+0.2*X3[:,8]

                p1 = np.exp(pmu1)/(1+np.exp(pmu1))
                p2 = np.exp(pmu2)/(1+np.exp(pmu2))
                p3 = np.exp(pmu3)/(1+np.exp(pmu3))

                A1 = np.random.binomial(1,p1)
                A2 = np.random.binomial(1,p2)
                A3 = np.random.binomial(1,p3)

                for i in range(n1):
                    if A1[i] == 0:
                        A1[i] = -1

                for i in range(n2):
                    if A2[i] == 0:
                        A2[i] = -1

                for i in range(n3):
                    if A3[i] == 0:
                        A3[i] = -1

            ## true labels T1, T2, Tindpt
            T1 = np.repeat(-1, repeats=n1)
            for i in range(n1):
                if 1+X1[i,0]+X1[i,1]-1.8*X1[i,2]-2.2*X1[i,3] > 0:
                    T1[i] = 1

            T2 = np.repeat(-1, repeats=n2)
            for i in range(n2):
                if 1+X2[i,0]+X2[i,1]-1.8*X2[i,2]-2.2*X2[i,3] > 0:
                    T2[i] = 1

            T3 = np.repeat(-1, repeats=n3)
            for i in range(n3):
                if 1+X3[i,0]+X3[i,1]-1.8*X3[i,2]-2.2*X3[i,3] > 0:
                    T3[i] = 1

            ## benefit values B1, B2
            mu1 = 0.01+0.02*X1[:,3]+coef1*A1*(1+X1[:,0]+X1[:,1]-1.8*X1[:,2]-2.2*X1[:,3])

            B1 = np.random.normal(loc=mu1, scale=sd1, size=n1)
            if min(B1) < 0:
                B1 = B1+abs(min(B1))+0.001

            mu2 = 0.01+0.02*X2[:,3]+coef2*A2*(1+X2[:,0]+X2[:,1]-1.8*X2[:,2]-2.2*X2[:,3])

            B2 = np.random.normal(loc=mu2, scale=sd2, size=n2)
            if min(B2) < 0:
                B2 = B2+abs(min(B2))+0.001

            self.X1 = X1
            self.A1 = A1
            self.T1 = T1
            self.mu1 = mu1
            self.B1 = B1
            self.X2 = X2
            self.A2 = A2
            self.T2 = T2
            self.mu2 = mu2
            self.B2 = B2
            self.X3 = X3
            self.A3 = A3
            self.T3 = T3
            self.n1 = n1
            self.n2 = n2
            self.n3 = n3


    def simB3(self, coef3, sd3):
        mu3 = 0.1+0.2*self.X3[:,3]+coef3*self.A3*(1+self.X3[:,0]+self.X3[:,1]-1.8*self.X3[:,2]-2.2*self.X3[:,3])

        B3 = np.random.normal(loc=mu3, scale=sd3, size=self.n3)
        if min(B3) < 0:
            B3 = B3+abs(min(B3))+0.001

        self.mu3 = mu3
        self.B3 = B3


########################################################################################################################
########################################################################################################################
## calculate propensity score
class propensityScore:
    def __init__(self, Xall, Aall, m, dataLabel):
        self.Xall = Xall
        self.Aall = Aall
        self.m = m
        self.dataLabel = dataLabel

    def p(self, studySetting = 'trial'):
        if studySetting == 'trial':
            pall = np.full(self.Aall.shape[0], 0.5)
        elif studySetting == 'observational':
            pall = []

            for i in range(self.m):
                index = [item for sublist in np.where(self.dataLabel == i) for item in sublist]
                pvec = np.zeros(len(index))
                Xtemp = self.Xall[index,:]
                Atemp = self.Aall[index]
                logReg = LogisticRegression()
                logReg.fit(Xtemp, Atemp)

                index1 = [item for sublist in np.where(Atemp == 1) for item in sublist]
                index0 = [item for sublist in np.where(Atemp == -1) for item in sublist]

                pvec[index1] = logReg.predict_proba(Xtemp)[index1, 1]
                pvec[index0] = logReg.predict_proba(Xtemp)[index0, 0]

                pall = np.concatenate((pall, pvec))

        return pall


########################################################################################################################
########################################################################################################################
## Model selection (parameter tuning).
class tuneStat:
    def __init__(self, Xall, Aall, Ball, m, dataLabel, model):
        self.Xall = Xall
        self.Aall = Aall
        self.Ball = Ball
        self.m = m
        self.dataLabel = dataLabel
        self.model = model

    def tsFS(self, method='fs', CI=0.95):
        '''
        possible method:
        fs --> F statistic
        rbfs --> Robust F statistic (using MAD).
        trfs --> truncated F statistic (like 95% CI predicted benefit value).
        '''
        dv = self.model.decision_function(self.Xall)
        label = self.model.predict(self.Xall)

        value_neg = dv[label == -1]
        value_pos = dv[label == 1]

        if len(value_neg) == 0 or len(value_pos) == 0:
            return 0
        else:
            if method == 'fs':
                mult_factor = ((value_neg.shape[0] + value_pos.shape[0] - 2) * value_neg.shape[0] * value_pos.shape[0] / (
                            value_neg.shape[0] + value_pos.shape[0]))
                numerator = (np.mean(value_pos) - np.mean(value_neg)) ** 2
                denominator = (np.var(value_pos) * value_pos.shape[0] + np.var(value_neg) * value_neg.shape[0])
            elif method == 'rbfs':
                mult_factor = ((value_neg.shape[0] + value_pos.shape[0] - 2) * value_neg.shape[0] * value_pos.shape[0] / (
                            value_neg.shape[0] + value_pos.shape[0]))
                numerator = (np.median(value_pos) - np.median(value_neg)) ** 2
                denominator = ((mad(value_pos) ** 2) * value_pos.shape[0] + (mad(value_neg) ** 2) * value_neg.shape[0])
            elif method == 'trfs':
                tr_neg = value_neg[value_neg > np.quantile(value_neg, (1-CI)/2)]
                tr_pos = value_pos[value_pos < np.quantile(value_pos, 1-(1-CI)/2)]
                mult_factor = ((tr_neg.shape[0] + tr_pos.shape[0] - 2) * tr_neg.shape[0] * tr_pos.shape[0] / (
                            tr_neg.shape[0] + tr_pos.shape[0]))
                numerator = (np.mean(tr_pos) - np.mean(tr_neg)) ** 2
                denominator = (np.var(tr_pos) * tr_pos.shape[0] + np.var(tr_neg) * tr_neg.shape[0])
            else:
                print('Method not supported.')

            return mult_factor * numerator / denominator

    def tsCBV(self, CI=0.95, scale=False):

        y_combine = []
        A_combine = []
        B_combine = []

        for i in range(self.m):

            index = [item for sublist in np.where(self.dataLabel == i) for item in sublist]
            Xfit = self.Xall[index,:]
            Afit = self.Aall[index]
            Bfit = self.Ball[index]

            ## linear regression model for B
            Af = Afit*self.model.decision_function(Xfit)
            Xmat = np.column_stack((Xfit, Af))
            Xmat = sm.add_constant(Xmat)
            BModel = sm.OLS(Bfit, Xmat)
            res = BModel.fit()
            prediction = res.get_prediction(Xmat)
            result = np.array(prediction.summary_frame(alpha=(1-CI)))
            pred = result[:,0]
            ciLower = result[:,-2]
            ciUpper = result[:,-1]

            within = np.zeros(Afit.shape[0])
            for j in range(Afit.shape[0]):
                if Bfit[j] >= ciLower[j] and Bfit[j] <= ciUpper[j]:
                    within[j] = 1

            indexSelect = [item for sublist in np.where(within == 1) for item in sublist]

            ## after selecting the points that should be included into the calculation, normalize the benefits
            if scale == True:
                scaler = preprocessing.StandardScaler()
                pred = scaler.fit_transform(pred)
                #pred = (pred-min(pred))/(max(pred)-min(pred))

            y = self.model.predict(Xfit)
            y_combine = np.concatenate((y_combine, [y[elem] for elem in indexSelect]))
            A_combine = np.concatenate((A_combine, [Afit[elem] for elem in indexSelect]))
            B_combine = np.concatenate((B_combine, [pred[elem] for elem in indexSelect]))

        if sum((A_combine == y_combine)) == 0:
            return 0
        else:
            return sum(B_combine*(A_combine == y_combine))/sum((A_combine == y_combine))

    def tsSSE(self, model='linear'):

        sse = 0

        for i in range(self.m):

            index = [item for sublist in np.where(self.dataLabel == i) for item in sublist]
            Xfit = self.Xall[index,:]
            Afit = self.Aall[index]
            Bfit = self.Ball[index]

            Af = Afit * self.model.decision_function(Xfit)
            Xmat = np.column_stack((Xfit, Af))

            if model == 'linear':
                ## linear regression model for B
                Xmat = sm.add_constant(Xmat)
                BModel = sm.OLS(Bfit, Xmat)
                res = BModel.fit()
                pred = res.predict()
            elif model == 'GAM':
                BModel = LinearGAM(fit_intercept=True)
                res = BModel.fit(Xmat, Bfit)  ##the GAM model can be specified differently
                pred = res.predict(Xmat)

            sse = sse + sum([(Bfit[elem]-pred[elem])**2 for elem in range(len(Bfit))])

        return sse


########################################################################################################################
########################################################################################################################
## calculate the estimated value function separately
class EVF:
    def evfCal(self, Xall, Aall, Ball, m, dataLabel, model):

        labelall = model.predict(Xall)

        evfSeq = np.zeros(m)

        for i in range(m):
            index = [item for sublist in np.where(dataLabel == i) for item in sublist]
            Acal = Aall[index]
            Bcal = Ball[index]
            labelcal = labelall[index]

            evfSeq[i] = sum(Bcal*(Acal==labelcal))/sum((Acal==labelcal))

        self.evfSeq = evfSeq

########################################################################################################################
########################################################################################################################
## calculate the norm of RKHS for nonlinear kernel
class RKHSnorm:
    def __init__(self, sv, svIndex, lagMulti, gamma):
        self.sv = sv  ##support vectors
        self.svIndex = svIndex  ##index of support vectors
        self.lagMulti = lagMulti  ##Lagrange Multiplier
        self.gamma = gamma

    def omegaNorm(self):

        omega_penalty = np.zeros(len(self.svIndex)**2)

        for i in range(len(self.svIndex)):
            for j in range(len(self.svIndex)):
                omega_penalty[i*len(self.svIndex)+j] = self.lagMulti[0,i]*self.lagMulti[0,j]*np.exp(-self.gamma*np.sum(np.power(self.sv[i]-self.sv[j],2)))

        norm_result = 0.5*np.sum(omega_penalty)

        return norm_result

########################################################################################################################
########################################################################################################################
## Self training OWL
class SS_learning:
    def __init__(self, Xall, Aall, Ball, m, dataLabel, pall, weightingSetting, kernel):
        self.Xall = Xall
        self.Aall = Aall
        self.Ball = Ball
        self.m = m
        self.dataLabel = dataLabel
        self.pall = pall
        self.weightingSetting = weightingSetting
        self.kernel = kernel

    def iniFit(self, cost, gamma=1):

        index = []
        for i in range(self.m):
            index.append([item for sublist in np.where(self.dataLabel == i) for item in sublist])

        X1 = self.Xall[index[0],:]
        A1 = self.Aall[index[0]]
        B1 = self.Ball[index[0]]
        p1 = self.pall[index[0]]

        ## ordered remain observations (in case dataLabel is not ordered)
        Xremain = np.full([0, self.Xall.shape[1]], None)
        Aremain = []
        Bremain = []
        premain = []

        for j in range(1,self.m):
            Xremain = np.concatenate((Xremain, self.Xall[index[j],:]), axis=0)
            Aremain = np.concatenate((Aremain, self.Aall[index[j]]))
            Bremain = np.concatenate((Bremain, self.Ball[index[j]]))
            premain = np.concatenate((premain, self.pall[index[j]]))

        if self.weightingSetting == "IPW":
            sample_weight = B1/p1
        elif self.weightingSetting == "overlap":
            sample_weight = B1*(1-p1)

        if self.kernel == "linear":
            model_ini = svm.SVC(kernel='linear', C=cost, decision_function_shape="ovo")
        elif self.kernel == "nonlinear":
            model_ini = svm.SVC(kernel='rbf', C=cost, gamma=gamma, decision_function_shape="ovo")

        model_ini.fit(X1, A1, sample_weight=sample_weight)
        pred_ini = model_ini.predict(Xremain)

        self.pred_ini = pred_ini
        self.cost = cost
        self.gamma = gamma
        self.X1 = X1
        self.A1 = A1
        self.B1 = B1
        self.p1 = p1
        self.Xremain = Xremain
        self.Aremain = Aremain
        self.Bremain = Bremain
        self.premain = premain
        self.index = index

    ## set itertol=0, track=True to perform convergence analysis
    def fit(self, lam, itermax=50, itertol=1e-4, track=True, tuning='SSE', model='linear', method='fs', CI=0.95, scale=False):
        if not hasattr(self, 'pred_ini'):
            print('Run iniFit() for initial predicted labels!')
        else:
            iter = 0
            rel_obj = 1
            obj_old = 1

            if self.m == 2:
                lam = [lam]

            predRemainK = np.copy(self.pred_ini)

            if self.weightingSetting == "IPW":
                B_aux = np.concatenate((self.B1/self.p1, self.Bremain/self.premain))
            elif self.weightingSetting == "overlap":
                B_aux = np.concatenate((self.B1*(1-self.p1), self.Bremain*(1-self.premain)))
            X_aux = np.concatenate((self.X1, self.Xremain, self.Xremain), axis=0)
            wts_aux = np.repeat(1, self.A1.shape[0])

            for j in range(1,self.m):
                Btemp = self.Ball[self.index[j]]
                ptemp = self.pall[self.index[j]]
                if self.weightingSetting == "IPW":
                    B_aux = np.concatenate((B_aux, np.repeat(np.average(Btemp/ptemp, axis=0), len(self.index[j]))))
                elif self.weightingSetting == "overlap":
                    B_aux = np.concatenate((B_aux, np.repeat(np.average(Btemp*(1-ptemp), axis=0), len(self.index[j]))))
                wts_aux = np.concatenate((wts_aux, np.repeat(lam[j-1], len(self.index[j]))))

            for k in range(1,self.m):
                wts_aux = np.concatenate((wts_aux, np.repeat((1-lam[k-1]), len(self.index[k]))))

            wts_aux = wts_aux*B_aux

            if track:
                obj_path = np.full(itermax, np.nan)
                pred_path = np.full([itermax, (self.Aall.shape[0])], np.nan)
                model_path = []
                ts_path = np.full(itermax, np.nan)


            while (iter < itermax and rel_obj >= itertol):
                A_aux = np.concatenate((self.A1, self.Aremain, predRemainK))


                if self.kernel == "linear":
                    modelk = svm.SVC(kernel='linear', C=self.cost, decision_function_shape="ovo")
                    modelk.fit(X_aux, A_aux, sample_weight=wts_aux)

                    xi_vec = 1-A_aux*modelk.decision_function(X_aux)
                    xi_vec[xi_vec < 0] = 0

                    obj_new = np.sum(np.power(modelk.coef_,2))*(1/2)+np.sum(xi_vec*wts_aux)*self.cost

                elif self.kernel == "nonlinear":
                    modelk = svm.SVC(kernel='rbf', C=self.cost, gamma=self.gamma, decision_function_shape="ovo")
                    modelk.fit(X_aux, A_aux, sample_weight=wts_aux)

                    lagMulti = modelk.dual_coef_  ## Lagrange multiplier times label y. Summation = 0.
                    svIndex = modelk.support_
                    sv = modelk.support_vectors_
                    baseObj = RKHSnorm(sv, svIndex, lagMulti, self.gamma)
                    rkhsNorm = baseObj.omegaNorm()

                    xi_vec = 1-A_aux*modelk.decision_function(X_aux)
                    xi_vec[xi_vec < 0] = 0

                    obj_new = rkhsNorm+np.sum(xi_vec*wts_aux)*self.cost

                rel_obj = abs(obj_new-obj_old)/obj_old

                obj_old = np.copy(obj_new)
                predRemainK = modelk.predict(self.Xremain)

                tuning_stat = tuneStat(self.Xall, self.Aall, self.Ball, self.m, self.dataLabel, modelk)
                if tuning == 'SSE':
                    ts_stat = tuning_stat.tsSSE(model=model)
                elif tuning == 'CBV':
                    ts_stat = tuning_stat.tsCBV(CI=CI, scale=scale)
                elif tuning == 'FS':
                    ts_stat = tuning_stat.tsFS(method=method, CI=CI)

                predk = modelk.predict(self.Xall)

                if track:
                    obj_path[iter] = obj_new
                    pred_path[iter,:] = predk ## unordered
                    model_path.append(modelk)
                    ts_path[iter] = ts_stat

                iter += 1

            if iter >= itermax:
                conv = 99
            else:
                conv = iter

            self.conv = conv
            self.objConv = obj_new
            self.predConv = predk
            self.modelConv = modelk
            self.tsConv = ts_stat

            if track:
                self.objPath = obj_path
                self.predPath = pred_path
                self.modelPath = model_path
                self.tsPath = ts_path

    ## predictions on new datasets
    def predict(self, Xindpt, track):
        ## results based on final model
        if not track:
            if not hasattr(self, 'modelConv'):
                print('Please run fit() first.')
            else:
                return self.modelConv.predict(Xindpt)  ##nindpt*1

        ## results based on the whole path of models (multi-D predicted labels)
        else:
            if not hasattr(self, 'modelPath'):
                print('Please run fit() with track = True.')
            else:
                predOut = []
                for model in self.modelPath:
                    predOut.append(model.predict(Xindpt))
                return np.asarray(predOut)  ##nindpt*conv


########################################################################################################################
########################################################################################################################
## Evaluate prediction performance (accuracy).
class evalPred:

    def __init__(self, predLabel, trueLabel):
        self.pred = predLabel
        self.true = trueLabel

    def acc(self):
        # only one set of predicted labels (e.g., conv)
        if self.pred.ndim == 1:
            return np.sum(self.pred == self.true)/self.true.shape[0]

        # multiple rows of predicted labels (e.g. path)
        else:
            acc_vec = np.full(self.pred.shape[0], np.nan)
            for ii in np.arange(self.pred.shape[0]):
                acc_vec[ii] = (np.sum(self.pred[ii, :] == self.true)/ self.true.shape[0])
            return(acc_vec)


########################################################################################################################
########################################################################################################################
## Evaluate tuning process.
class evalTune:
    def maxTune(self, ts_array, acc_array, method='max'):

        dim = len(ts_array.shape)

        opt_idx = []
        tune_idx = []

        for i in range(dim):
            opt_idx.append([np.where(acc_array == np.amax(acc_array))[i][0]])

        if method == 'max':
            for j in range(dim):
                tune_idx.append([np.where(ts_array == np.amax(ts_array))[j][0]])
        elif method == 'min':
            for j in range(dim):
                tune_idx.append([np.where(ts_array == np.amin(ts_array))[j][0]])

        self.tune_idx = tune_idx
        self.opt_idx = opt_idx


