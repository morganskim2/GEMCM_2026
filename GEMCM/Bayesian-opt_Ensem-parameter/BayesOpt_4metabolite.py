#!/usr/bin/env python
# coding: utf-8

import joblib
import pandas as pd
import numpy as np
from numpy.random import random
from scipy.stats import norm, qmc
from scipy.integrate import solve_ivp
from scipy.optimize import minimize, Bounds
from sklearn import metrics
from warnings import catch_warnings
from warnings import simplefilter
from sklearn.gaussian_process import GaussianProcessRegressor

file_path='path/to/GEMCM_model/Bayesian-opt_Ensem-parameters/'
rf_model='Gamma02_GlucArgGluMet_WT.joblib'

def surrogate(model, X):
    with catch_warnings():
        simplefilter("ignore")
        return model.predict(X, return_std=True)
    
def flux_calc_perform(user_input,x,y,z,cb_cap):
    def flux_monod(x,cN_max,km):
        if x==0:
            a_val=0.0
        else:
            a_val= -1.0*cN_max*(x/(km + x))
        return a_val
    
    def flux_logistic(x, cN_max, km, bp):
        return -1.0*cN_max*(x / (km + x))*(1 - (bp / 1.2)) # 1.2 gDW/L estimated from microscope images, Croze et al 2011 describes 3.8 x 10e8 cells/mL; 1.2 is approx. 1.2 x 10e9
    
    actions = {
        "Michaelis Menten": flux_monod,
        "Log_growth": flux_logistic}
    if user_input=="Michaelis Menten":
        action_select=actions.get(user_input)
        calc=action_select(x, y, z)
    elif user_input=="Log_growth":
        action_select=actions.get(user_input)
        calc=action_select(x, y, z, cb_cap)
    return(calc)

def ODE_RFMonod(t, y, rate_calc):
    # RATES CONVERTED TO 1/min FOR TIME ARRAY
    biomass = y[0]
    rate_conv=[rate_calc[x]/60. for x in range(len(rate_calc))]
    fluxes=np.array(rate_conv)
    fluxes *= biomass
    return fluxes

class EnsembleRavel:
    def __init__(self, filepath):
        self.RFmodel=joblib.load(filepath)
        self.feat_labels=list(self.RFmodel.feature_names_in_)
        self.cb_cap=1.2
        
    def init_sim(self, params):
        # CONDENSES INITIAL CONDITION DEFINITIONS FOR BLACK-BOX OBJECTIVE
        tset=np.linspace(0, 840, 169) # Record for 5 minute intervals
        if len(params) < 7:
            v1, k1, v2, k2, v3, k3 = params
            uptake_max = [v1, v2, v3]
            ks_set = [v1/k1, v2/k2, v3/k3]
            ic_= [0.003, 1.5, 1.0, 32.0, 0.0]
            
        elif 7 < len(params) < 9:
            v1, k1, v2, k2, v3, k3, v4, k4 = params
            uptake_max = [v1, v2, v3, v4]
            ks_set = [v1/k1, v2/k2, v3/k3, v4/k4]
            ic_= [0.003, 1.5, 1.0, 0.25, 32.0, 0.0]
            
        elif 9 < len(params) < 11:
            v1, k1, v2, k2, v3, k3, v4, k4, v5, k5 = params
            uptake_max = [v1, v2, v3, v4, v5]
            ks_set = [v1/k1, v2/k2, v3/k3, v4/k4, v5/k5]
            ic_= [0.003, 1.58, 1.59, 1, 1, 32.0, 0.0]
            
        self.v_max=uptake_max
        self.k_s=ks_set
        return tset, uptake_max, ks_set, ic_
    
    def estimator(self, Cn_set):
    # Cn_set: nutrient concentrations matching the features trained on 
    # Random forest ensemble predicts target in place of metabolic model FBA
    # Inner function 'action' converts concentration to fluxes
    # max_rate, half_sat matching length of features
        X_cN=[ [] for x in range(Cn_set.shape[0])]
        for j in range(Cn_set.shape[0]):
            for k in range(Cn_set.shape[1]):
                X_cN[j].extend([flux_calc_perform("Michaelis Menten",Cn_set[j][k], self.v_max[j], self.k_s[j],self.cb_cap)])
        for gx in range(len(X_cN)):
            for jx in range(len(X_cN[gx])):
                if Cn_set[gx][jx] < 0.0:
                    X_cN[gx][jx] = 0.0
        cN_cons=pd.DataFrame(data=np.array(X_cN).T,columns=self.feat_labels)
        rate_predict=self.RFmodel.predict(cN_cons)
        rate_final=np.copy(rate_predict)
        for y in range(rate_predict.shape[0]):
            if rate_predict[y,0] < 0.05:
                rate_final[y,0]=0.0
                rate_final[y,1]=0.0
        u1_var=rate_final[:,0]
        pyoc_prod=rate_final[:,1]
        return u1_var, cN_cons, pyoc_prod
    
    def slim_estimate(self,Cn_tocheck):
        # Method evoked to intervene model rates when a given metabolite is depleted
        # Value adjustment only applied to mesh spaces with depleted metabolite at points idx_alt
        df_0=pd.DataFrame(Cn_tocheck)
        nutr_return=[[] for j in range(len(Cn_tocheck))]
        mu_return=[]
        kcons_return=[]
        pyoc_return=[]
        idx_alt=np.where((df_0>0).all().to_numpy() == False)[0]
        bool_check=(df_0>=0).all().value_counts()
        bool_mask = df_0>=0
        if False in list(bool_check.keys()):
            check_status='Mixed_'
            arr_1=df_0[bool_mask].replace(np.nan,0.0).to_numpy()
            nutr_return=arr_1
            X_cN=[[] for j in range(arr_1.shape[0])]
            for ix in range(arr_1.shape[0]):
                for ik in range(arr_1.shape[1]):
                    X_cN[ix].extend([flux_calc_perform("Michaelis Menten",arr_1[ix][ik], self.v_max[ix], self.k_s[ix], self.cb_cap)])
            kcons_return=pd.DataFrame(data=np.array(X_cN).T,columns=self.feat_labels)
            rate_predict=self.RFmodel.predict(kcons_return)
            rate_final=np.copy(rate_predict)
            for y in range(rate_predict.shape[0]):
                if rate_predict[y,0] < 0.05:
                    rate_final[y,0]=0.0
                    rate_final[y,1]=0.0
            mu_return=rate_final[:,0]
            pyoc_return=rate_final[:,1]
        else:
            check_status='Positive_all'
        return check_status, idx_alt, mu_return, nutr_return, kcons_return, pyoc_return
    
def condition_check(c_tp, r_tp): 
    # MANUAL INTERVENTION IF ENSEMBLE TRAINING EVALULATES NON-ZERO
    # GROWTH RATE WITH 0 EXCHANGE FLUX
    rate_k=[]
    for i in range(len(r_tp.columns)):
        if np.isclose(c_tp[i+1],0) == True or c_tp[i+1] < 0.0:
            rate_k.append(0.0)
        else:
            rate_k.append(r_tp[r_tp.columns[i]].iloc[0])
    return rate_k

# BLACK-BOX CALCULATION USING THE RANDOM FOREST
def objective_calc(theta, ens_class, exper):
    # END CALCULATION IS THE SUM(RMSE / MEAN) OF PREDICTED VALUES
    # AGAINST AVAILABLE EXPERIMENT DATA
    t_iter, uptake_max, ks_set, ic_sim=ens_class.init_sim(theta)
    output_track=[]; cn_track=[]
    output_track.append(ic_sim)
    y_newstep = ic_sim
    for i in range(len(t_iter-1)):
        spec=y_newstep.copy()
        Cn_ts = []
        for j in range(1,len(spec)-1):
            if spec[j] > 0:
                Cn_ts.append(spec[j])
            else:
                Cn_ts.append(0.0)
        cn_track.append(Cn_ts)
        u_var, kcN_var, kpycn_var = ens_class.estimator(np.array([Cn_ts]).T)
        rate_val=condition_check(spec, kcN_var)
        rf_pred=[u_var[0]]
        rf_pred.extend(rate_val)
        rf_pred.append(kpycn_var[0])
        params_ode=([rf_pred])
        cn_dt=[spec[x] for x in range(len(spec))]
        for k in range(5):
            t_chunk=np.linspace(0,1,61)
            sol= solve_ivp(ODE_RFMonod, t_span=(t_chunk.min(), t_chunk.max()),
                           y0=cn_dt, t_eval=t_chunk, args=params_ode, rtol=1e-6, atol=1e-8, method='BDF')
            new_dt=[sol.y[ik][-1] for ik in range(len(spec))]
            cN_set2=[np.array([new_dt[x]]) for x in range(1,5)]
            check_status, idx_alt, mu_val, cN_val, kcons_val, pyoc_val = ens_class.slim_estimate(cN_set2)
            if check_status=='Mixed_':
                rf_alt=rf_pred.copy()
                for ik in idx_alt:
                    rf_alt[0]=mu_val[ik]
                    rf_alt[1]=kcons_val[kcons_val.columns[0]].iloc[ik]
                    rf_alt[2]=kcons_val[kcons_val.columns[1]].iloc[ik]
                    rf_alt[3]=kcons_val[kcons_val.columns[2]].iloc[ik]
                    rf_alt[4]=kcons_val[kcons_val.columns[3]].iloc[ik]
                    rf_alt[5]=pyoc_val[ik]
                    new_dt[1]=cN_val[0][ik]
                    new_dt[2]=cN_val[1][ik]
                    new_dt[3]=cN_val[2][ik]
                    new_dt[4]=cN_val[3][ik]
                params_ode=([rf_alt])
            cn_dt=new_dt
        y_newstep = cn_dt
        output_track.append(cn_dt)
    eval_loc = [11, 31, 52, 62, 89, 129] # discrete time indices for prediction scoring
    sim_predict=np.array(cn_track)[eval_loc]
    nrmse_cumul=0.0
    for j in range(exper.shape[1]):
        rmse_j = metrics.mean_squared_error(y_true=exper[:,j], y_pred=sim_predict[:,j])**0.5
        n_avg=np.mean(exper[:,j])
        nrmse_cumul += rmse_j / n_avg
    return -1*nrmse_cumul

# re-define for choice of n
def sampling_generate(lb_bounds, ub_bounds, n_count):
    lhc_sampler=qmc.LatinHypercube(d=len(lb_bounds))
    pool=lhc_sampler.random(n = n_count) # larger samplings for larger hypercubes?
    Xsamples=qmc.scale(pool, lb_bounds, ub_bounds)
    return Xsamples

def acquisition(X, Xsamples, model, f_max, kappa = 0.0):
    mu, std = surrogate(model, Xsamples)
    gamma_x=[]
    for i in range(len(mu)):
        gamma_x.append((mu[i] - f_max - kappa) / (std[i] + 1E-9))
    probs = []
    for x in range(len(gamma_x)):
        probs.append((mu[x] - f_max - kappa)*norm.cdf(gamma_x[x]) + std[x]*norm.pdf(gamma_x[x]))
    return np.asarray(probs)

def acquisition_localopt(GP_train, sample_pt, f_max, kappa = 0.0):
    GP_mu, GP_std = surrogate(GP_train, [sample_pt])
    gamma_x=[]; ei_score=[]
    for i in range(len(GP_mu)):
        gamma_x.append((GP_mu[i] - f_max) / (GP_std[i] + 1E-9))
    for x in range(len(GP_mu)):
        ei_score.append((GP_mu[x] - f_max - kappa)*norm.cdf(gamma_x[x]) + GP_std[x]*norm.pdf(gamma_x[x]))
    return -1*ei_score[0]

def opt_acquisition(X, y, model, lb_bounds, ub_bounds, set_compare):
    f_max = max(y)
    Xsamples = sampling_generate(lb_bounds, ub_bounds, 5000)
    scores = acquisition(X, Xsamples, model, f_max)
    disc_x = None #Xsamples[top_idx][0]
    disc_EI = -1000.0
    local_opt=Bounds(lb = lb_bounds, ub = ub_bounds)
    for g in range(int(len(scores)/10)):
        top_idx=np.argsort(scores)[-10*(g+1):-10*(g)]
        x0_loc = [Xsamples[j] for j in top_idx]
        for k in range(len(x0_loc)):
            res = minimize(lambda x: acquisition_localopt(model, x, f_max),
                           x0_loc[k][0].flatten(),
                           method='L-BFGS-B',
                           bounds=local_opt)
            if abs(res.fun) > disc_EI :
                if tuple(res.x) in set_compare:
                    continue
                else:
                    disc_x = res.x
                    disc_EI = abs(res.fun)
        if disc_x is not None:
            break
    if disc_x is None:
        print('Localized minima found again, selecting coarse point')
        disc_x = x0_loc[0][0]
    return np.array(disc_x).reshape(1,-1), scores

def markov_step(current_state, bayes_lb, bayes_ub):
    disp = [0.4*random(1)[0] - 0.2 for g in range(len(current_state))]
    free_state = [(1+disp[g])*current_state[g] for g in range(len(current_state))]
    lhc_lb=[]; lhc_ub=[]
    lhc_lbver=[]
    for i in range(len(current_state)):
        lhc_ub.append(min(bayes_ub[i], free_state[i]*1.2))
        lhc_lb.append(max(bayes_lb[i], free_state[i]*0.8))
    for x in range(len(current_state)):
        if lhc_lb[x]<lhc_ub[x]:
            lhc_lbver.append(lhc_lb[x])
        else:
            lhc_lbver.append(0.6*lhc_ub[x])
    return np.array(lhc_lbver), np.array(lhc_ub)

def bayes_looper(ens_calc, exper_metab, theta_lo, theta_hi):
    agent_stoppt=[]
    for rx in range(5):
        X_func = sampling_generate(theta_lo, theta_hi, 20)
        Y_func = np.asarray([objective_calc(X_func[g], ens_calc, exper_metab) for g in range(len(X_func))])
        Y_func = Y_func.reshape(len(Y_func),1)
                
        # Surrogate model
        model = GaussianProcessRegressor()
        model.fit(X_func, Y_func)
        tested_set=set()
        for j in range(len(X_func)):
            tested_set.add(tuple(X_func[j]))
            
        for i in range(50):
            lhc_lo, lhc_hi = markov_step(X_func[np.argmax(Y_func)], theta_lo, theta_hi)
            x_new, scores = opt_acquisition(X_func, Y_func, model, lhc_lo, lhc_hi, tested_set)
            actual = objective_calc(x_new[0], ens_calc, exper_metab)
            X_func = np.vstack((X_func, x_new))
            Y_func = np.vstack((Y_func, actual))
            tested_set.add(tuple(x_new[0]))
            # update the model
            model.fit(X_func, Y_func)
    #         print(f'Iteration complete. Best objective score {np.round(max(Y_func),4)}')
        opt_loc=np.argmax(Y_func)
        agent_stoppt.append(np.concatenate((X_func[opt_loc], Y_func[opt_loc])))
    return agent_stoppt

ens_calc=EnsembleRavel(file_path1+rf_model)

# Experiment data
df=pd.read_excel(file_path2+'Bayes optimization_McGill data.xlsx', sheet_name='TabS3_McGill_train')
exper_metab=df.loc[:,ens_calc.feat_labels].to_numpy()

#-----SEARCH BOUNDARY SETTINGS-------#
x1_bounds = np.array([1, 5])
x2_bounds = np.array([100, 4000])
x3_bounds = np.array([1, 5])
x4_bounds = np.array([100, 4000])
x5_bounds = np.array([0.5, 5])
x6_bounds = np.array([25, 4000])
x7_bounds = np.array([1, 10])
x8_bounds = np.array([200, 4000])
# theta_bounds = np.array([x1_bounds, x2_bounds, x3_bounds, x4_bounds, x7_bounds, x8_bounds])
theta_bounds = np.array([x1_bounds, x2_bounds, x3_bounds, x4_bounds, 
                         x5_bounds, x6_bounds, x7_bounds, x8_bounds])#, x2_bounds])
theta_lo = theta_bounds[:, 0]
theta_hi = theta_bounds[:, 1]

for swm in range(10):
    agent_5=bayes_looper(ens_calc,exper_metab,theta_lo, theta_hi)
    df_1=pd.DataFrame(data = agent_5)
    if swm ==0:
        df_1.to_csv('BayesOpt_WT_GlucArgGluMet.csv',index = False)
    else:
        df_1.to_csv('BayesOpt_WT_GlucArgGluMet.csv', mode='a', index=False, header=False)
    print(f'{5*(swm+1)} of 50 complete')