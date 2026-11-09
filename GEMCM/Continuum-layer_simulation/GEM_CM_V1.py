# -*- coding: utf-8 -*-

# Genome-scale and continuum platform hybrid for simulating a motile population of Pseudomonas aeruginosa
# as its biomass grows. Metabolic output of interest is the small molecule pyocyanin. Single gene knockouts
# also investigated.

# Information to update the dynamic system is reliant on a random forest regression, ensemble, trained on 
# flux balance outputs. Input parameters specify effective bacterial motility relative to nutrient diffusion,
# maximum metabolite uptake fluxes, and initial conditions of the system. PDE equation is solved using a 
# finite volume method in fipy module

file_load='path/to/GEMCM_model/Trained-ensembles/'
save_loc='path/to/GEMCM_model/GEM-CM_output/'
sim_set='parameter_wildtype_025agEndpt.csv'
nutr_= 'parameter_RFflux_GlucArgGlu.csv'

import multiprocessing as mp
import numpy as np
import pandas as pd
from itertools import repeat
import json
from fipy import *
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn import model_selection

def mp_simul_dictionary(df, col): # parameter read-in with child processes, col being the unique value assigned to processes
    g=df["Parameter"].to_numpy()
    sample={} # input of variable name producing simulation parameter value
    for j in range(len(g)):
        sample[g[j]]=df.iloc[j][col]
    return sample

def flux_monod(x,cN_max,km): 
    return -1.0*cN_max*(x/(km + x))
    
def flux_logistic(x, cN_max, km, bp):
    return -1.0*cN_max*(x / (km + x))*(1 - (bp / 1.2)) # 1.2 gDW/L estimated from microscope images, Croze et al 2011 describes 3.8 x 10e8 cells/mL; 1.2 is approx. 1.2 x 10e9

actions={
    "Michaelis Menten": flux_monod,
    "Log_growth": flux_logistic
}

#----- CHECK THAT 'action_select' and 'user_input' ARE AGREEING

def flux_calc_perform(user_input,x,y,z):
    action_select=actions.get(user_input)
    return action_select(x,y,z) # add / remove inputs depending on user input being performed

class EnsembleRavel:
    def __init__(self, filepath):
        self.RFmodel=joblib.load(filepath)
        self.feat_labels=list(self.RFmodel.feature_names_in_)

    def sim_settings(self, param_dict):
        self.sim_toutput_ = np.float64(param_dict['dt result (min)']) # minute
        self.sim_length_ = np.float64(param_dict['Well Radius (cm)']) # centimeter
        self.sim_rstep=0.002 # centimeters
        self.discr_ = int(self.sim_length_/self.sim_rstep)
        self.rad_cut_=0.00002
        self.GEM_cyc_ = int(self.sim_toutput_/5) # new metabolic rates calculated at 5 minute intervals
        self.ts_ = np.float64(param_dict['dt_step (s)']) # second
        self.t_fin = np.float64(param_dict['t_sim (s)'])
        self.Db_n1 = np.float64(param_dict['Diffu_N1 (cm2/s)'])
        self.Db_b = self.Db_n1 /np.float64(param_dict['Diffu_N1toD1'])
        
    def ic_sim(self, params_metab):
        cb_init=0.145; pcn_init=0.0
        ic_ = [cb_init]
        uptake_max=[]
        ks_set=[]
        for i in range(len(params_metab.columns)):
            strs=params_metab[params_metab.columns[i]]
            uptake_max.append(strs.iloc[0])
            ks_set.append(strs.iloc[0] / strs.iloc[1])
            ic_.append(strs.iloc[2])   
        ic_.append(pcn_init)
        return uptake_max, ks_set, ic_
        
    def estimator(self, Cn_set, max_rate, half_sat):
    # Cn_set: nutrient concentrations matching the features trained on 
    # Random forest ensemble predicts target in place of metabolic model FBA
    # Inner function 'action' converts concentration to fluxes
    # max_rate, half_sat matching length of features
        X_cN=[ [] for x in range(Cn_set.shape[0])]
        for j in range(Cn_set.shape[0]):
            for k in range(Cn_set.shape[1]):
                X_cN[j].extend([flux_calc_perform("Michaelis Menten",Cn_set[j][k], max_rate[j], half_sat[j])])
                
        for gx in range(len(X_cN)):
            for jx in range(len(X_cN[gx])):
                if Cn_set[gx][jx] < 0.0:
                    X_cN[gx][jx] = 0.0
                    
        cN_cons=pd.DataFrame(data=np.array(X_cN).T,columns=self.feat_labels)
        rate_predict=self.RFmodel.predict(cN_cons)
        u0_var=rate_predict[:,0] 
        u1_var=np.zeros(len(range(0,Cn_set.shape[1])))
        for y in range(len(u0_var)):
            if u0_var[y] < 0.05:
                u1_var[y] = 0.0
            else:
                u1_var[y] = u0_var[y]
        pyoc_prod=rate_predict[:,1]
        return u1_var, cN_cons, pyoc_prod

def GEMCMsimul_cond(df): # INPUT: .csv containing major condition values and .xlsx file of the lookup tables
    g=df["Parameter"].to_numpy() # have the first column be labeled "Parameter"
    dtag=df.iloc[0][-1][1:] # have the date at the end of the 1st row 'D(6-digits)'
    file_save="simul_conditions_"+dtag+".json"
    dc_list=[]
    for x in range(1,len(df.columns)-1):
        var1=df[df['Parameter']=='Diffu_N1toD1'].iloc[:,[x]].to_numpy()[0][0]
        var2=df[df['Parameter']=='Diffu_N1 (cm2/s)'].iloc[:,[x]].to_numpy()[0][0]
        sample={}
        for j in range(len(g)):
            sample[g[j]]=df.iloc[j][x]
        sample['Dbac (cm2/s)']=str(round(np.float64(var2)/np.float64(var1),8))
        dc_list.append(sample)
    with open(file_save,"w") as outfile:
        json.dump(dc_list,outfile,indent=True)
    print(f'Simulation parameters stored as {file_save}...')
    return dc_list, dtag

def GEM_CM_MPloop(df_conditions, sim_number, param_metab, file_job):
    proc_param=mp_simul_dictionary(df_conditions, sim_number)
    ens_calc=EnsembleRavel(file_job+proc_param['Trained_Model'])
    ens_calc.sim_settings(proc_param)
    vmax_set, ks_set, ic_set = ens_calc.ic_sim(param_metab)
    cb1_0, cn1_0, cn2_0, cn3_0, pcn_0 =  ic_set
    
    Dn1=ens_calc.Db_n1
    Db1=ens_calc.Db_b
    tfinal=ens_calc.t_fin
    timestep=ens_calc.ts_
    result_dt=ens_calc.sim_toutput_
    t_vis=int(tfinal/(60*result_dt))
    grwth_tspan=ens_calc.GEM_cyc_
    rstep=ens_calc.sim_rstep
    discr=ens_calc.discr_
    rad_cut=ens_calc.rad_cut_
    
    #--------- MODEL GRID SPACE-------#
    mesh=CylindricalGrid1D(dr=rstep,nr=discr+1,origin=(rad_cut,)) # Creates the grid over space
    rho=CellVariable(name="bacteria density", mesh=mesh,hasOld=True,value=0.) # units g dry weight/L
    nutr1=CellVariable(name="Nutr1",mesh=mesh,hasOld=True,value=cn1_0) # UNITS mM
    nutr2=CellVariable(name="Nutr2", mesh=mesh, hasOld=True, value=cn2_0)
    nutr3=CellVariable(name="Nutr3", mesh=mesh, hasOld=True, value=cn3_0)
    PyoC=CellVariable(name="PCN",mesh=mesh,hasOld=True,value= pcn_0) # UNITS mM
    u_biom=CellVariable(name="biomass grwth",mesh=mesh,hasOld=True,value=0.)
    k_cons1=CellVariable(name="Nutr1 cons.",mesh=mesh,hasOld=True,value=0.)
    k_cons2=CellVariable(name="Nutr2 cons.", mesh=mesh, hasOld=True, value=0.)
    k_cons3=CellVariable(name="Nutr3 cons.", mesh=mesh, hasOld=True, value=0.)
    k_prod=CellVariable(name="PCN prod.",mesh=mesh,hasOld=True, value=0.)
    
    rho.faceGrad.constrain([0.],mesh.facesLeft); rho.faceGrad.constrain([0.],mesh.facesRight)
    nutr1.faceGrad.constrain([0.],mesh.facesLeft); nutr1.faceGrad.constrain([0.],mesh.facesRight)
    nutr2.faceGrad.constrain([0.], mesh.facesLeft); nutr2.faceGrad.constrain([0.], mesh.facesRight)
    nutr3.faceGrad.constrain([0.], mesh.facesLeft); nutr3.faceGrad.constrain([0.], mesh.facesRight)
    PyoC.faceGrad.constrain([0.],mesh.facesLeft); PyoC.faceGrad.constrain([0.],mesh.facesRight)
    
    eqrho=TransientTerm(var=rho)==DiffusionTerm(coeff=Db1, var=rho)+ImplicitSourceTerm(u_biom,var=rho)
    eqnutr1=TransientTerm(var=nutr1)==DiffusionTerm(coeff=Dn1, var=nutr1)+ImplicitSourceTerm((-1)*k_cons1,var=rho) # k_cons output set as positive
    eqnutr2=TransientTerm(var=nutr2)==DiffusionTerm(coeff=Dn1, var=nutr2)+ImplicitSourceTerm((-1)*k_cons2,var=rho)
    eqnutr3=TransientTerm(var=nutr3)==DiffusionTerm(coeff=Dn1, var=nutr3)+ImplicitSourceTerm((-1)*k_cons3,var=rho)
    eqpyoc=TransientTerm(var=PyoC)==DiffusionTerm(coeff=Dn1,var=PyoC)+ImplicitSourceTerm(k_prod,var=rho)
    eqSys=eqrho & eqnutr1 & eqnutr2 & eqnutr3 & eqpyoc
    desres=1e-5; residual=1000
    stand_disct=timestep
    for iv in range(0,51,1): # 1 mm inoculation radius in a 20 um discretized space
        rho[iv]=cb1_0
        
    rho_output=np.zeros((t_vis+1,int(discr+1)))
    nutr1_output=np.zeros((t_vis+1,int(discr+1)))
    nutr2_output=np.zeros((t_vis+1,int(discr+1)))
    nutr3_output=np.zeros((t_vis+1,int(discr+1)))
    pcn_output=np.zeros((t_vis+1,int(discr+1)))
    mu_output=np.zeros((t_vis+1,int(discr+1)))
    prod_output=np.zeros((t_vis+1,int(discr+1)))
    
    rho_output[0,:]=rho[::1]
    nutr1_output[0,:]=nutr1[::1]
    nutr2_output[0,:]=nutr2[::1]
    nutr3_output[0,:]=nutr3[::1]
    pcn_output[0,:]=PyoC[::1]
    
    Cn_set=np.array([list(np.copy(nutr1)),list(np.copy(nutr2)),list(np.copy(nutr3))])

    u_var, kcN_var, kpycn_var=ens_calc.estimator(Cn_set, vmax_set, ks_set)
    for ix in range(len(u_var)): #Random forest regressed to mmol/gDW/hr
        u_biom[ix]=u_var[ix]/3600.
        k_cons1[ix]=abs(kcN_var[kcN_var.columns[0]].iloc[ix]/3600.)
        k_cons2[ix]=abs(kcN_var[kcN_var.columns[1]].iloc[ix]/3600.)
        k_cons3[ix]=abs(kcN_var[kcN_var.columns[2]].iloc[ix]/3600.)
        k_prod[ix]=kpycn_var[ix]/3600.
    
    mu_output[0,:] = u_biom[::1]
    prod_output[0,:]=k_prod[::1]
    
    timec=0
    for j in range(1,t_vis+1): # the cB and cG outputs are recorded for the desired visual timestep
        for iv in range(grwth_tspan):
            Cn_set=np.array([list(np.copy(nutr1)),list(np.copy(nutr2)),list(np.copy(nutr3))])
            u_var, kcN_var, kpycn_var=ens_calc.estimator(Cn_set, vmax_set, ks_set)
            for ix in range(len(u_var)):
                if u_var[ix] < 0.05:
                    u_biom[ix]=0.0
                    k_cons1[ix]=0.0
                    k_cons2[ix]=0.0
                    k_cons3[ix]=0.0
                    k_prod[ix]=0.0
                else:
                    u_biom[ix]=u_var[ix]/3600.
                    k_cons1[ix]=abs(kcN_var[kcN_var.columns[0]].iloc[ix]/3600.)
                    k_cons2[ix]=abs(kcN_var[kcN_var.columns[1]].iloc[ix]/3600.)
                    k_cons3[ix]=abs(kcN_var[kcN_var.columns[2]].iloc[ix]/3600.)
                    k_prod[ix]=kpycn_var[ix]/3600.
            
            for k in range(0,int(tfinal/t_vis/stand_disct/grwth_tspan)): #inner loop w/ k represents the desired visual timestep in seconds
                rho.updateOld()
                nutr1.updateOld()
                nutr2.updateOld()
                nutr3.updateOld()
                PyoC.updateOld()
                resid=residual
                while resid > desres: 
                    resid=eqSys.sweep(dt=stand_disct)
                
                nutr_set=[np.copy(nutr1),np.copy(nutr2),np.copy(nutr3)]
                check_list=[np.where(nx < 0)[0] for nx in nutr_set]
                if any(len(n) > 0 for n in check_list):
                    entry_test=[len(np.where(nx<0)[0]) > 0 for nx in nutr_set]
                    if entry_test[0]==True:
                        for s_1 in check_list[0]:
                            k_cons1[s_1]=0.0
                    elif entry_test[1]==True:
                        for s_2 in check_list[1]:
                            k_cons2[s_2]=0.0
                    elif entry_test[2]==True:
                        for s_3 in check_list[2]:
                            k_cons3[s_3]=0.0
                timec=timec+stand_disct
        rho_output[j,:]=rho[::1]
        nutr1_output[j,:]=nutr1[::1]
        nutr2_output[j,:]=nutr2[::1]
        nutr3_output[j,:]=nutr3[::1]
        pcn_output[j,:]=PyoC[::1]
        mu_output[j,:] = u_biom[::1]
        prod_output[j,:]=k_prod[::1]
    return rho_output, nutr1_output, nutr2_output, nutr3_output, pcn_output, mu_output, prod_output

#-----------MULTI PROCESS START---------#
df1=pd.read_csv(sim_set)
df2=pd.read_csv(nutr_)
sim_num=np.arange(1,len(df1.columns)-1)
if __name__ == '__main__':
    with mp.Pool(processes=len(sim_num)) as pool:
        results=pool.starmap(GEM_CM_MPloop,zip(repeat(df1), sim_num, repeat(df2[df2.columns[1:]]), repeat(file_load)))
        
    print("PARAMETER RUNS COMPLETED...")
    sim_param,date_tag=GEMCMsimul_cond(df1)
    sheet_label=[]
    res_label=['rho','cN1','cN2', 'cN3', 'pyoc', 'mu', 'prod']
    for i in range(len(sim_param)):
        sheet_label.append(date_tag+f'_Sim R{sim_num[i]}')
    tfinal = np.float64(sim_param[0]['t_sim (s)'])
    L=np.float64(sim_param[0]['Well Radius (cm)']) # well radius in centimeters
    discr=int(np.float64(sim_param[0]['r_steps (adjust for 20 um)'])) # number of discretization
    result_dt=np.float64(sim_param[0]['dt result (min)']) # desired time resolution (in minutes) of model output
    t_vis=int(tfinal/(60*result_dt)) # number of time outputs to report profiles at desired intervals
    #-------OUTPUT SAVING---------------------------#
    x_csv=np.linspace(0,L,int(discr+1))
    t_csv=np.linspace(0,tfinal,t_vis+1)/3600.
    dfs_set=[[] for x in range(len(results[0]))]

    for i in range(len(results)):
        for res in range(len(results[i])):# reforming res_tol needed first
            d=x_csv[::2]
            t_exc=np.zeros(len(t_csv)+1).reshape(-1,1)
            for qx in range(len(t_csv)+1):
                if qx==0:
                    t_exc[qx]=max(results[i][res][-1]) #first corner is the maximum at the endpoint
                else:
                    t_exc[qx]=t_csv[qx-1]
                    
            nmat=np.vstack((d,results[i][res][:,::2]))
            nmat2=np.hstack((t_exc,nmat))
            df_res=pd.DataFrame(nmat2)
            dfs_set[res].append(df_res)
            
    for fx in range(len(dfs_set)):
        with pd.ExcelWriter(save_loc + f'GEM_CM_{res_label[fx]}_{date_tag}.xlsx') as writer: #file labeling may need to change w/ different parameters
            for n in range(len(results)):
                dfs_set[fx][n].to_excel(writer,sheet_name=sheet_label[n])
        
