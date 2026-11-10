# import os
import numpy as np
import pandas as pd
from scipy.integrate import simpson

class BiomassDynamics:
    @staticmethod
    def radinteg_1D(r, c_i):
        return 2.0*np.pi*simpson(r*c_i,r)
    
    @staticmethod
    def avgarea_cyl1D(r1,r2):
        #annulus when r1 != 0
        return np.pi*(r2**2 - r1**2)

    @staticmethod
    def nan_remove(arr_1):
        arr_2=np.where(np.isnan(arr_1))
        filt_arr=np.delete(arr_1,arr_2)
        return filt_arr
    
    def __init__(self, xlsx_exper,time_offset):
     # xlsx_exper (list): filename strings of swimplate density profile data processed from microscope images
     #                    Sheet entries contain average and standard deviation gDW/L of each replicate (only averages will be considered for group analysis)
     # time_offset (float): Hour offset of first experiment observation if t = 0 is not the starting point  
        self.ws_exper=[]
        self.sheets_exper=[[] for x in range(len(xlsx_exper))]
        self.exper_rpos=[]
        self.exper_time=[]
        for j in range(len(xlsx_exper)):
            self.ws_exper.append(xlsx_exper[j])
            for k in pd.ExcelFile(xlsx_exper[j]).sheet_names:
                self.sheets_exper[j].append(k)
        df_ref=pd.read_excel(self.ws_exper[0],sheet_name=self.sheets_exper[0][0])
        self.exper_time= np.array([int(x+time_offset) for x in range(df_ref.shape[1]-1)])#df_ref.loc[:, list(df_ref.columns)[1]].to_numpy()
        self.exper_rpos=np.array([round(x,3) for x in df_ref["Position (cm)"].to_numpy()])
        self.time_displace=time_offset

    def radii_dt(self, gDW_cut=0.15):
        #gDW_cut: gDW/L threshold
        res_radii=[]
        for j in range(len(self.ws_exper)):
            group_radDT=pd.DataFrame(self.exper_time,columns=["Time (hours)"])
            for k in range(len(self.sheets_exper[j])):
                df_rep=pd.read_excel(self.ws_exper[j],sheet_name=self.sheets_exper[j][k])
                rep_radDT=[]
                for idx in range(len(self.exper_time)):
                    biom_t=self.nan_remove(df_rep[df_rep.columns[idx+1]].to_numpy())
        #contained loop
                    stop_point=-1
                    for idy in range(len(biom_t)):
                        if biom_t[idy] > gDW_cut:
                            stop_point+=1
                        else:
                            break
                    if stop_point==-1:
                        rep_radDT.append(0.0)
                    else:
                        rep_radDT.append(self.exper_rpos[stop_point])
                group_radDT[f"Rep. {k+1}"]=rep_radDT
            res_radii.append(group_radDT)
        return res_radii

    def bulkavg_dt(self):
        # integral of concentration profile(s) averaged by whole model space (self.exper_rpos)
        self.cb_bulkavg=[[] for x in range(len(self.ws_exper))]
        res_bulk=[]
        for j in range(len(self.ws_exper)):
            group_densDT=pd.DataFrame(self.exper_time,columns=["Time (hours)"])
            for k in range(len(self.sheets_exper[j])):
                df_rep=pd.read_excel(self.ws_exper[j],sheet_name=self.sheets_exper[j][k])
                rep_densDT=[]
                for idx in range(len(self.exper_time)):
                    biom_t=self.nan_remove(df_rep[df_rep.columns[idx+1]].to_numpy())
                    rep_densDT.append(self.radinteg_1D(self.exper_rpos[:len(biom_t)],biom_t)/self.avgarea_cyl1D(0,self.exper_rpos[len(biom_t)-1]))
                self.cb_bulkavg[j].append(rep_densDT[-1])
                group_densDT[f"Rep. {k+1}"]=rep_densDT
            n_val=[1.0]
            n_val.extend(self.cb_bulkavg[j])
            scaled_densDT=group_densDT.div(n_val)
            scaled_densDT=scaled_densDT.set_index("Time (hours)")
            res_bulk.append(scaled_densDT)
        return res_bulk
    
    def __repr__(self):
        print(f"{self.__class__.__name__}("
              f"Experiment biomass distribution taken at discrete points")

class ProfileMultiTime(BiomassDynamics):
    @ staticmethod
    def nan_adding(arr_to_change,arr_set):
        g1=list(np.copy(arr_to_change))
        for n in range(int(len(arr_set)-len(arr_to_change))):
            g1.append(np.nan)
        return np.array(g1)
    
    def __init__(self, xlsx_exper, time_offset,tpoints_j):
        super().__init__(xlsx_exper, time_offset)
        self.tpoints_idt=[int(tpoints_j[x]-self.time_displace) for x in range(len(tpoints_j))]

    def densityprofile_compile(self):
        densprofile=[]
        for j in range(len(self.ws_exper)):
            val_parse=[{} for x in range(len(self.tpoints_idt))]
            for k in range(len(self.sheets_exper[j])):
                df_ix=pd.read_excel(self.ws_exper[j],sheet_name=self.sheets_exper[j][k])
                df_eval=df_ix.loc[:,list(df_ix.columns[1:])]
                for n in range(len(self.tpoints_idt)):
                    arr1=self.nan_remove(df_eval.loc[:, df_eval.columns[self.tpoints_idt[n]]].to_numpy())
                    if np.isclose(self.cb_bulkavg[j][k],0)==False:
                        arr2=arr1/self.cb_bulkavg[j][k]
                    else:
                        arr2=arr1
                    arr_nafull=self.nan_adding(arr2,self.exper_rpos)
                    # val_parse.append({f'Rep {k+1}': pd.Series(arr_nafull)})
                    val_parse[n][f'Rep {k+1}']=pd.Series(arr_nafull)
            vals_df=[pd.DataFrame(x) for x in val_parse]
            densprofile.append(vals_df)
        res_normavg=[]
        res_std=[]
        for i in range(len(densprofile)):
            avgcb_tpoint=pd.DataFrame(data=self.exper_rpos, columns=["Position (cm)"])
            stdcb_tpoint=pd.DataFrame(data=self.exper_rpos, columns=["Position (cm)"])
            for t in range(len(self.tpoints_idt)):
                avgcb_tpoint[f"{self.exper_time[self.tpoints_idt[t]]} hour"]=densprofile[i][t].apply(np.nanmean, axis=1).to_numpy()
                stdcb_tpoint[f"{self.exper_time[self.tpoints_idt[t]]} hour"]=densprofile[i][t].apply(np.nanstd, axis=1).to_numpy()
            avgcb_tpoint=avgcb_tpoint.set_index("Position (cm)")
            stdcb_tpoint=stdcb_tpoint.set_index("Position (cm)")
            res_normavg.append(avgcb_tpoint)
            res_std.append(stdcb_tpoint)
        return densprofile, res_normavg, res_std
    
    def __repr__(self):
        print(f"{self.__class__.__name__}("
              f"Enter desired timepoints as their absolute experiment time")

class BiomassEndpoint():
    @staticmethod
    def radinteg_1D(r, c_i):
        return 2.0*np.pi*simpson(r*c_i,r)
    
    @staticmethod
    def avgarea_cyl1D(r1,r2):
        #annulus when r1 != 0
        return np.pi*(r2**2 - r1**2)

    @staticmethod
    def nan_remove(arr_1):
        arr_2=np.where(np.isnan(arr_1))
        filt_arr=np.delete(arr_1,arr_2)
        return filt_arr
    
    @ staticmethod
    def nan_adding(arr_to_change,arr_set):
        g1=list(np.copy(arr_to_change))
        for n in range(int(len(arr_set)-len(arr_to_change))):
            g1.append(np.nan)
        return np.array(g1)
    
    def __init__(self, xlsx_endpt):
        # xlsx_endpt (list): .xlsx string with file containing sheets of different group data
        # scaling by bulk biom unconventional with near- zero growth entries
        self.ws_exper=[]
        self.sheets_group=[[] for x in range(len(xlsx_endpt))]
        self.group_rpos=[]
        # self.cb_bulkavg=[[] for x in range(len(xlsx_endpt))]
        for j in range(len(xlsx_endpt)):
            self.ws_exper.append(xlsx_endpt[j])
            sheet_labels = list(pd.ExcelFile(xlsx_endpt[j]).sheet_names)
            for k in range(len(pd.ExcelFile(xlsx_endpt[j]).sheet_names)):
                self.sheets_group[j].append(sheet_labels[k])
                # self.cb_bulkavg[j].append([])
        df_ref=pd.read_excel(self.ws_exper[0],sheet_name=self.sheets_group[0][0])
        self.exper_rpos=np.array([round(x,3) for x in df_ref["Position (cm)"].to_numpy()])

    def densityprofile_(self):
        # Biomass profile normalized against bulk biomass density
        densprofile=[]
        for j in range(len(self.ws_exper)):
            val_parse=[{} for x in range(len(self.sheets_group[j]))]
            for k in range(len(self.sheets_group[j])):
                df_ix=pd.read_excel(self.ws_exper[j],sheet_name=self.sheets_group[j][k])
                df_eval=df_ix.loc[:,list(df_ix.columns[1:])]
                for n in range(df_eval.shape[1]):
                    # arr1=self.nan_remove(df_eval.loc[:, df_eval.columns[n]].to_numpy())
                    # biom_avg=self.radinteg_1D(self.exper_rpos[:len(arr1)],arr1)/self.avgarea_cyl1D(0,self.exper_rpos[len(arr1)-1])
                    # self.cb_bulkavg[j][k].append(biom_avg)
                    # if np.isclose(biom_avg,0)==False:
                    #     arr2=arr1/biom_avg
                    # else:
                    #     arr2=arr1
                    # arr_nafull=self.nan_adding(arr2,self.exper_rpos)
                    val_parse[k][f'Rep {n+1}']=pd.Series(df_eval[df_eval.columns[n]].to_numpy())
            vals_df=[pd.DataFrame(x) for x in val_parse]
            densprofile.append(vals_df)
        avgcb_endpt=pd.DataFrame(data=self.exper_rpos, columns=["Position (cm)"])
        stdcb_endpt=pd.DataFrame(data=self.exper_rpos, columns=["Position (cm)"])
        for i in range(len(densprofile)):
            for t in range(len(self.sheets_group[i])):
                avgcb_endpt[f"{self.sheets_group[i][t]}_20hour"]=densprofile[i][t].apply(np.nanmean, axis=1).to_numpy()
                stdcb_endpt[f"{self.sheets_group[i][t]}_20hour"]=densprofile[i][t].apply(np.nanstd, axis=1).to_numpy()
        avgcb_endpt=avgcb_endpt.set_index("Position (cm)")
        stdcb_endpt=stdcb_endpt.set_index("Position (cm)")

        return densprofile, avgcb_endpt, stdcb_endpt