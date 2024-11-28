import streamlit as st
import pandas as pd
import numpy as np

from model import SQ_survival, SQ_plot
import pickle
import os

st.title("Survival Calculator: Major Adverse Cardiovascular Events after Discharge in Patients Underwent Percutaneous Coronary Intervention")
# st.markdown("need name")

feat_list = [
    'Age',
    'BMI',
    'TC',
    'HDL',
    'New_LDL',
    'TG',
    'CKMB',
    'TroponinI',
    'Creatinine',
    'MDRD',
    'HbA1c',
    'UA',
    'Hemoglobin',
    'CRP',
    'alb_lab',
    'Fluoro_time',
    'Contrast_volume',
    'LVEF',
    'NRI',
    'MRCI_K_dis',
    'MRCI_slop',
    'Sex',
    'Antiplatelets',
    'GP_2b3a',
    'Statin',
    'Beta_blocker',
    'ACEI',
    'ARB',
    'Nitrate',
    'CCB',
    'Previous_CHF',
    'HTN',
    'Previous_CVA',
    'PAD',
    'Previous_CRF',
    'Dialysis',
    'Dyslipidemia',
    'COPD',
    'FHx_Premature_CAD',
    'AF',
    'IABP',
    'RWMA',
    'LM_disease',
    'Adm_status_0.0',
    'Adm_status_1.0',
    'Adm_status_2.0',
    'Adm_status_3.0',
    'ECG_0',
    'ECG_1',
    'ECG_2',
    'ECG_3',
    'Vascular_access_2_0.0',
    'Vascular_access_2_1.0',
    'Vascular_access_2_2.0',
    'Cath_status_0.0',
    'Cath_status_1.0',
    'Cath_status_2.0',
    'Cath_status_3.0',
    'VD_0',
    'VD_1',
    'VD_2',
    'VD_3',
    'TFG_0.0',
    'TFG_1.0',
    'TFG_2.0',
    'TFG_3.0',
    'Insurance_Cate_1',
    'Insurance_Cate_2',
    'Insurance_Cate_3',
    'Smoking_info_0',
    'Smoking_info_1',
    'Smoking_info_2',
    'Smoking_info_3',
    'DM_Treat_0',
    'DM_Treat_1',
    'DM_Treat_2',
    'DM_Treat_3',
    'alcohol_classification_0.0',
    'alcohol_classification_1.0',
    'alcohol_classification_2.0',
    'RCA',
    'LM',
    'LAD',
    'LCx',
]

num_list = [
    'Age',
    'BMI',
    'TC',
    'HDL',
    'New_LDL',
    'TG',
    'CKMB',
    'TroponinI',
    'Creatinine',
    'MDRD',
    'HbA1c',
    'UA',
    'Hemoglobin',
    'CRP',
    'alb_lab',
    'Fluoro_time',
    'Contrast_volume',
    'LVEF',
    'NRI',
    'MRCI_K_dis',
    'MRCI_slop',
] 


## continous values -> normalize
## mean value check!
mapper_yes_no = {
    "No": 0, 
    "Yes": 1
}

mapper_sex = {
    "Male": 1, 
    "Female":0
}

mapper_adm = {
    "Outpatient department": [1,0,0,0],
    "Emergency department": [0,1,0,0],
    "Non-acute transfer": [0,0,1,0],
    "Acute transfer": [0,0,0,1]
}

mapper_ecg = {
    "Normal": [1,0,0,0],
    "ST depression": [0,1,0,0],
    "ST elevation": [0,0,1,0],
    "Nonspecific": [0,0,0,1]
}

mapper_vascular_access = {
    "Radial artery": [1,0,0],
    "Femoral artery": [0,1,0],
    "Other": [0,0,1]
}

mapper_vd = {  
    "No vessle disease": [1,0,0,0],
    "1 vessle disease": [0,1,0,0],
    "2 vessle disease": [0,0,1,0],
    "3 vessle disease": [0,0,0,1]
}

mapper_tfg = {  
    "Grade 0": [1,0,0,0],
    "Grade 1": [0,1,0,0],
    "Grade 2": [0,0,1,0],
    "Grade 3": [0,0,0,1]
}

mapper_insurance = {
    "NHI": [1,0,0],
    "Medical aid": [0,1,0],
    "Other": [0,0,1]
}

mapper_smoking = {  
    "Never": [1,0,0,0],
    "Ex-smoker": [0,1,0,0],
    "Current smoker": [0,0,1,0],
    "Unknown": [0,0,0,1]
}

mapper_dm = {  
    "None": [1,0,0,0],
    "Without medication": [0,1,0,0],
    "With OHA": [0,0,1,0],
    "With insulin": [0,0,0,1]
}

mapper_alcohol = {
    "None": [1,0,0],
    "Mild": [0,1,0],
    "Heavy": [0,0,1]
}



X_in                     = pd.DataFrame(np.zeros([1, len(feat_list)]), columns=feat_list)


st.subheader("Patient Characteristics")

col1 = st.columns(2)
with col1[0]:
    age                      = st.slider("Age", min_value=22, max_value=100, value=65, step=1)   
    bmi                      = st.slider("Body mass index (BMI)", min_value=14.0, max_value=43.0, value=25.0, step=0.1)
with col1[1]:
    sex                      = st.radio("Gender", list(mapper_sex), horizontal=True, index=1)
    
X_in['Age'] = age
X_in['BMI'] = bmi
X_in['Sex'] = mapper_sex[sex]


col2 = st.columns(2)
with col2[0]:
    TC                    = st.slider("Total cholesterol, mg/dL", min_value=50, max_value=400, value=170, step=10)   
    New_LDL               = st.slider("LDL, mg/dL", min_value=11, max_value=300, value=100, step=1)   
    CKMB                  = st.slider("CKMB, ng/ml", min_value=0, max_value=2000, value=50, step=1)   
    Creatinine            = st.slider("Creatinine, mg/dL", min_value=0.2, max_value=10.0, value=1.2, step=0.1)
    HbA1c                 = st.slider("HbA1c, %", min_value=5.0, max_value=15, value=6.5, step=0.5)
    Hemoglobin            = st.slider("Hemoglobin, g/dL", min_value=6.0, max_value=18.0, value=13.0, step=0.1)   
    alb_lab               = st.slider("Albumin, g/dL", min_value=2.0, max_value=5.0, value=3.8, step=0.1)
    Contrast_volume       = st.slider("Contrast volume, ml", min_value=0, max_value=1000, value=250, step=10)   
    NRI                   = st.slider("Nutritional risk index", min_value=35, max_value=100, value=50, step=5)
    MRCI_slop             = st.slider("MRCI changed (adm-dis)", min_value=-50, max_value=80, value=3, step=1)       
with col2[0]:
    HDL                   = st.slider("HDL, mg/dL", min_value=13, max_value=100, value=45, step=1)   
    TG                    = st.slider("Triglyceride, mg/dL", min_value=15, max_value=300, value=130, step=1)   
    TroponinI             = st.slider("Troponin I, ng/ml", min_value=0, max_value=1000, value=20, step=1)
    MDRD                  = st.slider(r"GFR, ml/min/1.73m$^2$", min_value=5, max_value=300, value=75, step=5)
    UA                    = st.slider("Uric acid, mg/dL", min_value=1.0, max_value=50.0, value=5.5, step=0.5)
    CRP                   = st.slider("CRP, mg/dL", min_value=0, max_value=30.0, value=1.5, step=0.5)   
    Fluoro_time           = st.slider("Fluoro duration, min", min_value=5, max_value=240, value=30, step=5)   
    LVEF                  = st.slider("LVEF, %", min_value=15, max_value=80, value=60, step=5)
    MRCI_K_dis            = st.slider("MRCI on discharge", min_value=0, max_value=110, value=24, step=1)   

X_in['TC']                = TC*(400-50)+50
X_in['HDL']               = HDL*(100-13)+13
X_in['New_LDL']           = New_LDL*(300-11)+11
X_in['TG']                = TG*(300-15)+15
X_in['CKMB']              = CKMB*(2000-0)+0
X_in['TroponinI']         = TroponinI*(1000-0)+0
X_in['Creatinine']        = Creatinine*(10.0-0.2)+0.2
X_in['MDRD']              = MDRD*(300-5)+5
X_in['HbA1c']             = HbA1c*(15-5)+5
X_in['UA']                = UA*(50.0-1.0)+1.0
X_in['Hemoglobin']        = Hemoglobin*(18.0-6.0)+6.0
X_in['CRP']               = CRP*(30-0)+0
X_in['alb_lab']           = alb_lab*(5.0-2.0)+2.0
X_in['Fluoro_time']       = Fluoro_time*(240-5)+5
X_in['Contrast_volume']   = Contrast_volume*(1000-0)+0
X_in['LVEF']              = LVEF*(80-15)+15
X_in['NRI']               = NRI*(100-35)+35
X_in['MRCI_K_dis']        = MRCI_K_dis*(110-0)+0
X_in['MRCI_slop']         = MRCI_slop*(80-(-50))+(-50)    
    

col3 = st.columns(3)
with col3[0]:
    Antiplatelets     = st.radio("Antiplatelets", list(mapper_yes_no), horizontal=True, index=0)
    Beta_blocker      = st.radio("Beta_blocker", list(mapper_yes_no), horizontal=True, index=0)
    Nitrate           = st.radio("Nitrate", list(mapper_yes_no), horizontal=True, index=0)
    HTN               = st.radio("Hypertension", list(mapper_yes_no), horizontal=True, index=0)
    Previous_CRF      = st.radio("Previous_CRF", list(mapper_yes_no), horizontal=True, index=0)
    COPD              = st.radio("COPD", list(mapper_yes_no), horizontal=True, index=0)
    IABP              = st.radio("IABP used", list(mapper_yes_no), horizontal=True, index=0)
with col3[1]:
    GP_2b3a           = st.radio("GpIIb/IIIa inhibitors", list(mapper_yes_no), horizontal=True, index=0)
    ACEI              = st.radio("ACEI", list(mapper_yes_no), horizontal=True, index=0)
    CCB               = st.radio("CCB", list(mapper_yes_no), horizontal=True, index=0)
    Previous_CVA      = st.radio("Previous_CVA", list(mapper_yes_no), horizontal=True, index=0)
    Dialysis          = st.radio("Dialysis", list(mapper_yes_no), horizontal=True, index=0)
    FHx_Premature_CAD = st.radio("FHx_Premature_CAD", list(mapper_yes_no), horizontal=True, index=0)
    RWMA              = st.radio("RWMA", list(mapper_yes_no), horizontal=True, index=0)
with col3[2]:
    Statin            = st.radio("Statin", list(mapper_yes_no), horizontal=True, index=0)
    ARB               = st.radio("ARB", list(mapper_yes_no), horizontal=True, index=0)
    Previous_CHF      = st.radio("Previous_CHF", list(mapper_yes_no), horizontal=True, index=0)
    PAD               = st.radio("Peripheral artery disease", list(mapper_yes_no), horizontal=True, index=0)   
    Dyslipidemia      = st.radio("Dyslipidemia", list(mapper_yes_no), horizontal=True, index=0)
    AF                = st.radio("Atrial fibrillation", list(mapper_yes_no), horizontal=True, index=0)       
    LM_disease        = st.radio("LM_disease", list(mapper_yes_no), horizontal=True, index=0)
    
X_in['Antiplatelets'] = mapper_yes_no[Antiplatelets]
X_in['GP_2b3a'] = mapper_yes_no[GP_2b3a]
X_in['Statin'] = mapper_yes_no[Statin]
X_in['Beta_blocker'] = mapper_yes_no[Beta_blocker]
X_in['ACEI'] = mapper_yes_no[ACEI]
X_in['ARB'] = mapper_yes_no[ARB]
X_in['Nitrate'] = mapper_yes_no[Nitrate]
X_in['CCB'] = mapper_yes_no[CCB]
X_in['Previous_CHF'] = mapper_yes_no[Previous_CHF]
X_in['HTN'] = mapper_yes_no[HTN]
X_in['Previous_CVA'] = mapper_yes_no[Previous_CVA]
X_in['PAD'] = mapper_yes_no[PAD]
X_in['Previous_CRF'] = mapper_yes_no[Previous_CRF]
X_in['Dialysis'] = mapper_yes_no[Dialysis]
X_in['Dyslipidemia'] = mapper_yes_no[Dyslipidemia]
X_in['COPD'] = mapper_yes_no[COPD]
X_in['FHx_Premature_CAD'] = mapper_yes_no[FHx_Premature_CAD]
X_in['AF'] = mapper_yes_no[AF]
X_in['IABP'] = mapper_yes_no[IABP]
X_in['RWMA'] = mapper_yes_no[RWMA]
X_in['LM_disease'] = mapper_yes_no[LM_disease]


col4 = st.columns(1)
with col4[0]:
    Adm_statu               = st.radio("Adm_status", list(mapper_adm), horizontal=True, index=0)
    ECG                     = st.radio("ECG", list(mapper_ecg), horizontal=True, index=0)
    Vascular_access         = st.radio("Access route", list(mapper_vascular_access), horizontal=True, index=0)
    Cath_status             = st.radio("Cath_status", list(mapper_cath_status), horizontal=True, index=0)
    VD                      = st.radio("Severity CAD", list(mapper_vd), horizontal=True, index=0)
    TFG                     = st.radio("TIMI grade flow", list(mapper_tfg), horizontal=True, index=0)
    Insurance_Cate          = st.radio("Insurance type", list(mapper_insurance), horizontal=True, index=0)
    Smoking_info            = st.radio("Smoking status", list(mapper_smoking), horizontal=True, index=0)
    DM_Treat                = st.radio("Diabetes mellitus", list(mapper_dm), horizontal=True, index=0)
    alcohol_classification  = st.radio("Alcohol consumption ", list(mapper_alcohol), horizontal=True, index=0)
    
X_in[['Adm_status_0.0', 'Adm_status_1.0', 'Adm_status_2.0', 'Adm_status_3.0']] = mapper_adm[Adm_status]
X_in[['ECG_0','ECG_1','ECG_2','ECG_3']] = mapper_ecg[ECG]
X_in[['Vascular_access_2_0.0', 'Vascular_access_2_1.0', 'Vascular_access_2_2.0']] = mapper_vascular_access[Vascular_access]
X_in[['Cath_status_0.0', 'Cath_status_1.0', 'Cath_status_2.0', 'Cath_status_3.0']] = mapper_cath_status[Cath_status]
X_in[['VD_0', 'VD_1', 'VD_2', 'VD_3']] = mapper_vd[VD]
X_in[['TFG_0.0', 'TFG_1.0', 'TFG_2.0', 'TFG_3.0']] = mapper_tfg[TFG]
X_in[['Insurance_Cate_1', 'Insurance_Cate_2', 'Insurance_Cate_3']] = mapper_insurance[Insurance_Cate]
X_in[['Smoking_info_0', 'Smoking_info_1', 'Smoking_info_2', 'Smoking_info_3']] = mapper_smoking[Smoking_info]
X_in[['DM_Treat_0', 'DM_Treat_1', 'DM_Treat_2', 'DM_Treat_3']] = mapper_dm[DM_Treat]
X_in[['alcohol_classification_0.0', 'alcohol_classification_1.0', 'alcohol_classification_2.0']] = mapper_alcohol[alcohol_classification]

    
col5 = st.columns(2)
with col5[0]:
    RCA               = st.radio("PCI-RCA territory", list(mapper_yes_no), horizontal=True, index=0)
    LAD               = st.radio("PCI-LAD territory", list(mapper_yes_no), horizontal=True, index=0)

with col5[1]:
    LM                = st.radio("PCI-LM", list(mapper_yes_no), horizontal=True, index=0)
    LCx               = st.radio("PCI-LCx territory", list(mapper_yes_no), horizontal=True, index=0)

X_in['RCA'] = mapper_yes_no[RCA]
X_in['LM'] = mapper_yes_no[LM]
X_in['LAD'] = mapper_yes_no[LAD]
X_in['LCx'] = mapper_yes_no[LCx]


st.subheader(" ")

BUTTON = st.button("Predict Risk")


def update_risk_score(x):
    survival_ = SQ_survival(x) #percentage

    # data = "Predicted Survival Probability: {:.2f}%".format(risk_)
    data = "Predicted Survival Probability: {}%".format(survival_)

    return data


if BUTTON:
    # c = st.container()
    fig = SQ_plot(X_in)
    # with c: 
    st.plotly_chart(fig, theme=None, use_container_width=True)
        # st.plotly_chart(fig, theme="streamlit", use_container_width=True)
