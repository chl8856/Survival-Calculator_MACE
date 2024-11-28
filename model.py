import numpy as np
import pandas as pd
import plotly.graph_objs as go

import warnings
import joblib



warnings.filterwarnings('ignore')

def get_ensemble_prediction(models, W, X, time_horizons):
    # print('conducting ensemble prediction...')
    time_optimization = [30, 60, 90, 180, 270, 365]
    
    time_horizons_superset = np.unique(time_optimization + time_horizons).astype(int).tolist()

    pred = np.zeros([len(X), len(time_horizons_superset)])

    for m_idx in range(len(models)):
        # print('ing...' + models[m_idx].name)
        tmp_pred1 = models[m_idx].predict(X, time_horizons_superset)
        tmp_pred2 = models[m_idx].predict(X, time_optimization)

        for tt in range(len(time_optimization)):
            if tt == 0:
                tmp_time_idx1 = np.asarray(time_horizons_superset) <= time_optimization[tt]
                tmp_time_idx2 = np.asarray(time_horizons_superset) >  time_optimization[tt]

                prev_val      = np.zeros([len(X), 1])
                next_val      = tmp_pred2[:, [tt]]

                increment              = tmp_pred1[:, tmp_time_idx1] - prev_val
                pred[:, tmp_time_idx1] = pred[:, tmp_time_idx1] + W[tt,m_idx] * increment                
                pred[:, tmp_time_idx2] = pred[:, tmp_time_idx2] + W[tt,m_idx] * (next_val - prev_val)

            elif tt == len(time_optimization) - 1: #the last index  
                # tmp_time_idx1 = (np.asarray(time_horizons_superset) > self.time_optimization[tt-1]) & (np.asarray(time_horizons_superset) <= self.time_optimization[tt])
                tmp_time_idx1 = (np.asarray(time_horizons_superset) > time_optimization[tt-1])
                prev_val      = tmp_pred2[:, [tt-1]]

                increment              = tmp_pred1[:, tmp_time_idx1] - prev_val
                pred[:, tmp_time_idx1] = pred[:, tmp_time_idx1] + W[tt,m_idx] * increment                

            else:
                tmp_time_idx1 = (np.asarray(time_horizons_superset) > time_optimization[tt-1]) & (np.asarray(time_horizons_superset) <= time_optimization[tt])
                tmp_time_idx2 = np.asarray(time_horizons_superset) >  time_optimization[tt]
                prev_val      = tmp_pred2[:, [tt-1]]
                next_val      = tmp_pred2[:, [tt]]

                increment              = tmp_pred1[:, tmp_time_idx1] - prev_val
                pred[:, tmp_time_idx1] = pred[:, tmp_time_idx1] + W[tt,m_idx] * increment                
                pred[:, tmp_time_idx2] = pred[:, tmp_time_idx2] + W[tt,m_idx] * (next_val - prev_val)

    return pred[:, [f_idx for f_idx, f in enumerate(time_horizons_superset) if f in time_horizons]]



eval_times = np.arange(1,366).tolist()

surv_avg   = np.asarray([[0.99920442, 0.99882281, 0.99882281, 0.99801285, 0.99762202,
       0.99723514, 0.99684378, 0.99606775, 0.99565234, 0.99525677,
       0.99525677, 0.99525677, 0.99487729, 0.99447603, 0.99447603,
       0.99447603, 0.99447603, 0.99447603, 0.99407075, 0.99367498,
       0.99325792, 0.99325792, 0.99286347, 0.99247076, 0.99247076,
       0.99247076, 0.99247076, 0.99119663, 0.99119663, 0.99119663,
       0.9907922 , 0.9907922 , 0.9907922 , 0.9907922 , 0.99038756,
       0.98998275, 0.98998275, 0.98998275, 0.98998275, 0.98998275,
       0.98998275, 0.98998275, 0.98957718, 0.98836113, 0.98795214,
       0.98795214, 0.98754325, 0.98754325, 0.98672293, 0.98672293,
       0.98672293, 0.98672293, 0.985902  , 0.985902  , 0.98549048,
       0.98549048, 0.98549048, 0.98549048, 0.98549048, 0.98507908,
       0.98466745, 0.98384455, 0.98343274, 0.98260907, 0.98260907,
       0.98260907, 0.98260907, 0.98219523, 0.98219523, 0.98136798,
       0.9809541 , 0.9809541 , 0.9809541 , 0.9809541 , 0.98012569,
       0.98012569, 0.98012569, 0.98012569, 0.98012569, 0.97971145,
       0.97971145, 0.9792955 , 0.97887949, 0.97887949, 0.97846322,
       0.97804691, 0.97804691, 0.97721395, 0.97679739, 0.97679739,
       0.97679739, 0.97679739, 0.97679739, 0.97638028, 0.97596324,
       0.97596324, 0.97596324, 0.97596324, 0.97596324, 0.97596324,
       0.97554585, 0.97554585, 0.97554585, 0.97554585, 0.97512744,
       0.97512744, 0.97512744, 0.97470894, 0.97470894, 0.97429055,
       0.97429055, 0.97429055, 0.97387207, 0.97387207, 0.97387207,
       0.97387207, 0.97387207, 0.97387207, 0.97387207, 0.97387207,
       0.97345151, 0.97303004, 0.97303004, 0.97303004, 0.97260855,
       0.97218606, 0.97218606, 0.97218606, 0.97218606, 0.97133937,
       0.97133937, 0.97091598, 0.97091598, 0.97091598, 0.97091598,
       0.97091598, 0.97091598, 0.97091598, 0.97091598, 0.97049255,
       0.97049255, 0.97049255, 0.97049255, 0.97049255, 0.97049255,
       0.97049255, 0.97049255, 0.97049255, 0.97049255, 0.97049255,
       0.97049255, 0.96964237, 0.96964237, 0.9692166 , 0.9692166 ,
       0.9692166 , 0.9692166 , 0.96879005, 0.96879005, 0.96836359,
       0.96836359, 0.96836359, 0.96836359, 0.96836359, 0.96836359,
       0.96793555, 0.96793555, 0.96750684, 0.96750684, 0.96707786,
       0.96707786, 0.96622035, 0.96579113, 0.96579113, 0.96536201,
       0.96536201, 0.96536201, 0.96536201, 0.96536201, 0.96450368,
       0.96450368, 0.96450368, 0.96450368, 0.96450368, 0.96450368,
       0.96450368, 0.96450368, 0.96364326, 0.96322224, 0.96322224,
       0.96322224, 0.96322224, 0.96236208, 0.96236208, 0.96236208,
       0.96236208, 0.96236208, 0.96236208, 0.9619304 , 0.9619304 ,
       0.9619304 , 0.96150181, 0.96150181, 0.96150181, 0.96150181,
       0.96150181, 0.96150181, 0.9610736 , 0.9610736 , 0.9610736 ,
       0.9610736 , 0.9610736 , 0.9610736 , 0.96064212, 0.96064212,
       0.96064212, 0.96020591, 0.96020591, 0.96020591, 0.96020591,
       0.96020591, 0.95977599, 0.95977599, 0.95891339, 0.95891339,
       0.95891339, 0.95891339, 0.95891339, 0.95891339, 0.95891339,
       0.95891339, 0.95891339, 0.95891339, 0.95891339, 0.95891339,
       0.95848693, 0.95718753, 0.9567572 , 0.955895  , 0.955895  ,
       0.955895  , 0.95503546, 0.95460463, 0.95417129, 0.95417129,
       0.95373727, 0.95330187, 0.95286775, 0.95286775, 0.95286775,
       0.95286775, 0.95286775, 0.95286775, 0.95286775, 0.95286775,
       0.95199557, 0.95068672, 0.95068672, 0.94937483, 0.94894025,
       0.94894025, 0.94849832, 0.94761456, 0.94628343, 0.94452321,
       0.94320351, 0.94276339, 0.94232553, 0.94055912, 0.9401185 ,
       0.93836445, 0.93616764, 0.93396882, 0.93265403, 0.93132619,
       0.93088446, 0.92999227, 0.92821112, 0.92732199, 0.92510106,
       0.92376989, 0.92376989, 0.92243559, 0.92110904, 0.92022173,
       0.91889114, 0.91755914, 0.91533768, 0.91489012, 0.91310289,
       0.91310289, 0.91221308, 0.91176918, 0.91087741, 0.90776636,
       0.90688284, 0.90688284, 0.90643342, 0.90643342, 0.90554417,
       0.9046549 , 0.90421555, 0.90376951, 0.90287845, 0.90198672,
       0.90198672, 0.90109351, 0.90109351, 0.90109351, 0.90109351,
       0.90109351, 0.90109351, 0.89974232, 0.89885619, 0.89885619,
       0.89885619, 0.89885619, 0.89795219, 0.89704693, 0.89524645,
       0.89524645, 0.89524645, 0.89524645, 0.89479378, 0.89434219,
       0.89389273, 0.89298449, 0.89253579, 0.89208742, 0.89208742,
       0.89208742, 0.89208742, 0.89208742, 0.89073857, 0.89073857,
       0.89028572, 0.89028572, 0.88983528, 0.88848167, 0.88848167,
       0.88848167, 0.88848167, 0.88848167, 0.88802631, 0.88756859,
       0.88756859, 0.88756859, 0.88711117, 0.88711117, 0.88620149,
       0.88620149, 0.88620149, 0.88620149, 0.88620149, 0.88620149,
       0.88620149, 0.88620149, 0.88529044, 0.88529044, 0.88483664,
       0.88483664, 0.88483664, 0.88438559, 0.88393476, 0.88393476]])

def SQ_survival(x):

#     models_in = []
#     for m in range(7):
#         models_in += [joblib.load('./saved/model{}.joblib'.format(m))]

#     W_in      = np.array(
#         [[0.17677209, 0.11356408, 0.70966381],
#          [0.26585542, 0.        , 0.73414458],
#          [0.26585542, 0.        , 0.73414458],
#          [0.26585542, 0.        , 0.73414458],
#          [0.00467423, 0.03682702, 0.95849874],
#          [0.00467423, 0.03682702, 0.95849874]])
        
#     pred      = get_ensemble_prediction(models_in, W_in, x, eval_times)
        
    model = joblib.load('./saved/trained_SQ.gz')
    pred  = model.predict(x, time_horizons=eval_times)

    return 1. - pred


def SQ_plot(x):
    surv_s     = SQ_survival(x)

    tmp_df = pd.DataFrame(
        np.concatenate([
            np.asarray([eval_times]), 
            np.round(surv_avg, 4),
            np.round(surv_s, 4)
        ], axis=0).T,
        columns = ['Time', 'Avg', 'Yours']
    )

    fig = go.Figure([
        go.Scatter(
            name='Avg.',
            x=tmp_df['Time'],
            y=tmp_df['Avg'],
            mode='lines',
            line=dict(color='royalblue', width=2, dash='dash'),
        ),
        go.Scatter(
            name='Yours',
            x=tmp_df['Time'],
            y=tmp_df['Yours'],
            mode='lines',
            line=dict(color='crimson', width=2.5,),
        ), 
    ])

    fig.update_layout(
        xaxis_title='Time (day)',
        yaxis_title='Survival Probability',
        # title='Survival Probability',
        hovermode="x",
        yaxis=dict(
            range=[0., 1.]
        ),
        xaxis=dict(
            range=[1., 365.]
        ),
        font=dict(
            size=14
        ),
        height=400,
        margin=dict(
            l=50, #left margin
            r=0, #right margin
            b=50, #bottom margin
            t=30, #top margin)
        )
        # margin=dict(
        #     l=10, #left margin
        #     r=5, #right margin
        #     b=10, #bottom margin
        #     t=15, #top margin)
        # )
    )
    return fig
