import os 
# hardcoded constants.
data_type_options = [
    'endpoint_pose',
    'wrench',
    'endpoint_pose_and_wrench'        
]

data_fields_store = {
    "endpoint_pose": [
        '.endpoint_state.pose.position.x',
        '.endpoint_state.pose.position.y',
        '.endpoint_state.pose.position.z',
        '.endpoint_state.pose.orientation.x',
        '.endpoint_state.pose.orientation.y',
        '.endpoint_state.pose.orientation.z',
        '.endpoint_state.pose.orientation.w'
    ],
    'wrench': [
         '.wrench_stamped.wrench.force.x',
         '.wrench_stamped.wrench.force.y',
         '.wrench_stamped.wrench.force.z',
         '.wrench_stamped.wrench.torque.x',
         '.wrench_stamped.wrench.torque.y',
         '.wrench_stamped.wrench.torque.z',
    ] 
}


# config provided by the user
config_by_user = {
    'data_type_chosen': data_type_options[2],
    'base_path': '/home/Homls/npBayesHMM/python/bnpy/online/REAL_BAXTER_PICK_N_PLACE_with_5_states_20170711',
    'norm_style': 'l2',
    'hmm_max_train_iteration': 1000,
    'hmm_hidden_state_amount': 25,
    'alloModel' : 'HDPHMM',     # DPMixtureModel
                                # HDPHMM
    'obsModel'  : 'DiagGauss',  # DiagGauss     : Diagonal-covariance Gaussian
                                # Gauss         : Full-covariance Gaussian
                                # ZeroMeanGauss : Zero-mean, full-covariance
                                # AutoRegGauss  : First-order auto-regressive Gaussian

    'varMethod' : 'memoVB'      # FullDataAlgSet = set(['EM', 'VB', 'GS', 'pVB'])
                                # OnlineDataAlgSet = set(['soVB', 'moVB', 'memoVB', 'pmoVB'])
    }

for config_key in config_by_user:
    print config_key, ":", config_by_user[config_key]


# auto config generation
data_type_split = config_by_user['data_type_chosen'].split("_and_")
interested_data_fields = []
for data_type in data_type_split:
    interested_data_fields += data_fields_store[data_type]
interested_data_fields.append('.tag')
print "interested_data_fields:", interested_data_fields
print "press any key to continue."
raw_input()
alloModel        = config_by_user['alloModel']
obsModel         = config_by_user['obsModel']
varMethod        = config_by_user['varMethod']
success_path     = os.path.join(config_by_user['base_path'], "success")
model_save_path  = os.path.join(config_by_user['base_path'], "model", config_by_user['data_type_chosen'],
                                alloModel + '+' + obsModel + '+' + varMethod)
figure_save_path = os.path.join(config_by_user['base_path'], "figure", config_by_user['data_type_chosen'],
                                alloModel + '+' + obsModel + '+' + varMethod)

exec '\n'.join("%s=%r"%i for i in config_by_user.items())



