@Hongmin Wu : This folder is used for the online anomaly detection in baxter-pick-n-place task
# for training the models and threshold
step-by-step tutorial
1. training_config.py: for setting the customize path for training data
2. birl_hmm_user_interface.py: all-in-one run file,



# for online real-robot anomaly detection
'0. python birl_hmm_user_interface.py --train-model --train-threshold
1. rosrun baxter_interface joint_trajectory_action_server.py
2. rosrun birl_sim_examples pick_n_place_srv_server.py
3. rosrun birl_sim_examples real_topic_multimodal.py
4. rosrun robotiq_force_torque_sensor rq_sensor
5. python birl_hmm_user_interface.py --online-anomaly-detection
6. rqt_plot
7. rosrun smach_viewer smach_viewer.py
8. rosrun birl_sim_examples real_pick_n_place_joint_trajectory_smach_recovery.py