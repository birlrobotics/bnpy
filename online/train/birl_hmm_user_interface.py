from optparse import OptionParser
import training_config
from time import time

def get_trials_group_by_folder_name():
    import copy
    if (get_trials_group_by_folder_name.done):
        return copy.deepcopy(get_trials_group_by_folder_name.trials_group_by_folder_name)


    import load_csv_data
    trials_group_by_folder_name = load_csv_data.run(
        success_path            = training_config.success_path,
        interested_data_fields  = training_config.interested_data_fields,
        preprocessing_normalize = False,
        preprocessing_scaling   = False
    )

    get_trials_group_by_folder_name.done = True
    get_trials_group_by_folder_name.trials_group_by_folder_name = trials_group_by_folder_name
    return copy.deepcopy(get_trials_group_by_folder_name.trials_group_by_folder_name)
        
def build_parser():
    usage = "usage: %prog --train-model --train-threshold --online-anomaly-detection"
    parser = OptionParser(usage=usage)
    parser.add_option(
        "--train-model",
        action="store_true", 
        dest="train_model",
        default = False,
        help="True if you want to train HMM models.")

    parser.add_option(
        "--train-threshold",
        action="store_true", 
        dest="train_threshold",
        default = False,
        help="True if you want to train log likelihook curve threshold.")

    parser.add_option(
        "--online-anomaly-detection",
        action="store_true", 
        dest="online_anomaly_detection",
        default = False,
        help="True if you want to run online anomaly detection.")

    return parser

if __name__ == "__main__":

    start_time = time()
    get_trials_group_by_folder_name.done = False

    parser = build_parser()
    (options, args) = parser.parse_args()

    if options.online_anomaly_detection is False:
        if options.train_model is True:
            print "trainning HMM model ....."
            trials_group_by_folder_name = get_trials_group_by_folder_name()

            import hmm_model_training
            hmm_model_training.run(
                model_save_path             = training_config.model_save_path,
                n_state                     = training_config.hmm_hidden_state_amount,
                n_iteration                 = training_config.hmm_max_train_iteration,
                alloModel                   = training_config.alloModel,
                obsModel                    = training_config.obsModel,
                varMethod                   = training_config.varMethod,
                trials_group_by_folder_name = trials_group_by_folder_name,
                output_path                 = training_config.output_path)

        if options.train_threshold is True:
            print "train threshold ...."
            trials_group_by_folder_name = get_trials_group_by_folder_name()

            import log_likelihood_training
            log_likelihood_training.run(
                model_save_path             = training_config.model_save_path,
                figure_save_path            = training_config.figure_save_path,
                trials_group_by_folder_name = trials_group_by_folder_name,
            )

    else:
        print "online anomaly detection is about to run.."
        import hmm_online_anomaly_detection

        trials_group_by_folder_name = get_trials_group_by_folder_name()
        one_trial_data_group_by_state = trials_group_by_folder_name.itervalues().next()
        state_amount = len(one_trial_data_group_by_state)

        hmm_online_anomaly_detection.run(
            interested_data_fields = training_config.interested_data_fields,
            model_save_path = training_config.model_save_path,
            state_amount = state_amount)

    end_time = time()
    print "The cost time of the process is %f s" %(end_time-start_time)
            
