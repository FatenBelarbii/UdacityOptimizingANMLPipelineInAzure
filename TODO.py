
from azureml.core.compute import ComputeTarget, AmlCompute

# TODO: Create compute cluster
# Use vm_size = "Standard_D2_V2" in your provisioning configuration.
# max_nodes should be no greater than 4.

### YOUR CODE HERE ###
from azureml.core.compute_target import ComputeTargetException

# Choose a name for your CPU cluster
cpu_cluster_name = "cpucluster"

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found existing cluster, use it.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2',
                                                           max_nodes=4)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

cpu_cluster.wait_for_completion(show_output=True)
print("Cluster details: ", cpu_cluster.get_status().serialize())





# Specify parameter sampler
ps = RandomParameterSampling({
       '--C': uniform(0.01,1),
        '--max_iter': choice(100, 125, 150, 175, 200, 225, 250, 275, 300)})
        
 #Specify a policy 
policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5)


#Create a SKLearn estimator for use with train.py
est = SKLearn(source_directory='./',
                compute_target=cpu_cluster,
                entry_script='train.py')

#Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig(
                                   hyperparameter_sampling = ps,
                                   primary_metric_name = 'accuracy',
                                   primary_metric_goal = PrimaryMetricGoal.MAXIMIZE,
                                   max_total_runs = 20,
                                   max_concurrent_runs = 4,
                                   policy = policy,
                                   estimator = est)


#or
#env = Environment.get(ws, name='MyEnvironment')
#config = ScriptRunConfig(source_directory='./',
 #                           script='train.py',
  #                          compute_target=cpu_cluster,
   #                         environment=env)
#script_run = experiment.submit(config)

#Submit the hyperdrive run and show run details with the widget.
hd_run = exp.submit(hyperdrive_config)
RunDetails(hd_run).show()

import joblib
from azureml.core.model import Model

#Get the best run and save the model from that run.
best_run_hd = hd_run.get_best_run_by_primary_metric()
model_hd = best_run_hd.register_model(model_name='hyperdrive_best_model', 
                                model_path='./outputs/model.pkl',
                                model_framework=Model.Framework.SCIKITLEARN, 
                                model_framework_version='0.19.1')
print("Model successfully saved.")

####AUTOML
# Create TabularDataset using TabularDatasetFactory
# Data is available at: 
# "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

### YOUR CODE HERE ###
ds = TabularDatasetFactory.from_delimited_files(path = 'https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv')

# Use the clean_data function to clean your data.
x, y = clean_data(ds)


automl_config = AutoMLConfig(
    experiment_timeout_minutes=30,
    task="classification",
    primary_metric="accuracy",
    training_data=train_data,
    label_column_name='y',
    n_cross_validations=5,
    compute_target=cpu_cluster)
    
remote_run = exp.submit(config=automl_config, show_output=True)

RunDetails(remote_run).show()

best_run_AutoML, fitted_model_AutoML = remote_run.get_output()
model_AutoML = best_run_AutoML.register_model(model_path='./outputs/', model_name='Output_automl.pkl')
print("Model saved successfully")

#HyperDrive
print("Scikit-learn based Logistic regression: ")
best_run_metrics_hd = best_run_hd.get_metrics()
print("Best Run Id: ", best_run_hd.id)
print("Accuracy: ", best_run_metrics_hd['accuracy'])

#AutoML
print("\nAutoML:")
best_run_metrics_AutoML = best_run_AutoML.get_metrics()
print("Best run Id: ",best_run_AutoML.id)
print("Accuracy: ", best_run_metrics_AutoML['accuracy'])
print("Other details: ")
#best_run_AutoML.get_tags()
print("Fitted model:",fitted_model_AutoML)
