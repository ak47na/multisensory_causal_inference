import submitit

function = submitit.helpers.CommandFunction(["which", "python"])
executor = submitit.AutoExecutor(folder="slurm/logs")
executor.update_parameters(timeout_min=1,partition='cpu')
job = executor.submit(function)

# The returned python path is the one used in slurm.
# It should be the same as when running out of slurm!
# This means that everything that is installed in your
# conda environment should work just as well in the cluster

print(job.result())

log_folder = 'slurm/logs/%j'
def add(a, b):
    return a + b


a = [1, 2, 3, 4]
b = [10, 20, 30, 40]
executor = submitit.AutoExecutor(folder=log_folder)
# the following line tells the scheduler to only run\
# at most 2 jobs at once. By default, this is several hundreds
executor.update_parameters(slurm_array_parallelism=2,partition='cpu')
jobs = executor.map_array(add, a, b)  # just a list of jobs
