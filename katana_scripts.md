
https://unsw-restech.github.io/using_katana/running_jobs.html
# Job script

To run:
```bash
qsub job.pbs
qsub centrality_job.pbs
qsub find_disassortative_datasets_job.pbs
qsub one_heat_map_job.pbs
```

or 
```
qsub -l select=1:ncpus=24:mem=248gb,walltime=12:00:00 job_2.pbs
```

List my jobs:
```bash
qstat -u $USER
```


Delete a job:
```bash
qdel 2575766
```

```bash
#!/bin/bash

#PBS -l select=1:ncpus=8:mem=124gb
#PBS -l walltime=12:00:00
#PBS -M boqian.ma@student.unsw.edu.au
#PBS -m ae

cd /home/z5260890/Resilience_of_Multiplex_Networks_against_Attacks/Resilience_of_Multiplex_Networks_against_Attacks

echo "testing"
nohup python main.py aps o 0.1 5 &
nohup python main.py aps o 0.2 5 &
```

Interective shell
```bash
qsub -I -l select=1:ncpus=24:mem=248gb,walltime=12:00:00
```

