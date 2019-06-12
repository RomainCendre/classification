# Check current job on specific node
qhost -h webern07 -j
qlogin -q 3d@webern07
# Wait for other job
qsub -hold_jid job_id my_job
