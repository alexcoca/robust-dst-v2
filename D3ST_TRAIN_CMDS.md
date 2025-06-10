CONFIG_FILE=d3st_flant5_large_2_sanity_check.json sbatch train_d3st.sh
Submitted batch job 4600312

CONFIG_FILE=train_d3st_centroids_replication.json sbatch -J centroids_run_1 train_d3st.sh
Submitted batch job 9626799

CONFIG_FILE=train_d3st_centroids_replication_2.json sbatch -J centroids_run_2 -p ampere-long train_d3st.sh
Submitted batch job 9626147