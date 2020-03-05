#PBS -N game-of-life
#PBS -q pdlab
#PBS -j oe
#PBS -l nodes=2:ppn=8

module load mpi/mpich3-x86_64

cd $PBS_O_WORKDIR
echo "====test Run starts now ======= `date` "

mpiexec -ppn 1 ./game-of-life 80000 40000 0.7 3 0 8 &> $PBS_JOBNAME.log

echo "====test Run ends now ======= `date` "