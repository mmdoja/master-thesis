# Inferring instrument tune from visual data


### Requirements

```bash
pip install -r requirements.txt
pip install pretty_midi
pip install pyhocon
pip install pyFluidSynth
```

### Datasets

The link to the datasets are : 
- URMP: [here](http://www2.ece.rochester.edu/projects/air/projects/URMP.html)
- MUSIC: [here](https://github.com/roudimit/MUSIC_dataset)
- Piano: [At Your Fingertips: Automatic Piano Fingering Detection](https://openreview.net/forum?id=H1MOqeHYvB). The dataset from [here](https://drive.google.com/file/d/1kDPZSA7ppOaup9Q1Dab7bW4OXNh9mAQA/view)

To extract poses from videos: 
Use OpenPose API from [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

To extract MIDI from the audio
Use Lunarverus from [here](https://www.lunaverus.com/)

Once you have poses and midi from the videos and audio,
make a 'data' folder such that for each dataset there is a folder for midi and pose for each instrument
```
-config
-data
  -URMP
    -midi
      -cello
      -violin
      -...
    -pose
      -cello
      -violin
      -...
  -MUSIC
    -midi
      -...
    -pose
      -...
-core
```

### Training

For URMP
```bash
bash train_test/URMP_train.sh
```

For Piano
```bash
bash train_test/Piano_train.sh
```

For MUSIC
```bash
bash train_test/MUSIC_train.sh
```

You will notice a new directory called 'exps' is generated having checkpoints. 

### Testing

Install fluidsynth
```
apt install fluidsynth
pip install pyfluidsynth
pip uninstall fluidsynth
pip show pyfluidsynth
```
For URMP
```bash
bash train_test/URMP_train.sh
```

For MUSIC
```bash
bash train_test/MUSIC_train.sh
```

For Piano
```bash
bash train_test/Piano_train.sh
```

For NDB evaluation, use [this](https://github.com/eitanrich/gans-n-gmms/tree/master)

To submit a job in a slurm system, here is an example below.:
```
#!/bin/bash
#SBATCH --account=userdef
#SBATCH --job-name=genmusic
#SBATCH --output=%J.out
#SBATCH --error=%J.error
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --partition=A100short
#SBATCH --cpus-per-task=12
#SBATCH --time=8:00:00

module load Anaconda3/2022.05
module load CUDA/11.7.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/s6modoja/.conda/envs/thesis/lib
eval "$(conda shell.bash hook)"
conda activate thesis

cd /home/s6modoja/thesis/master-thesis
srun bash train_test/URMP_train.sh
```
Create the above code in a ```slurmjob.sh``` file and run using ```sbatch slurmjob,sh```


