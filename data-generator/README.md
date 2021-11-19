This directory contains the information and code needed to generate the sets of
simulated noisy and reverberant binaural signals used in the paper.  These sets
can be generated as follows on a unix machine.  (The MATLAB part should work on
Windows but the content of download.sh would work only on Linux/Mac)

In download.sh edit the variable 'data_src' at the top of the file.  This point
to the directory where you want to download the source corpora used to generate
the binaural signals.

In generate_binaural_wav.m edit the variables: 
corpora_dir
output_dir

Then do:

1) cd to this directory in the command line and execute: ./download.sh
(This first step can take several hours depending on your internet connection, better to run it over night)

2) Got to this page:
https://www.myairbridge.com/en/#!/folder/iAvNdeGHtgE46bzB4o1KpRwJf0Lsphut
and download the archive called 'clarity_CEC1_data.scenes_train.target.v1_2.tgz'
Extract it and copy directory 'clarity_CEC1_data' to obtain the structure below.
The content of the corpora_dir that you set should look like:
####################
├── Noise
│   ├── ambient_sound_cafeteria
│   ├── ambient_sound_courtyard
│   └── musan
├── RIR
│   ├── AIR_1_4
│   └── HRIR_database_wav
└── Speech
    ├── LibriSpeech
    └── clarity_CEC1_data
####################

In Matlab, run the script 'generate_binaural_wav.m'.  Note that a parallel for
loop is used to loop through all files for the 4 considered sets.  It is
recommended to use a machine with a large amount of workers, but may take a
while to complete.
