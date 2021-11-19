#!/bin/bash

# Edit the following line, same as 'corpora_dir' in matlab file:
data_src=/path/to/downloaded/corpora


fun_dir=functions
mkdir -p ${data_src}/Speech
mkdir -p ${data_src}/RIR
mkdir -p ${data_src}/Noise
mkdir -p ${fun_dir}/dbstoi_mbstoi_nistoi
mkdir -p ${fun_dir}/voicebox

# Download external functions

wget -O ${fun_dir}/tmp.rar https://medi.uni-oldenburg.de/download/BSIM_2020/BSIM_2020.rar
rar x ${fun_dir}/tmp.rar ${fun_dir}
rm ${fun_dir}/tmp.rar

wget -O ${fun_dir}/dbstoi_mbstoi_nistoi/tmp.zip http://ah-andersen.net/wp-content/uploads/2018/12/dbstoi_mbstoi_nistoi.zip
unzip ${fun_dir}/dbstoi_mbstoi_nistoi/tmp.zip -d ${fun_dir}/dbstoi_mbstoi_nistoi
rm ${fun_dir}/dbstoi_mbstoi_nistoi/tmp.zip

wget -O ${fun_dir}/voicebox/tmp.zip http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/sap-voicebox.zip
unzip ${fun_dir}/voicebox/tmp.zip -d ${fun_dir}/voicebox
rm ${fun_dir}/voicebox/tmp.zip

# Download speech from LibriSpeech and noise from Musan

libri_url=https://www.openslr.org/resources/
for libri_file in 12/train-clean-360 12/dev-clean 12/test-clean 17/musan
do
	wget -O ${data_src}/tmp.tar.gz ${libri_url}${libri_file}.tar.gz
    tar -zxvf ${data_src}/tmp.tar.gz -C ${data_src}/Speech
    rm ${data_src}/tmp.tar.gz
done
mv ${data_src}/musan ${data_src}/Noise

# Download RIRs from Achen RIRs database and Oldenburg database:

for rir_url in https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/air_database_release_1_4.zip http://sirius.physik.uni-oldenburg.de/downloads/hrir/HRIR_database_wav_V1-1.zip
do
	wget --no-check-certificate -O ${data_src}/tmp.zip ${rir_url}
    unzip ${data_src}/tmp.zip -d ${data_src}/RIR
    rm ${data_src}/tmp.zip
done

for noise_url in http://sirius.physik.uni-oldenburg.de/downloads/hrir/ambient_sound_cafeteria_V1-1.zip http://sirius.physik.uni-oldenburg.de/downloads/hrir/ambient_sound_courtyard_V1-1.zip
do
	wget --no-check-certificate -O ${data_src}/Noise/tmp.zip ${noise_url}
    unzip ${data_src}/Noise/tmp.zip -d ${data_src}/Noise
    rm ${data_src}/Noise/tmp.zip
done

# Speech from the Clarity challenge database has to be obtained by downloading the file:
# clarity_CEC1_data.scenes_train.target.v1_2.tgz
# from:
# https://www.myairbridge.com/en/#!/folder/iAvNdeGHtgE46bzB4o1KpRwJf0Lsphut
# note that the file is nearly 50 GB (though our script just use part of it)
