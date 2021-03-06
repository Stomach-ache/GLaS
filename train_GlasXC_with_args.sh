#!/bin/bash

#dataset_name=RCV1-x
#all_dataset_names=( "Bibtex" "Delicious" "Mediamill" "Eurlex" "RCV1-x")
#val=0
#for name in ${all_dataset_names[*]};
#   do
#        if [ "$name" == "$dataset_name" ] ; then
#            val=1
#        fi
#    done

#if [ "$val" == 0 ] ; then
#    echo "Invalid dataset name specified"
#    exit 1
#fi

setup_path="./setups/Eurlex"
data_root="./data/Eurlex"
#setup_path="./setups/AmazonCat"
#data_root="./data/AmazonCat-13K.bow"
dataset_info="dataset_info.yml"
inp_enc_cfg="input_encoder_cfg.yml"
inp_dec_cfg="input_decoder_cfg.yml"
otp_enc_cfg="output_encoder_cfg.yml"
otp_dec_cfg="output_decoder_cfg.yml"
reg_cfg="regressor_cfg.yml"
opt_cfg="opt.yml"


# Please specify the options below
# device -> PyTorch device string
# epochs -> Number of epochs
#device='cpu' # 'cuda' for GPU, 'cpu' for CPU
device='cuda' # 'cuda' for GPU, 'cpu' for CPU
epochs=500

# This are static, feel free to change them as required
#batch_size=4096 # 4096 for AmazonCat-13K as given in paper(Glas) #TAKE BATCH_SIZE=32 FOR ERROR FREE EXECUTION
batch_size=64 # 4096 for AmazonCat-13K as given in paper(Glas) #TAKE BATCH_SIZE=32 FOR ERROR FREE EXECUTION
inp_ae_wgt=0
otp_ae_wgt=0
seed=1729
#all_k=( 1 3 5 )
#all_k = ( 1 )
interval=20
init_scheme="xavier_uniform"

#for k in ${all_k[*]};
for k in 1;
    do
       # added -u for printing to SLURM output file
        python -u train_GlasXC.py --data_root $data_root --dataset_info "$setup_path/$dataset_info" \
                                 --input_encoder_cfg "$setup_path/$inp_enc_cfg" --input_decoder_cfg "$setup_path/$inp_dec_cfg" \
                                 --output_encoder_cfg "$setup_path/$otp_enc_cfg" --output_decoder_cfg "$setup_path/$otp_dec_cfg" \
                                 --regressor_cfg "$setup_path/$reg_cfg" --device $device --epochs $epochs --batch_size $batch_size \
                                 --input_ae_loss_weight $inp_ae_wgt --output_ae_loss_weight $otp_ae_wgt \
                                 --optimizer_cfg "$setup_path/$opt_cfg" --seed $seed --plot --k $k --interval $interval \
                                 --init_scheme $init_scheme
    done
