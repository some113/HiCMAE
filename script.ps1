$env:path += ';C:\Users\Lenovo\miniconda3'
$env:path += ';C:\Users\Lenovo\miniconda3\Scripts'
$env:path += ';C:\Users\Lenovo\miniconda3\Library\bin'

# Define variables
$server = 170
$pretrain_dataset = 'voxceleb2'
$finetune_dataset = 'crema-d'
$num_labels = 6
$model_dir = "hicmae_pretrain_base"
$ckpts = @(99)
$input_size = 160
$input_size_audio = 256
$sr = 4
$lr = 1e-3
$epochs = 50
$splits = @(1)

foreach ($split in $splits) {
    foreach ($ckpt in $ckpts) {
        # Output directory
        $OUTPUT_DIR = ".\saved\model\finetuning\$finetune_dataset\audio_visual\$pretrain_dataset`_$model_dir\checkpoint-$ckpt\eval_split0$split\_lr_$lr\_epoch_$epochs\_size$input_size\_a$input_size_audio\_sr$sr\_server$server"
        if (!(Test-Path $OUTPUT_DIR -PathType Container)) {
            New-Item -Path $OUTPUT_DIR -ItemType Directory -Force | Out-Null
        }
        # Path to split files
        $DATA_PATH = ".\saved\data\$finetune_dataset\audio_visual\split0$split"
        # Path to pre-trained model
        $MODEL_PATH = ".\saved\model\pretraining\$pretrain_dataset\audio_visual\$model_dir\checkpoint-$ckpt.pth"

        # Run script
	set CUDA_VISIBLE_DEVICES=0,1,2,3
        $pythonCommand = "python -m torch.distributed.launch --nproc_per_node=4 --master_port 13296 run_class_finetuning_av.py --model avit_dim512_patch16_160_a256 --data_set $($finetune_dataset.ToUpper()) --nb_classes $num_labels --data_path $DATA_PATH --finetune $MODEL_PATH --log_dir $OUTPUT_DIR --output_dir $OUTPUT_DIR --batch_size 14 --num_sample 1 --input_size $input_size --input_size_audio $input_size_audio --short_side_size $input_size --depth 10 --depth_audio 10 --fusion_depth 2 --save_ckpt_freq 1000 --num_frames 16 --sampling_rate $sr --opt adamw --lr $lr --opt_betas 0.9 0.999 --weight_decay 0.05 --epochs $epochs --dist_eval --test_num_segment 2 --test_num_crop 2 --num_workers 16 >$OUTPUT_DIR\nohup.out 2>&1"

        # Execute Python command
        Invoke-Expression $pythonCommand
    }
}

Write-Host "Done!"
