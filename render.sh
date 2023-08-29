set -aux

export CUDA_VISIBLE_DEVICES="0"
# --load-config outputs/clone/nsg-vkitti-car-depth-recon/2023-08-01_104913/config.yml \ # 好的
# --load-config outputs/clone/nsg-vkitti-car-depth-recon/2023-08-02_084422/config.yml \


# config=outputs/15-deg-left/nsg-vkitti-car-depth-recon/2023-08-02_084557/config.yml
# config=outputs/reduce_eval/nsg-vkitti-car-depth-recon/2023-08-02_155442/config.yml 
# config=outputs/clone/nsg-vkitti-car-depth-recon/2023-08-02_084422/config.yml
config=outputs/nvs50fullseq/nsg-vkitti-car-depth-nvs/2023-06-21_140103/config.yml
time=20

output_name=`echo $(dirname "$config") | sed 's/outputs\///; s/\//-/g'`
folder_path=$(dirname "$config")
latest_ckpt=$(find "$folder_path" -type f -name "*.ckpt" | sort -r | head -1)
ckpt_name=$(basename -- "$latest_ckpt")

# exit
python scripts/cicai_render.py \
--load-config $config \
--output_path "renders/pretrain_changeX02_-$ckpt_name.mp4" \
--seconds $time
