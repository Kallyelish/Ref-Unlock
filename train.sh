# set -x
# GPUS=$1

# PY_ARGS=${@:2}

# CUDA_VISIBLE_DEVICES=${GPUS} python train.py ${PY_ARGS}

# CUDA_VISIBLE_DEVICES=0 python train.py -s data/art1 --eval -m results/art1_0.0001depth_0.001ref
# CUDA_VISIBLE_DEVICES=1 python train.py -s data/art2 --eval -m results/art2_0.0001depth_0.001ref
# CUDA_VISIBLE_DEVICES=2 python train.py -s data/art3 --eval -m results/art3_0.0001depth_0.001ref
# CUDA_VISIBLE_DEVICES=3 python train.py -s data/bookcase --eval -m results/bookcase_0.0001depth_0.001ref
# CUDA_VISIBLE_DEVICES=3 python train.py -s data/tv --eval -m results/tv_0.0001depth_0.001ref

# CUDA_VISIBLE_DEVICES=0 python render.py -s data/art1 -m results/art1_0.0001depth_0.001ref --eval --render_images --skip_train
# CUDA_VISIBLE_DEVICES=1 python render.py -s data/art2 -m results/art2_0.0001depth_0.001ref --eval --render_images --skip_train
# CUDA_VISIBLE_DEVICES=2 python render.py -s data/art3 -m results/art3_0.0001depth_0.001ref --eval --render_images --skip_train
# CUDA_VISIBLE_DEVICES=3 python render.py -s data/bookcase -m results/bookcase_0.0001depth_0.001ref --eval --render_images --skip_train
# CUDA_VISIBLE_DEVICES=3 python render.py -s data/tv -m results/tv_0.0001depth_0.001ref --eval --render_images --skip_train

# CUDA_VISIBLE_DEVICES=3 python metrics.py -m results/art1_0.0001depth_0.001ref
# CUDA_VISIBLE_DEVICES=3 python metrics.py -m results/art2_0.0001depth_0.001ref
# CUDA_VISIBLE_DEVICES=3 python metrics.py -m results/art3_0.0001depth_0.001ref
# CUDA_VISIBLE_DEVICES=3 python metrics.py -m results/bookcase_0.0001depth_0.001ref
# CUDA_VISIBLE_DEVICES=3 python metrics.py -m results/tv_0.0001depth_0.001ref

# CUDA_VISIBLE_DEVICES=0 python train.py -s /home/fb_21110240032/refnerf/car --eval -m results/car -w

# CUDA_VISIBLE_DEVICES=0 python render.py -s /home/fb_21110240032/refnerf/car -m results/car --eval --render_images --skip_train -w

# CUDA_VISIBLE_DEVICES=0 python metrics.py -m results/car

# CUDA_VISIBLE_DEVICES=0 python train.py -s /home/fb_21110240032/refnerf/coffee --eval -m results/coffee -w

# CUDA_VISIBLE_DEVICES=0 python train.py -s /home/fb_21110240032/refnerf/helmet --eval -m results/helmet -w

# CUDA_VISIBLE_DEVICES=0 python train.py -s /home/fb_21110240032/refnerf/teapot --eval -m results/teapot -w

# CUDA_VISIBLE_DEVICES=0 python train.py -s /home/fb_21110240032/refnerf/toaster --eval -m results/toaster -w

# CUDA_VISIBLE_DEVICES=0 python render.py -s /home/fb_21110240032/refnerf/coffee -m results/coffee --eval --render_images --skip_train -w

# CUDA_VISIBLE_DEVICES=0 python render.py -s /home/fb_21110240032/refnerf/helmet -m results/helmet --eval --render_images --skip_train -w

# CUDA_VISIBLE_DEVICES=0 python render.py -s /home/fb_21110240032/refnerf/teapot -m results/teapot --eval --render_images --skip_train -w

# CUDA_VISIBLE_DEVICES=0 python render.py -s /home/fb_21110240032/refnerf/toaster -m results/toaster --eval --render_images --skip_train -w



# CUDA_VISIBLE_DEVICES=0 python metrics.py -m results/coffee

# CUDA_VISIBLE_DEVICES=0 python metrics.py -m results/helmet

# CUDA_VISIBLE_DEVICES=0 python metrics.py -m results/teapot

# CUDA_VISIBLE_DEVICES=0 python metrics.py -m results/toaster



CUDA_VISIBLE_DEVICES=0 python train.py -s /home/fb_21110240032/art1 --eval -m results/art1_1 -d /home/fb_21110240032/art1/depth 

CUDA_VISIBLE_DEVICES=0 python render.py -s /home/fb_21110240032/art1 -m results/art1_1 --eval --render_images --skip_train  -d /home/fb_21110240032/art1/depth

CUDA_VISIBLE_DEVICES=0 python metrics.py -m results/art1_1

CUDA_VISIBLE_DEVICES=0 python train.py -s /home/fb_21110240032/art2 --eval -m results/art2_1 -d /home/fb_21110240032/art2/depth 

CUDA_VISIBLE_DEVICES=0 python render.py -s /home/fb_21110240032/art2 -m results/art2_1 --eval --render_images --skip_train  -d /home/fb_21110240032/art2/depth

CUDA_VISIBLE_DEVICES=0 python metrics.py -m results/art2_1

CUDA_VISIBLE_DEVICES=0 python train.py -s /home/fb_21110240032/art3 --eval -m results/art3_1 -d /home/fb_21110240032/art3/depth 

CUDA_VISIBLE_DEVICES=0 python render.py -s /home/fb_21110240032/art3 -m results/art3_1 --eval --render_images --skip_train  -d /home/fb_21110240032/art3/depth

CUDA_VISIBLE_DEVICES=0 python metrics.py -m results/art3_1