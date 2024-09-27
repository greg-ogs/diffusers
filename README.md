# build:
docker build -t gregogs/cuda:diffusers-2.16.1 .
# run:
docker run --gpus all -it --rm -v D:\dev\NeuralNet\Stable_difussion:/app gregogs/cuda:diffusers-2.16.1
