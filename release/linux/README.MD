### Recommended: Using docker to create the release
```
docker build -t backprop_tools:release_linux . --no-cache
```
```
docker run --rm -it --mount type=bind,source=$(cd ../..; pwd),target=/backprop_tools,readonly --mount type=bind,source=$(pwd)/install,target=/install -v $(pwd)/build.sh:/build.sh:ro backprop_tools:release_linux
/build.sh /install
````

#### Testing the release
#### Headless
```
docker run --rm -it -w / --mount type=bind,source=$(pwd)/checkpoints,target=/checkpoints --mount type=bind,source=$(pwd)/install,target=/install --cap-add=sys_nice --runtime=nvidia --gpus all archlinux nice -n -20 /install/backprop_tools/bin/rl_environments_mujoco_ant_training_ppo_cuda_full 
```
#### With UI
```
docker build -