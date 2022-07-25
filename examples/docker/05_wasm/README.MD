Change to the current directory
```
cd examples/docker/05_wasm
```
Build the docker image (which is tagged as `backprop_tools:wasm`)
```
docker build -f Dockerfile -t backprop_tools:wasm . --no-cache
````
Run the image just built (exposing port 8000 for the web server and mounting the source code from the host in readonly mode)
```
docker r