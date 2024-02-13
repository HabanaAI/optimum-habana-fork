## Docker build

```shell
docker build . -f Dockerfile-hpu -t text-embeddings-inference:hpu-0.6
```

## Docker run

```shell
model=sentence-transformers/all-MiniLM-L6-v2
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --ipc=host -p 8080:80 -v $volume:/data text-embeddings-inference:hpu-0.6 --model-id $model --pooling cls
```
