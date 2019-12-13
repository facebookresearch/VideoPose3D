# VideoPose3d Dockerfile

>This dockerfile aims to help to get started with the [Inference](../INFERENCE.md#Step-3:-inferring-2D-keypoints-with-Detectron) for testing VideoPose3d with your own videos.
>
>It conducts [Step 1: Setup](../INFERENCE.md#Step-1:-setup)

It is created for Systems that have a RTX graphic card *(maybe other work as well)* and Cuda10 installed on the host system.

## How to work with docker:
- `docker build -t detectron_cu10:latest .` *(this takes quite some time, don't worry if there pop up some red warnings)*
- `nvidia-docker run --rm -it detectron_cu10:latest python detectron/tests/test_batch_permutation_op.py` *(should say: 2 tests OK)*
- `docker run -itd --name detcu10 --runtime=nvidia detectron_cu10` *(run container in detached mode)*
- `docker exec -it detcu10 /bin/bash` &rarr; log into container

&rarr; now you can continue with [Step 2](../INFERENCE.md#Step-2-(optional):-video-preprocessing)

Happy Coding :computer: :tada: