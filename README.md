## EMLO V2 - Assignment 2 


- To create a docker image for this project, run 

  ```
  make build
  ```

- To train and evaluate a pretrained model from the `timm` library on `CIFAR10` dataset based on config defined in the `configs/experiment/example/yaml` file, run : 
  ```
  make run
  ```

- To create a docker image from config specified in the `cog.yaml` file, run 

  ```
  make cog-train
  ```

- To run inference on an image using a pretrained model available in the `timm` library, run 

  ```
  cog predict -i image=@input.jpg -i model_name=resnet18 
  ```