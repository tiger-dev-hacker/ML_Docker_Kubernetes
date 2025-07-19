#Prototype Scikit-learn ML model with Streamlit App deployed to Docker and Kubernetes

#Requirements
1. Python

#To install the requirements
```sh
sudo apt update && sudo apt upgrade -y
pip install -r requirements.txt
```
#Install and create conda environment
```sh
pip install conda
```
restart the terminal 

```sh
conda activate 
```

```sh
conda create <new-env>
```

#Run Streamlit App
```sh
streamlit run app.py
```

##To run on the browser
```sh
http://localhost:8501
```

```sh
http://<public-ip>:8501
```
Setup docker by creating a docker account 
```sh
sudo snap install docker
```

#Build and Run the Docker Image Using the Dockerfile
```sh
docker build -t streamlit-ml-app:latest
```

```sh
docker run -p 8501:8501 streamlit-ml-app:latest
```

##Push Streamlit Docker Image to DockerHub
Go to your Dockerhub account and create a Personal Access Token(PAT) - This will be your password -
```sh
docker login -u <user-name>
```

##Tag your Local Docker Image
```sh
docker tag streamlit-ml-app:latest <user-name>/streamlit-ml-app:latest
```

###Push image to dockerHub
```sh
docker push <user-name>/streamlit-ml-app:latest
```
###Deploy Stremlit App Docker Image from DockerHub to Kubernetes
Review manifest files

###Create minikube cluster
```sh
minikube start --driver docker
```

##Deploy the Kubernetes Manifest Files
```sh
kubectl apply -f deployment.yaml
```

```sh
kubectl apply -f service.yaml
```

##Check the resources created
```sh
kubectl get pods
```

```sh
kubectl get svc
```
```sh
minikube ip
```

##open in browser

```sh
http:<minikube-ip>:30001
```

##In case firewall or security rules redirect the above url, check your service running at
```sh
minikube start streamlit-ml-app or <name-of-service>

##Clean up
```sh
minikube stop
```

```sh
minikube delete --all
```

