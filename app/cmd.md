
# 1. Docker 
## build
docker build -t search-app .
docker build -t search-app . --progress=plain

docker build --no-cache -t search-app .
The --no-cache flag tells Docker to ignore any cached layers and rebuild the image from scratch.


## run
docker run -it search-app


## access docker image 
docker run -it search-app /bin/bash
cd /app/cpu_and_mobile
ls

docker run -it search-app /bin/bash
cd /app/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4
ls


## remove 
docker stop $(docker ps -aq --filter ancestor=search-app)
docker rm $(docker ps -aq --filter ancestor=search-app)

docker rmi search-app


# Other 
node generate_description.js  


# Push colab to git 
## Generate SSH Key
!ssh-keygen -t rsa -b 4096 -C "your-email@example.com" -f /root/.ssh/id_rsa -N ""

## Display the SSH Public Key
!cat /root/.ssh/id_rsa.pub

## Manually: Add the SSH key to your GitHub account at https://github.com/settings/keys

## Add GitHub to known hosts
!ssh-keyscan github.com >> /root/.ssh/known_hosts

git config --global user.email "a.jliouat@yahoo.fr"
git config --global user.name "ajliouat"

## Clone the Repository using SSH
!git clone git@github.com:ajliouat/rag-based-recommendation-system-with-phi-3-mini-4k-instruct.git
%cd rag-based-recommendation-system-with-phi-3-mini-4k-instruct

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy Your Notebook
cp "/content/drive/MyDrive/Colab Notebooks/RAG-Based Recommendation System with Phi-3-Mini-4K-Instruct.ipynb" "/content/rag-based-recommendation-system-with-phi-3-mini-4k-instruct/"




# Commit and Push to GitHub
!git add your_notebook.ipynb
!git commit -m "Add initial notebook"

git push origin main
git push origin main


