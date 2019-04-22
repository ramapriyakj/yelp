# Yelp dataset analysis

In this project we perform data analysis on yelp text data set (https://www.yelp.com/dataset).

The following files are used to perform the analysis:

**review.json** file is used to perform sentimental analysis on businesses.

**business.json** file is used to perform clustering of restaurants.

The **YelpAnalysis.ipynb** and **YelpUnsupervisedAnalysis.ipynb** shows the result of the analysis.

Steps to run the analysis using docker (assuming docker is already installed):

1. Download JSON (yelp_dataset.tar) from (https://www.yelp.com/dataset/download)
2. Extract business.json and review.json from yelp_dataset.tar file into **~/data** folder.
3. Download **code** folder to your local machine's home folder **~/code**
4. Run the below commands to open the docker terminal:
```
cd ~/code
docker build . -t yelprun
docker run -it yelprun bash
```
6. Note the docker container name<**container_name**> by running the below command:
```
docker ps -a 
```
5. Open new terminal from local machine and run the below commands to copy files to docker container:
```
sudo docker cp ~/code/YelpAnalysis.py <container_name>:/home
sudo docker cp ~/code/YelpUnsupervisedAnalysis.py <container_name>:/home
sudo docker cp ~/data/review.json <container_name>:/home
sudo docker cp ~/code/business.json <container_name>:/home
```
6. From the docker terminal, run the below commands one after the other to perform supervised and unsupervised analysis on yelp dataset respectively:
```
python YelpAnalysis.py
python YelpUnsupervisedAnalysis.py
```
