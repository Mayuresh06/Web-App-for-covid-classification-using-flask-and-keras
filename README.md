# Web-App-for-covid-classification-using-flask-and-keras
Covid-19 is a pandemic disease caused by a virus (the SARS-CoV-2 corona virus), which has already infected millions of people, causing the death of hundreds of thousands in a few months. According to the World Health Organization (WHO), most patients with COVID-19 (about 80%) may be asymptomatic and about 20% of cases may require hospital care because they have difficulty breathing. Of those cases, approximately 5% may need support for the treatment of respiratory failure (ventilator support), a situation that can collapse Intensive Care facilities. A method to fast test that who has the virus is a key in combating the pandemic. The use of imaging data has been reported to be useful for rapid diagnosis of COVID-19.Although computed tomography (CT) scans show a variety of signs caused by the viral infection, given a large amount of images, these visual features are difficult and can take a long time to be recognized by radiologists. Artificial intelligence methods for automated classification of COVID-19 on chest X-Ray scans have been found to be very promising. X-ray machines are cheaper, more straightforward, and faster to operate, and are therefore more accessible than CTs to healthcare professionals working in more impoverished or more remote regions. One of the significant challenges in combating Covid-19 is testing the presence of the virus in people. Thus, the objective of this project is to automatically detect the virus that causes Covid-19 in patients with Pneumonia (and even in asymptomatic, or not sick people), using scanned chest X-ray images. These images are pre-processed and used for the training of Convolutional Neural Network (CNN) model. CNN-type networks generally need an extensive dataset to function, so we downloaded the dataset from kaggle containing 1000’s of labelled images divided into three classes i.e. COVID, Viral Pneumonia and healthy. For the training of the models, tools, libraries, and resources of Tensor Flow (with Keras) are used, which is an open-source platform used in Deep Learning. All the training part was completed on Google Colab platform as it provides a GPU for training large models. The final model was deployed to a web application which was developed using Flask framework and the front end was developed in HTML for testing in situations close to reality. From the scanned image of a Chest X-Ray (User_A.png), stored locally on the web-app user’s computer, the application decides whether the image belongs to a person who is contaminated by the Corona virus or Pneumonia or is Healthy person (Model Prediction: [COVID] or [VIRAL] or [NORMAL]). The user has to input his name, age, gender and symptoms along with the chest x-ray and all this with the prediction is stored on the local device, this helps in maintaining a proper database of the patients and increases the speed of testing.

### The work is divided into 3 parts:
1.	Environment setup, data acquisition, cleaning and preparation
2.	Model Training (Covid/Normal/Pneumonia)
3.	Development and testing of a Web App for the detection of Covid-19 in X-ray images

### OUTPUT IMAGES
![](https://github.com/Mayuresh06/Web-App-for-covid-classification-using-flask-and-keras/blob/main/Screenshot%20(183).png)

![](https://github.com/Mayuresh06/Web-App-for-covid-classification-using-flask-and-keras/blob/main/Screenshot%20(183).png)

![](https://github.com/Mayuresh06/Web-App-for-covid-classification-using-flask-and-keras/blob/main/Screenshot%20(183).png)

* Dataset was downloaded from kaggle(https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).
* Data acquisition, data preprocessing and CNN model training code is given in .py
* Web app creation and API for CNN model integration given in App.py file.
* CNN model saved file is given as covid_classifier_final_model.h5
* Front end related HTML and CSS files are gicen in Static and Tempelate folders. 

