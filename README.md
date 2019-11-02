# mlpWeatherLabeling

This software creates a MultiLayerPerceptron to fit a multi-class classification to recognize weather events. The forecasting model is based on MLP with three hidden-layer with 100 neurons, 10 input neurons, dinamic output neurons in base the number of labeling; regarding iper-parameters this machine learning softwar, use as loss function the "categorical_crossentropy" and as optimizer "adam". This software extracts csv file created by using Weather Labeling Web App; it will be downloadable from url https://github.com/CCMMMA/Weather-Labeling-Web-Application .

To resolve the problem of unbalanced dataset, I have used two solutions:
1. python mlMain1.py : this version creates a sample of 50% of the data set; another it can apply specific weight to each class
2. python mlMain2.py this version uses a oversampling RandomOverSampler by using library imblearn (python's component)

Configuration

1. Install Weather Labeling Web App: https://github.com/rosario-riccio/Weather-Labeling-Web-Application. It's necessary to execute all comands of configuration
2. git clone https://github.com/rosario-riccio/mlpWeatherLabeling.git
3. cd mlpWeatherLabeling
4. virtualenv venv
5. source venv/bin/activate
6. pip install --upgrade pip
7. pip install -r requirements.txt
8. insert your path where there are csv files for training set in mlMain1.py and mlMain2.py: src = "" 
9. insert your path where there are csv files for evalutation set in mlMain1.py and mlMain2.py: src1 = "" 
10. insert your path where there are csv files for prediction set in mlMain1.py and mlMain2.py: src2 = "" 
11. flag1 in mlMain1.py and mlMain2.py:True if h5 file existed, otherwise False.
12. flag2 in mlMain1.py and mlMain2.py: if flag1 is True, you set flag2 to True for evaluation, otherwise prediction
13. python mlMain1.py or python mlMain2.py 
