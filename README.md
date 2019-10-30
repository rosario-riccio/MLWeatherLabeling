# mlpWeatherLabeling

This software creates a MultiLayerPerceptron to fit a multi-class classification to recognize weather events. The forecasting model is based on MLP with only hidden-layer with 50 eurons, 10 input neurons, dinamic output neurons in base the number of labeling; regarding iper-parameters this machine learning softwar, use as loss function the "categorical_crossentropy" and as optimizer "adam". This software extracts csv file created by using Weather Labeling Web App; it will be downloadable from url https://github.com/CCMMMA/Weather-Labeling-Web-Application

Configuration

1. Install Weather Labeling Web App: https://github.com/rosario-riccio/Weather-Labeling-Web-Application. It's necessary to execute all comands of configuration.
2. git clone https://github.com/rosario-riccio/mlpWeatherLabeling.git
3. cd Weather-Labeling-Web-Application/
4. virtualenv venv
5. source venv/bin/activate
6. pip install --upgrade pip
7. pip install -r requirements.txt
8. insert your path where there are csv files into variable src for training set in mlMain1.py and mlMain2.py: src = ""
9. insert your path where there are csv files into variable src for prediction set in mlMain1.py and mlMain2.py: src1 = ""
10. python mlMain1.py : this version create a sample from dataset of training of 50%; another it can apply specific weight to each class
11. python mlMain2.py this version use a oversampling RandomOverSampler by using library imblearn (python's component)
