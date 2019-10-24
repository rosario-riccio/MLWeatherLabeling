# mlpWeatherLabeling

This software creates a MultiLayerPerceptron to fit a multi-class classification to recognize weather events. The forecasting model is based on MLP with only hidden-layer with 50 eurons, 10 input neurons, dinamic output neurons in base the number of labeling; regarding iper-parameters this machine learning softwar, use as loss function the "categorical_crossentropy" and as optimizer "adam". This software extracts csv file created by using Weather Labeling Web App; it will be downloadable from url https://github.com/CCMMMA/Weather-Labeling-Web-Application

Configuration

1. git clone https://github.com/rosario-riccio/mlpWeatherLabeling.git
2. cd Weather-Labeling-Web-Application/
3. virtualenv venv
4. source venv/bin/activate
5. pip install --upgrade pip
6. pip install -r requirements.txt
7. insert your path where there are csv files into variable src in mlMain.py: src = ""
8. python mlMain.py
