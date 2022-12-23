# Model_template
This is Machine learning template for python with sklearn

## requirements.txt 
- package in 'requirements.txt', version as below:
```
numpy == 1.20.3
pandas == 1.3.4
matplotlib == 3.4.3
torch == 1.10.2
torchvision == 0.11.3
tensorflow == 2.8.0
scikit-learn == 0.24.2
```
*run ```pip install -r requirements.txt``` in CLI to isntall.

## my_temp
```
|_ main.py
   utils.py
   pipeline
   |_ pipeline.py
      steps
      |_ model.py
         preglight.py
         preprocess.py
         step.py
 ```

## main.py
 main function serves as the entry point for the program.

## utils.py
 common extended function, including drawing,....

## step.py
 framework for the program (use abstract method).

## preflight.py
 default for the program initial (now is empty).

## preprocess.py
 data preprocess, including data transform, nan or null data cleaning,...

## model.py
 build ML model and model process,  including model, imbalanced data,...
 
## model_compare.py
compare diff ML model r2 or accuracy.

## plot_model_compare.py
plot the result of diff ML model comparation (r2 or accuracy).

## remark



