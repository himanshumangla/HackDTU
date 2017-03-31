# HackDTU
Cardiac Assist

Motivation: To provide quality heart care assistance where expert cardiologists are not available such as rural areas or areas where medical expertise is needed. 
Problem Statements: A) To predict the probability of heart disease for patients admitted to a hospital emergency room with symptoms of chest pain
B) To produce a method that can classify real heart audio (also known as “beat classification”) into one of four categories:
1.	Normal 
2.	Murmur 
3.	Extra Heart Sound
4.	Artifact

and hence suggest suitable care, treatment or medication.  






A) Heart Disease Risk Analysis 


Step 1) Acquiring Dataset :

1. Title: UCI Machine Learning Repository: Heart Disease Dataset

2. Source Information:
   (a) Creators: 
       -- 1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
       -- 2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
       -- 3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
       -- 4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:
             Robert Detrano, M.D., Ph.D.
   (b) Donor: David W. Aha (aha@ics.uci.edu) (714) 856-8779   
   (c) Date: July, 1988



3. Relevant Information:
     This database contains 76 attributes. In particular, the Cleveland
     database is the only one that has been used by ML researchers to 
     this date.  The "goal" field refers to the presence of heart disease
     in the patient.  It is an integer which is zero or 1 (presence/ highly likely).
   
     The names and social security numbers of the patients were recently 
     removed from the database, replaced with dummy values. (PII)

     One file has been "processed", that one containing the Cleveland 
     database.  All four unprocessed files also exist in this directory.
    
4. Number of Instances: 
        Database:    # of instances:
          Cleveland: 303
          Hungarian: 294
        Switzerland: 123
      Long Beach VA: 200

5. Number of Attributes: 76 (including the predicted attribute)

6. Attribute Information:
   -- Only 14 used
      -- 1. #3  (age)       
      -- 2. #4  (sex)       
      -- 3. #9  (cp)        
      -- 4. #10 (trestbps)  
      -- 5. #12 (chol)      
      -- 6. #16 (fbs)       
      -- 7. #19 (restecg)   
      -- 8. #32 (thalach)   
      -- 9. #38 (exang)     
      -- 10. #40 (oldpeak)   
      -- 11. #41 (slope)     
      -- 12. #44 (ca)        
      -- 13. #51 (thal)      
      -- 14. #58 (num)       (the predicted attribute)
      3 age: age in years
      4 sex: sex (1 = male; 0 = female)
      9 cp: chest pain type
        -- Value 1: typical angina
        -- Value 2: atypical angina
        -- Value 3: non-anginal pain
        -- Value 4: asymptomatic
     10 trestbps: resting blood pressure (in mm Hg on admission to the 
        hospital)
     12 chol: serum cholestoral in mg/dl
     16 fbs: (fasting blood sugar > 120 mg/dl)  (1 = true; 0 = false)
     19 restecg: resting electrocardiographic results
        -- Value 0: normal
        -- Value 1: having ST-T wave abnormality (T wave inversions and/or ST 
                    elevation or depression of > 0.05 mV)
        -- Value 2: showing probable or definite left ventricular hypertrophy
                    by Estes' criteria
     32 thalach: maximum heart rate achieved
     38 exang: exercise induced angina (1 = yes; 0 = no)
     40 oldpeak = ST depression induced by exercise relative to rest
     41 slope: the slope of the peak exercise ST segment
        -- Value 1: upsloping
        -- Value 2: flat
        -- Value 3: downsloping
     44 ca: number of major vessels (0-3) colored by flourosopy
     51 thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
     58 num: diagnosis of heart disease (angiographic disease status)
        -- Value 0: < 50% diameter narrowing
        -- Value 1: > 50% diameter narrowing
        (in any major vessel: attributes 59 through 68 are vessels)

9. Missing Attribute Values: Several.  Distinguished with value -9.0.

10. Class Distribution:
        Database:      0   1   2   3   4 Total
          Cleveland: 164  55  36  35  13   303
          Hungarian: 188  37  26  28  15   294
        Switzerland:   8  48  32  30   5   123
      Long Beach VA:  51  56  41  42  10   200


This data set dates from 1988 and consists of four databases: Cleveland (303 instances), Hungary (294), Switzerland (123), and Long Beach VA (200). Each database provides 76 attributes, including the predicted attribute. There are many missing attribute values. 

Step 2) Feature Selection and cleaning the data:

The features in the data set are correlated, and the sample size is too small to determine all the feature coefficients with sufficient precision. This means that we need to be extra careful in selecting features for the final logistic regression model.  
We start by including all features, and then eliminate non-significant features one-by-one, in such a way so as to minimize the increase in deviance after each elimination
Step 3) Feature Selection and cleaning the data:
Building the model was tried out using various methods like CART, Random Forests, Naïve Bayes, and Logistic Regression with Logit having the highest precision.
The output of feature selection algorithm is the largest subset of features whose fit coefficients are at least two standard deviations away from zero, and such that the total change in fit deviance is the smallest possible. When applied to the logistic regression model for the Cleveland data, nine features were selected, five of which are binary variables coming from three of the original categorical attributes.
Age is not part of the optimal set of features, and the coefficients of all selected features have a z-value of at least 2 (in absolute value). Note also that the top four attributes by order of intrinsic discrepancy (thal, cp, ca, and thalach, see Table 2) are part of this final selection.

Step 4) Validation and evaluating precision results:
To obtain unbiased estimates of the accuracy, precision, and recall properties of the logistic model, three-way cross-validation procedure was used, dividing the Cleveland dataset randomly into three almost equal parts. 
For each part, training the model on the other two parts combined and measured its properties on the part not used for training. This gave three unbiased estimates of the desired model properties, which were then averaged.
The training included both the feature selection and the model fit. 
Accuracy (fraction of disease or no-disease predictions that are correct  (79.6±1.4)%. 
For the precision, (82.7±6.6)% (fraction of disease predictions that are correct) and (78.6±2.7)% (fraction of no-disease predictions that are correct). 
Finally, for the recall the numbers are (73.0±4.1)% (fraction of disease cases that are correctly identified) and (86.0±6.2)% (fraction of no-disease cases that are correctly identified).




B) Real-time Heart Audio classification
Step 1) Acquiring Dataset:
Kaggle Heart Beat Sounds Data : https://www.kaggle.com/kinguistics/heartbeat-sounds 
Citation : http://www.peterjbentley.com/heartchallenge/index.html

The data was collected from the general public via an iPhone app. The 4 categories of heartbeat sounds are-
In the Normal category there are normal, healthy heart sounds. These may contain noise in the final second of the recording as the device is removed from the body
A heart murmur itself does not require treatment. If it is caused by a more serious heart condition, your doctor may recommend treatment for that heart condition. Treatment may include medicines, cardiac catheterization, or surgery. 
The outlook and treatment for abnormal heart murmurs depend on the type and severity of the heart condition that is causing the murmur.
An extra heart sound may not be a sign of disease. However, in some situations it is an important sign of disease, which if detected early could help a person. The extra heart sound is important to be able to detect as it cannot be detected by ultrasound very well.
Artifact is the most different from the others. It is important to be able to distinguish this category from the other three categories, so that someone gathering the data can be instructed to try again
Description  
•	fname: the audio file 
•	label: either "normal", blank (for unlabelled data), or one of various categories of abnormal heartbeats 

Step 2) PyAudioAnalysis: 
We classify heartbeat sounds in one of 4 categories using PyAudioAnalysis.
pyAudioAnalysis is a Python library covering a wide range of audio analysis tasks. Through pyAudioAnalysis we can extract audio features and representations (e.g. mfccs, spectrogram, chromagram)
The code first trains an audio segment classifier, given a set of audio files stored in folders (each folder representing a different class) and then the trained classifier is used to classify an unknown audio file.
Step 3) Choice of Classifier:
We tried using SVM, KNN, and random forest classifier models to obtain a maximum  precision rate of ~0.8 using SVM. Random forest model is a good depiction, but takes too much of preprocessing time.

Step 4) Evaluation of results and precision: 
Classify unknown sounds. Train, parameter tune and evaluate classifiers of audio segments. Apply dimensionality reduction to visualize audio data and content similarities 


