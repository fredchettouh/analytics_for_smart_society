# -*- coding: utf-8 -*-
"""Lab_Exercise1_students.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1HQAM4jvvlanxy-65iuyS3agmyHHOYGaE

![uc3m](http://materplat.org/wp-content/uploads/LogoUC3M.jpg)
#### Mount your Google Drive
"""

# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')

# Change to assignment directory ('Lab_Exercises_DASS/Lab_Exercise1' by default)
import os
os.chdir('/content/drive/My Drive/Colab Notebooks/Lab_Exercises_DASS/Lab_Exercise1')

"""#### Required Python libraries for the lab session"""

import numpy  as np
import matplotlib.pyplot as plt
import librosa  # package for speech and audio analysis
from sklearn import mixture
from sklearn.metrics import accuracy_score,confusion_matrix
plt.style.use('default')

"""# LAB EXERCISE 1
---

# SPEAKER IDENTIFICATION

### DATA ANALYTICS FOR THE SMART SOCIETY

### MASTER IN BIG DATA ANALITYCS
### COURSE 2019/2020

1. OBJECTIVE

2. DATABASE

3. READING SPEECH FILES AND FEATURE EXTRACTION

4. SPEAKER IDENTIFICATION SYSTEM

> 4.1. TRAINING STAGE

> 4.2. TEST STAGE (CLEAN CONDITIONS)

> 4.3. EVALUATION (CLEAN CONDITIONS)

> 4.4. TEST STAGE (NOISY CONDITIONS)

> 4.5. EVALUATION (NOISY CONDITIONS)

5. IMPROVEMENTS ON THE BASELINE SYSTEM [OPTIONAL]

---
## 1. OBJECTIVE 
---
The objective of this lab exercise is to implement a **text-independent Speaker Identification (SI) system based on Gaussian Mixture Models (GMM)**. This system takes a speech utterance from an unknown speaker and provides the name or identification code of that speaker.

---
## 2. DATABASE 
---
The database contains speech from **16 different speakers** and  it has been recorded at a **sampling frequency of 16 kHz** (subdirectory ***“speechdata”***). It comprises speech files of three different types:

* **Training material.** These files  (listed in the file ***“Train_List.txt”***) will be used for building the specific-speaker models.

* **Test material 1.** These files (listed in the file ***“Test_List1.txt”***) will be used for evaluating the system in clean conditions.

* **Test material 2.** These files (listed in the file ***“Test_List2.txt”***) will be used for evaluating the system in noisy conditions.

The three lists have the following format per line:

```
name_of_speech_file   speaker_identifier
```
"""

# --------------------------------------------------------------------
# Lists of training and testing speech files
# --------------------------------------------------------------------
nomlist_train = 'Train_List.txt';
nomlist_test = ['Test_List1.txt', 'Test_List2.txt']
print("Training list: " + nomlist_train)
print("Test list 1 (clean conditions): "+ nomlist_test[0])
print("Test list 2 (noisy conditions): "+ nomlist_test[1])

"""---
## 3. READING SPEECH FILES AND FEATURE EXTRACTION 
---
The following code shows an example of reading a speech file with the function **load** from the *librosa* library.
"""

# --------------------------------------------------------------------
# Reading a speech file from the database
# --------------------------------------------------------------------
speech_name = 'speechdata/irm01/irm01_s01_test1.wav'
print(speech_name)

# x: speech signal
# fs: sampling frequency
x, fs = librosa.load(speech_name, sr=None)
print("Number of samples in file "+speech_name+" = "+str(x.shape[0]))
print("Sampling frequency = "+str(fs)+" Hz")

fig, ax = plt.subplots()
plt.plot(x)
plt.title('Waveform')
ax.set_xlabel('Time (samples)')
plt.tight_layout()
plt.show()

"""In the following, we show an example of feature extraction for the previous speech signal *x*.

In particular, we want to compute a set of mel-frequency cepstrum coefficients (MFCC) with the following configuration:

* Size of the analysis window = 20 ms

* Frame period or hop length = 10 ms

* Number of filters in the mel filterbank = 40

* Number of MFCC components = 20

For doing that, we are going to use the function **mfcc** from the module *feature* of the *librosa* package. This function has, among others, the following input arguments:

* y: speech signal 
* sr: sampling frequency
* n_fft: window size (in samples)
* hop_length: frame period or hop length (in samples)
* n_mels: number of filters in the mel filterbank
* n_mfcc: number of MFCC components

Note that in this function the window size and the hop length must be expressed in samples. Taking into account that the sampling frequency (fs) indicates that 1 second correspond to fs samples (in our case, as fs=16 kHz, 1 second corresponds to 16000 samples), the conversion from **seconds** to **samples** is performed by:

```
samples = seconds*fs = seconds*16000
```

With the previous information, fill the values of the configuration variables in the next code.
"""

# --------------------------------------------------------------------
# Configuration variables for feature extraction
# --------------------------------------------------------------------

##################################
# - FILL THE VALUES OF THE FOLLOWING CONFIGURATION VARIABLES
##################################

fs = ;        # Sampling frequency
wst = ;       # Window size (seconds)
fpt = ;       # Frame period (seconds) 
nfft = ;      # Window size (samples)
fp = ;        # Frame period (samples)
nbands = ;    # Number of filters in the filterbank
ncomp = ;     # Number of MFCC components

"""The dimensions of the resulting MFCC must be,

```
(ncomp x T)
```

being *T* the number of frames of the speech signal, which is computed as the number of samples of the speech signal divided by the hop length (in samples), rounded off to the ceil integer.

The following code computes the MFCC features considering the previously configured variables.

Check the dimensions of the resulting MFCC features.
"""

# --------------------------------------------------------------------
# Example of MFCC computation
# --------------------------------------------------------------------

# Feature extraction (MFCC computation)
mfcc = librosa.feature.mfcc(y=x, sr=fs, n_fft=nfft, hop_length=fp, n_mels=nbands, n_mfcc=ncomp)

##################################
# - WRITE YOUT CODE HERE
##################################

"""---
## 4. SPEAKER IDENTIFICATION SYSTEM 
---
The speaker identification system to be implemented consists of two main stages,

* **Enrollment or training phase.** 

* **Identification or test phase.**

# 4.1. TRAINING STAGE
In this stage, the models of the different speakers in the system are built. For doing that, the following steps must be done,

*	Loading of the training material (speech files listed in ***“Train_List.txt”***).

* For each speaker (recall that there are 16 differente speakers in the database),

> *	**Feature extraction**. As baseline features, we recommend the use of the conventional **MFCC parameters**, setting **the window size, the frame period and the number of cepstral components to 20 ms, 10 ms and 20, respectively** (see Section 3).

> * **Speaker GMM models building.** The **GaussianMixture** function from the *mixture* module of the *scklearn* package, which implements the EM algorithm, can be used for this purpose. Use **8 gaussians** per model and **diagonal covariances**. 

At the end of this process, a set of GMM models (one model per speaker) is generated. Note that in this stage, only the training material can be used (listed in ***“Train_List.txt”***).
"""

# --------------------------------------------------------------------
# Training stage
# --------------------------------------------------------------------

##################################
# - FILL THE VALUES OF THE FOLLOWING CONFIGURATION VARIABLES
##################################
# Configuration variables for the training stage
nspk = ;     # Number of speakers 
ngauss = ;   # Number of gaussians in the GMM models

# Loading of the training files for all speakers
# The format of the list (per line) is,
# name_of_speech_file   speaker_identifier
fid = open(nomlist_train,"r")
data = np.genfromtxt(fid,dtype='str')
fid.close()

# Number of training files
nfiles_train = data.shape[0]
print(nfiles_train)

# 'files_speech_train': name of each speech file in the training set
# 'labels_train': label (speaker identifier) corresponding to each speech file
files_speech_train = []
labels_train = []
for i in range(nfiles_train):
  files_speech_train.append(data[i][0])
  labels_train.append(data[i][1])

labels_train = np.array(labels_train,dtype=np.int16)

# Speaker GMM models building
models = []
for i in range(nspk):

  # Loading of the training file for speaker "i"
  ##################################
  # - WRITE YOUT CODE HERE
  ##################################
  
  # Feature extraction
  ##################################
  # - WRITE YOUT CODE HERE
  ##################################   

  # Speaker GMM model building for speaker "i"
  ##################################
  # - WRITE YOUT CODE HERE
  ##################################

"""# 4.2. TEST STAGE (CLEAN CONDITIONS)
In this stage, the identified or predicted speaker is obtained for the case of clean conditions. For doing that, the following steps must be done,

*	Loading of the test material in **clean conditions** (speech files listed in ***“Test_List1.txt”***).

* For each test file of the unknown speaker,

> *	**Feature extraction**. We extract MFCC parameters with the same configuration as in the training stage.

> * **Log-likelihood computation of each model for the current test file.** 
We have to compute the log-likelihood of the test MFCCs given each one of the 16 speaker models obtained in the training stage. The **score_samples** function can be used for this purpose.

> * **Selection of the identified speaker**. We select the identified or predicted speaker according to the model which achieves the maximum log-likelihood.
"""

# --------------------------------------------------------------------
# Test stage (clean conditions)
# --------------------------------------------------------------------

# Reading of the list containing the speech test files
# The format of the list (per line) is,
# name_of_speech_file   speaker_identifier
fid = open(nomlist_test[0],"r")
data = np.genfromtxt(fid,dtype='str')
fid.close()

# Number of test files  
nfiles_test = data.shape[0]
print(nfiles_test)

# 'files_speech_test': name of each speech file in the test set
# 'labels_test': label (speaker identifier) corresponding to each speech file
files_speech_test = []
labels_test = []
for i in range(nfiles_test):
  files_speech_test.append(data[i][0])
  labels_test.append(data[i][1])

labels_test = np.array(labels_test,dtype=np.int16)

# 'pred_spk': predicted spakers
pred_spk = np.zeros(nfiles_test,dtype=np.int16)

# Loop for each test file
for k in range(nfiles_test):

  # Loading of the test files
  ##################################
  # - WRITE YOUT CODE HERE
  ##################################
      
  # Feature extraction
  ##################################
  # - WRITE YOUT CODE HERE
  ##################################

  # Log-likelihood computation of each model for the current test file
  ##################################
  # - WRITE YOUT CODE HERE
  ##################################

  # Selection of the identified speaker
  ##################################
  # - WRITE YOUT CODE HERE
  ##################################

"""# 4.3. EVALUATION (CLEAN CONDITIONS)
This task consists of the evaluation of the system in clean conditions.

The performance of the speaker identification system is measured in terms of the **identification accuracy** (percentage of correctly identified test files with respect to the total number of test files).
"""

# --------------------------------------------------------------------
# Evaluation (clean conditions)
# --------------------------------------------------------------------

# Computation of the identification accuracy
##################################
# - WRITE YOUT CODE HERE
##################################

"""# 4.4. TEST STAGE (NOISY CONDITIONS)
In this stage, the identified or predicted speaker is obtained for the case of noisy conditions.

Repeat the Section 4.2 but now using the test material in **noisy conditions** (speech files listed in ***“Test_List2.txt”***).
"""

# --------------------------------------------------------------------
# Test stage (noisy conditions)
# --------------------------------------------------------------------

##################################
# - WRITE YOUT CODE HERE
##################################

"""# 4.5. EVALUATION (NOISY CONDITIONS)
Evaluate the performance of the system in noisy conditions by computing the **identification accuracy** as in Section 4.3.
"""

# --------------------------------------------------------------------
# Evaluation (noisy conditions)
# --------------------------------------------------------------------

# Computation of the identification accuracy
##################################
# - WRITE YOUT CODE HERE
##################################

"""---
## 5. IMPROVEMENTS ON THE BASELINE SYSTEM [OPTIONAL] 
---
In this task, students can propose any improvement of the baseline system by, for example, including new acoustic features (LPC, …) or trying different number of Gaussians in the GMM models or computing confusion matrices for the evalution of the system.
"""

# --------------------------------------------------------------------
# Improvements on the baseline system
# --------------------------------------------------------------------

##################################
# - WRITE YOUT CODE HERE
##################################