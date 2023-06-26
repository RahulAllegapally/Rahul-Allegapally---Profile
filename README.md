python version -> 3.5
Dependencies:
1)Numpy
2)Scipy
3)Pandas
4)Sklearn
5)NLTK

***steps*****************

$ sudo pip3 install numpy
$ sudo pip3 install scipy
$ sudo pip3 install pandas
$ sudo pip3 install sklearn
$ sudo pip3 install nltk

For nltk open your CLI and run python3
Python 3.5.2 (default, Nov 17 2016, 17:05:23) 
[GCC 5.4.0 20160609] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import nltk
>>> nltk.download()
NLTK Downloader
---------------------------------------------------------------------------
    d) Download   l) List    u) Update   c) Config   h) Help   q) Quit
---------------------------------------------------------------------------
Downloader> d

Download which package (l=list; x=cancel)?
  Identifier> l
Packages:
  [ ] averaged_perceptron_tagger_ru Averaged Perceptron Tagger (Russian)
  [ ] pe08................ Cross-Framework and Cross-Domain Parser
                           Evaluation Shared Task

Collections:

([*] marks installed packages)

Download which package (l=list; x=cancel)?
  Identifier> all

########### untar the phrase_extractor.tar.gz file ####################

$ tar -xvzf phrase_extractor.tar.gz
$ cd phrase_extractor/


######################### running the code ########################### 

$ python3 script.py

############################## Done ###################################
