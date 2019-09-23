# World-Music
### Project in DD2430 at KTH

#### Structuring the data
1. Make sure you have the full smithsonian dataset. 
It is ignored by git due to size.
2. Create a directory named *audio*. 
This will contain the resulting data and will also be ignored by git.
3. Run 
``
python subset.py
``
This might take a few minutes since 10k files are processed.

#### Extracting features

###### Get dependencies:
- python v. 2.7
- numpy
``
pip install numpy
``
- librosa v. 0.6.1
``
pip install librosa==0.6.1
``
- joblib v. 0.11
``
pip install joblib==0.11 --force-reinstall
``
- ffmpeg 
``
apt get ffmpeg
``
###### Create the following structure in the project's root directory:
├── csvfiles\
`` ``    ├── rhythm\
`` ``    ├── timbre\
`` ``    ├── harmony\
`` ``    └── melody

###### Run: 
``
python extract_features.py
``
###### Changes to original code:
- util/smoothiescore.py: Change operand *-* to *~* 
- extract_features.py: set *write_output* to *True*
- util/mfccs.py: Change *librosa.core.logamplitude* to *librosa.amplitude_to_db*
