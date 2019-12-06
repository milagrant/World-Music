# World-Music
### Project in DD2430 at KTH

#### All credist given to Maria Panteli et.al. This project is based on their original project which can be found [here](https://code.soundsoftware.ac.uk/projects/feature-space-world-music) 

### Data Set Analysis
Our analysis of the data set used in this project can be found in *all_metadata.xlsx*.

### Reproducing results

Bellow follow the steps taken to reproduce the results by Panteli et.al. 

#### Structuring the data
1. Make sure you have the full smithsonian dataset. 
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

###### Create lists of csv files:
This will create four text files in data/ pointing to the csv files containing the extracted features.

Run
``
python create_csvfiles_lists.py
``

#### Split data set and post-process

Run
``
python load_dataset.py
``

#### Train classifiers and evaluate them using test set
Not that this takes time, so make some coffee!

Run
``
python map_space.py
``

To print results, run:
``
python results.py
``

