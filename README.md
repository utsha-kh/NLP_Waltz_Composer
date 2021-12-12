# NLP waltz composition
In this project, I used NLP techniques to generate new music resembling Chopin's waltz. 
I used GRU(Gated Recurrent Unit)-RNN based model to model original waltz musics as a language modeling,
then used the learned model to predict unseen music that inherits original styles.

## Installation
Create a virtual environment with python3.7.5. Then install requirements by "pip install -r requirements.txt".
Currently my code requires CUDA.

  virtualenv nlpwaltz -p 3.7.5
  
  source activate ./nlpwaltz/bin/activate
  
  pip install -r requirements.txt

## Acknowledgement
I reffered very little code for data preprocessing (e.g. creating dictionary mappings from data, function to get batch of data) from the repo https://github.com/AndySECP/Neural-Network-Music-Generation .

All the rest, including the model design, training, predictions, as well as tokenizing back and forth to MIDI format are my own work. 

The MIDI data of Chopin's waltz musics are obtained from https://www.kunstderfuge.com/chopin.htm . I am not publishing any of this project as it 
is used in a class project, and I respect all the copy rights.

In the future, I will play each Waltz by myself and record them to create my own MIDI dataset, which will be publishable.
