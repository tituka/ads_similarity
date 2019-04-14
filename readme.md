# Similarity engine readme

### REQUIREMENTS:
-Python3.6 (though 2.something works fine, but python3.6 replaced with python and some other stuff)

-virtualev (install with command `python pip install install virtualenv`))
Used 3rd party libraries can be found in the requirement.txt file for reference.

### SETUP
if 'demo' subfolder is found, all libraries are installed in the virtual environment of that name, which is activated with the terminal in the base directory with:
`source demo/bin/activate`

If the folder not is there or you don't want to use it, requirements can be installed like so:
`sudo cat tmp/requirements.txt | xargs -n 1 python3.6 -m pip install --user`
there's a bug that requires manuaaly installing the following:
`sudo apt-get install python3.6-tk`
Then open python interpreter with `python3.6`
then run the commands
`import nltk`
`nltk.download('stopwords')`
`nltk.download('punkt')`
`exit()`

_(https://pip.pypa.io/en/stable/reference/pip_install/#example-requirements-file) 
(first creating a new virtual environment to work in is a good idea: http://www.pythonforbeginners.com/basics/how-to-use-python-virtualenv/):_

### RUN UNIT TESTS:`
Run `python3.6 -m unittest discover`


### BUILD YOUR OWN MODEL:`
`python3.6 build_pickle.py -max_entries=1000 -filename='your_file_name_here`

(max_entries is the maximum number of unique searches the model will have. A timestamp will be added to the file name, in the pickles folder)

### TRY A MODEL:
Open python interpreter in the terminal with 
`python`

Then enter the following (line by line):
```python
gensim_models import KeyedVectors
import Company_system
word_vectors =KeyedVectors.load_word2vec_format('gensim_glove_vectors25.txt', binary=False)`
system=Company_system.Company_system(w_vectors=word_vectors, pickle_file='pickles/your_file_name_here1440_230618.pickle')`
```
With these in place, try:
To randomly pick some search and find similar ones:
``` model.random_similar()```
To find similar based on text you input:
``` model.similar_by_text('free money')```

### RUN FLASK APPLICATION:
_(This uses a hard-coded model, but this can be changed by editing the model specified by _system=..._ in the file tryy.py. If you do this, make sure the word_vectors file is the same one you used when building the model)_
Run locally: 
`python demo.py`

To serve application: https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-uwsgi-and-nginx-on-ubuntu-16-04 _(equivalent tutorials exist for different versions of Ubuntu)_

