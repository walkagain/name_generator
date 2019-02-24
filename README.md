**[this code belongs to  the "GENERATING NAMES WITH A CHARACTER-LEVEL RNN".][https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html]** 

## Requirements

- Python 3
- pytorch 1.0.0

	

## Run

Print parameters:

```bash
./name_generator_rnn.py --help
```

```
optional arguments:
  -h, --help            show this help message and exit
  -it ITERATION, --iteration ITERATION
                        iterations of training
  -p PRINT_EVERY, --print_every PRINT_EVERY
                        print the training result every iterations
  -pl PLOT_EVERY, --plot_every PLOT_EVERY
                        plotting the loss every iterations
  -s SAVE_EVERY, --save_every SAVE_EVERY
                        save model params every iterations
  -tr, --train          Train the model with dataset
  -te, --test           test the saved model
  -lm LOAD_MODEL, --load_model LOAD_MODEL
                        load the saved
                        model(e.g.model/name_generator_model_100000.tar)
  -fn FILENAME, --filename FILENAME
                        dataset file for training (e.g.data/names/*.txt)
  -sl SINGLE_LETTER, --single_letter SINGLE_LETTER
                        generate name with a letter, e.g. -sl A
  -ml MULTI_LETTERS, --multi_letters MULTI_LETTERS
                        generate names with letters, e.g. -ml ACD
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate for training
   -c {Arabic,Chinese,Czech,Dutch,English,French,German,Greek,Irish,Italian,\
   Japanese,Korean,Polish,Portuguese,Russian,Scottish,Spanish,Vietnamese}, \
   --category {Arabic,Chinese,Czech,Dutch,English,French,German,Greek,Irish,Italian,\
   Japanese,Korean,Polish,Portuguese,Russian,Scottish,Spanish,Vietnamese}\
   language category to train or test

```

Train:

```bash
./name_generator_rnn.py --train
```

Evaluate:

```bash
./name_generator_rnn.py --test --load_model modelFile 
```


## References

- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
