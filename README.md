# Primates-Classifier

A small convolutional neural network (CNN) that tells chimpanzees, gorillas and orangutans apart from photos. Built with TensorFlow/Keras.

There are two parts:

- **The first experiment**: [img_classification.ipynb](img_classification.ipynb), the notebook I trained and ran.
- **A cleaner version of the pipeline**: [train.py](train.py) and [predict.py](predict.py), written afterwards to fix problems I found in the notebook (see Limitations).

## Data

- 291 photos in three folders: `chimpanzee`, `gorilla` and `orangutan`, collected by hand.
- The photos are not in this repo, and I no longer have them. To run anything here, put your own images in `data/chimpanzee`, `data/gorilla` and `data/orangutan`.
- Class numbers: 0 is chimpanzee, 1 is gorilla, 2 is orangutan.

## Model

| Layer | Details |
|---|---|
| Conv2D | 10 filters, 3 x 3, ReLU |
| AveragePooling2D | |
| Conv2D | 20 filters, 3 x 3, ReLU |
| AveragePooling2D | |
| Flatten | |
| Dense | 120 units, ReLU |
| Dense | 84 units, ReLU |
| Dense | 3 units, softmax |

9,238,239 parameters. Trained for 20 epochs with the Adam optimiser and sparse categorical cross-entropy. Photos are resized to 256 x 256 and scaled to 0-1.

## Results from the first experiment (the notebook)

- In the notebook, 7 batches of 30 photos (210) were used for training, 2 batches (60) for validation and the last batch (21) for testing.
- Validation accuracy reached 100% at epoch 6 and stayed at or very close to 100% from there. After epoch 20, training accuracy was 100% and validation accuracy was 100%, with a loss of 0.0012 on both.
- The notebook shows the model's predictions on 10 test photos, which I checked by eye. I did not calculate an accuracy on the test photos.

## Limitations

- It is a small dataset (291 photos), and the model is trained from scratch.
- In the notebook, the split takes batches from a dataset that is reshuffled on every pass, so training and validation photos can mix between epochs. The 100% validation accuracy is probably too optimistic.

## What `train.py` changes

- **One fixed split.** It splits each animal's photos once (about 70% train, 20% validation, 10% test, with a fixed seed), so no photo is ever in two sets.
- **A real test accuracy.** The test photos are used once, after training, and it prints the accuracy and a confusion matrix.
- **Augmentation.** Random flips, small turns and zoom, applied only while training.
- **Skips files that are not images**, and saves the trained model as one file.

**It has not been run on the original photos**, because I no longer have them, so there are no results for it yet. I checked that the code runs from start to finish on generated stand-in images (plain coloured pictures, not primates), which only shows that it works, not how accurate it would be on real photos.

## How to run

```
pip install -r requirements.txt
python train.py --data data --epochs 20
python predict.py some_photo.jpg --model primate_model.keras
```

`train.py` prints the split sizes, trains, then prints the test accuracy and saves `primate_model.keras`. `predict.py` prints which animal it thinks is in a photo, with its confidence.

I tested both scripts with Python 3.11, TensorFlow 2.21 and Keras 3.

To run the original notebook instead, open `img_classification.ipynb` in Jupyter or VS Code. It imports `imghdr`, which was removed in Python 3.13, so use Python 3.12 or older (I wrote it with 3.9). The last few cells of the notebook load a saved model from paths on my own machine; skip them.

## License

MIT, see [LICENSE](LICENSE).
