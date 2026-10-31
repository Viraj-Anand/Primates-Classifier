# Primates-Classifier

A small convolutional neural network (CNN) that tells chimpanzees, gorillas and orangutans apart from photos. Built with TensorFlow/Keras in one notebook: [img_classification.ipynb](img_classification.ipynb).

## Data

- 291 photos in three folders: `chimpanzee`, `gorilla` and `orangutan`, collected by hand.
- The photos are not in this repo. To run the notebook, put your own images in `data/chimpanzee`, `data/gorilla` and `data/orangutan`.
- The notebook deletes any file that is not a jpg, jpeg, png or bmp, then resizes the images to 256 x 256 and scales the pixels to 0-1.
- Batches of 30. The first 7 batches (210 images) are for training, the next 2 (60 images) for validation and the last one (21 images) for testing.
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

9,238,239 parameters in total. Trained for 20 epochs with the Adam optimiser and sparse categorical cross-entropy.

## Results

- Validation accuracy reached 100% at epoch 6 and stayed at or very close to 100% from there. After epoch 20, training accuracy was 100% and validation accuracy was 100%, with a loss of 0.0012 on both.
- The notebook also shows the model's predictions on 10 test photos, which I checked by eye. I did not calculate an accuracy on the test images.

## Limitations

- It is a small dataset (291 photos), and the model is trained from scratch.
- The split takes batches from a dataset that is reshuffled on every pass, so training and validation images can mix between epochs. The 100% validation accuracy is probably too optimistic.
- Next steps: make one fixed split before training, measure accuracy on the test images, and try data augmentation.

## How to run

```
pip install -r requirements.txt
```

Use Python 3.12 or older. The notebook imports `imghdr`, which was removed in Python 3.13. I wrote it with Python 3.9. Open `img_classification.ipynb` in Jupyter or VS Code and run it from the top. A GPU helps but is not needed.

The last few cells of the notebook load a saved model from paths on my own machine. Skip them unless you have your own saved model.

## License

MIT, see [LICENSE](LICENSE).
