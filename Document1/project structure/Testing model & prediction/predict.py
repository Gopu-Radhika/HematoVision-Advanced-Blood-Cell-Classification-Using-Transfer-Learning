pred = model.predict(test)
pred = np.argmax(pred, axis=1)  # Pick class with highest probability

# Reverse the class indices mapping
labels = (train.class_indices)
labels = dict((v, k) for k, v in labels.items())

# Convert numeric predictions to class labels
pred2 = [labels[k] for k in pred]
