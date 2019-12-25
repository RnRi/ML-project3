import numpy as np
import random
from graphs import *
from sklearn import preprocessing
from sklearn import decomposition
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

def main():
    print "Loading train output..."
    all_train_Y = np.load('train_outputs.npy')
    categories = all_train_Y
    # print len(categories)
    categories = categories[:40000]
    # print all_train_Y.shape
    print "Loading train input..."
    all_train_X = np.float32(np.load('train_inputs.npy'))
    # print all_train_X.shape
    # # N,M = all_train_X.shape
    train_set = all_train_X[:40000,:]
    valid_set = all_train_X[40000:49999,:]
    # print train_set[:4]

    # print 'Standardizing...'
    # scaler = preprocessing.StandardScaler().fit(train_set)
    # train_set = scaler.transform(train_set)
    # valid_set = scaler.transform(valid_set)
    # print train_set[:4]

    print 'Normalizing...'
    scaler = preprocessing.Normalizer().fit(all_train_X)
    train_set = scaler.transform(train_set)
    valid_set = scaler.transform(valid_set)
    # print train_set[:4]


    print 'predicting...'
    classifier = MLP(layers= [2304,10, 10], alpha=0.1, epochs=30)
    classifier.train(train_set, np.asarray(map(one_hot_vectorizer, categories)))
    predictions = map(classifier.predict, valid_set)
    write_test_output(predictions)


    test_outputs = []
    with open('validation_output.csv', 'rb') as csvfile:
       reader = csv.reader(csvfile, delimiter=',')
       next(reader, None)  # skip the header
       for test_output in reader:
           test_output_no_id = int(test_output[1])
           test_outputs.append(test_output_no_id)
    test_outputs_predict = np.asarray(test_outputs)

    all_train_Y = np.load('train_outputs.npy')
    Y_in_true = all_train_Y[40000:49999]

    num_correct = 0
    length = test_outputs_predict.shape[0]

    for line in range(length):
        if test_outputs_predict[line] == Y_in_true[line]:
            num_correct += 1

    print "Classified %d corrrectly out of %d for an accuracy of %f" %(num_correct, length, (float(num_correct)/length)*100)

    # Compute confusion matrix
    cm = confusion_matrix(test_outputs_predict, Y_in_true)
    cm = cm.astype('float')
    Row_Normalized = normalize(cm, norm='l2', axis=1)
    plt.matshow(Row_Normalized,cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

    # ROC  CURVE
    y_pred = test_outputs_predict
    y_score = Y_in_true - test_outputs_predict
    fpr, tpr, th = roc_curve(y_pred, y_score, pos_label=1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(fpr, tpr)
    plt.show()

def write_test_output(output_data):
    with open('validation_output.csv', 'wb') as fp:
        writer = csv.writer(fp, delimiter=',')
        writer.writerow(['Id', 'Prediction'])  # write header
        for i, category in enumerate(output_data):
            writer.writerow((str(i+1), category))


def step(x):
	return np.sign(x)

def sigmoid_one(x):
    return float(1.0 / (1.0 + np.exp(-x)))
sigmoid = np.vectorize(sigmoid_one)

def sigmoid_der(x):
    return sigmoid(x)*(1 - sigmoid(x))
sigmoid_prime = np.vectorize(sigmoid_der)


def randomize_params(alphas, n_layers, nodes_layer):
    alpha = random.choice(alphas)
    n_lay = random.choice(n_layers)
    layers = [48*48]
    for _ in xrange(n_lay):
        layers.append(random.choice(nodes_layer))
    layers.append(10)

    return alpha, layers

def one_hot_vectorizer(n):
	v = np.zeros(10)
	v[n] = 1
	return v


class MLP(object):
	def __init__(self,layers,alpha=0.01, epochs=10):
		self.activation = sigmoid
		self.activation_prime = sigmoid_prime
		self.alpha = alpha
		self.epochs = epochs
		self.weights = []

		for i in range(1, len(layers) - 1):
			r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) - 1
			self.weights.append(r)
		r = 2*np.random.random((layers[i] + 1, layers[i+1])) - 1
		self.weights.append(r)

	def train(self, examples, outputs):
		ones = np.atleast_2d(np.ones(examples.shape[0]))
		examples = np.concatenate((ones.T, examples),axis=1)

		for k in range(self.epochs):
			for i, ex in enumerate(examples):

				a = [ex]
				for l in range(len(self.weights)):
					dot_value = np.dot(a[l], self.weights[l])
					activation = self.activation(dot_value)
					a.append(activation)

				error = outputs[i] - a[-1]
				deltas = [error*self.activation_prime(a[-1])]
				for l in range(len(a) - 2, 0, -1):
					deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))
				deltas.reverse()

				for i in range(len(self.weights)):
					layer = np.atleast_2d(a[i])
					delta = np.atleast_2d(deltas[i])
					self.weights[i] += self.alpha * layer.T.dot(delta)
			print 'Epoch %d is Done!' % (k+1)

	def predict(self, example):
		a = np.concatenate((np.ones(1).T, np.array(example)))
		for l in range(0, len(self.weights)):
			a = self.activation(np.dot(a, self.weights[l]))
		return np.argmax(a)



if __name__=="__main__":
    main()