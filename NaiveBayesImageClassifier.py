import numpy as np
import math

class naive_bayes_image_classifer:
    
    def __init__(self, NumOfClass:int = 10) -> None:
        '''
        Using Naive Bayes thereom to implement the image classification


        Attribute
        ---
        NumOfClass: number of class

        Function
        ---
        1. get_prior(label)
        2. get_condictional_prob(X, y, NumOfBin, Mode)
        3. test(condiction_prob, prior, X, y, NumOfBin, Mode)
        4. prob_visualize(prob, threshold)
        5. forward(X_train, y_train, X_test, y_test, NumOfBin, Mode)
        '''
        self.NumOfClass = NumOfClass

    def get_prior(self, label:int):
        '''
        Input
        ---
        label: (int) index of class

        Output
        ---
        prior: (float) prior probability of each class.
        '''
        prior = np.zeros(self.NumOfClass)
        for i in range(self.NumOfClass):
            sum = 0
            for j in range(len(label)):
                if label[j] == i:
                    sum += 1
            prior[i] = sum / len(label)
        return prior
    
    def get_conditional_prob(self, X:np.array, y:np.array, NumOfBin:int=1, Mode:str='discrete'):
        '''
        Input
        ---
        X: (ndarray) dataset, size(n, h * w, c) 
            n: number of images
            (h, w, c): (channel, height, width) of image size

        y: (ndarray) label of image

        NumOfBin: (int) the number of bin for only discrete mode

        Mode: (str) choose only discrete or continuous mode

        Output
        ---
        ConditionProb: (ndarray) the conditional probability of each label, size(NumOfClass, h, w, b)
        '''
        n, hxw, c = X.shape
        each_bin = 256 // NumOfBin

        

        if Mode == 'discrete':
            ContitionProb = np.zeros((self.NumOfClass, hxw, c, NumOfBin))
            for i in range(n):
                tmp = y[i]
                for j in range(hxw):
                    for k in range(c):
                        ContitionProb[tmp][j][k][int(X[i, j, k])//each_bin] += 1
            
            for i in range(self.NumOfClass):
                for j in range(hxw):
                    count = 0
                    for k in range(c):
                        for b in range(NumOfBin):
                            count += ContitionProb[i][j][k][b]
                        ContitionProb[i][j][k][:] /= count
        
        elif Mode == 'continuous':
            ContitionProb = np.zeros((self.NumOfClass, hxw, c, 256))

            for i in range(self.NumOfClass):
                tmp = X[y == i]
                for j in range(hxw):
                    for k in range(c):
                        mu = np.mean(tmp[:, j, k])
                        var = np.var(tmp[:, j, k])
                        if var == 0:
                            var = 10
                        for m in range(256):
                            ContitionProb[i, j, k, m] = ((1/math.sqrt(2 * math.pi * var))) * math.exp(-((m-mu) ** 2) / (2 * var))

        return ContitionProb


    def test(self, condiction_prob:np.array, prior:np.array, X:np.array, y:np.array, NumOfBin:int=1, Mode:str='discrete'):
        '''
        Input
        ---
        n: (int) the number of test dataset

        condiction_prob: (ndarray) the condictional probability

        X: (ndarray) dataset size(n, h * w, c) 
            n: number of images
            (h, w, c): (channel, height, width) of image size

        y: (ndarray) label of image
        '''
        error = 0

        n, hxw, c = X.shape
        b = 256 // NumOfBin


        for i in range(n):
            each_class_prob = np.zeros(self.NumOfClass)
            for j in range(self.NumOfClass):
                for k in range(hxw):
                    for m in range(c):
                        if Mode == 'discrete':
                            each_class_prob[j] += np.log(max(1e-4, condiction_prob[j, k, m, int(X[i, k, m])//b]))
                        elif Mode == 'continuous':
                            each_class_prob[j] += np.log(max(1e-30, condiction_prob[j, k, m, int(X[i, k, m])]))
                each_class_prob[j] += np.log(prior[j])

            each_class_prob /= np.sum(each_class_prob)

            print('Posterior (in log scale):')
            for j in range(self.NumOfClass):
                print("{}: {}".format(j, each_class_prob[j]))
            
            prediction = np.argmin(each_class_prob)

            print("Prediction: {}, Ans: {}".format(prediction, y[i]))

            if prediction != y[i]:
                error += 1
            error /= n

        print('Error rate: {}'.format(error))


    def prob_visualize(self, prob:float, threshold:float):

        print("Imagination of numbers in Bayesian classifier:")

        for i in range(self.NumOfClass):
            print("{}: ".format(i))
            for j in range(28):
                for k in range(28):
                    if np.argmax(prob[i, j * 28 + k]) >= threshold:
                        print("1", end=" ")
                    else:
                        print("0", end=" ")
                print()
            print()

    def forward(self, X_train:np.array, y_train:np.array, X_test:np.array, y_test:np.array, NumOfBin:int=1, Mode:str='discrete'):
        '''
        Input
        ---
        X_train: (ndarray) train dataset, size(n, h * w, c) 
            n: number of images
            (h, w, c): (channel, height, width) of image size

        y_train: (ndarray) label of train image

        X_test: (ndarray) test dataset, size(n, h * w, c) 
            n: number of images
            (h, w, c): (channel, height, width) of image size

        y_test: (ndarray) label of test image

        NumOfBin: (int) the number of bin for only discrete mode

        Mode: (str) choose only discrete or continuous mode
        '''
        Prior = self.get_prior(y_train)
        Condiction_prob = self.get_conditional_prob(X_train, y_train, NumOfBin, Mode)
        self.test(Condiction_prob, Prior, X_test, y_test, NumOfBin, Mode)
