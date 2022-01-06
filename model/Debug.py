from sys import path


def print_check(func):
        def inner(self,labels_file, test_size,path):
            X_train, X_valid, Y_train, Y_valid = func(self,labels_file, test_size,path)
            print("\n+====================+")
            print("Training Samples: {0}\nValid Samples: {1} ".format(len(X_train), len(X_valid)))
            print("+====================+")
            return X_train, X_valid, Y_train, Y_valid
        return inner
