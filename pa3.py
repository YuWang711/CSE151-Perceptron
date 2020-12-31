import numpy as np
import math
import random

def readFile(filename):
    data_array = []
    #Read the file name that was passed in as paramter
    with open(filename,'r') as testing:
        #for each line store the array as pair [[0:783],784]
        for eachline in testing:
            split_line = eachline.split()
            data = []
            for i in range(0, len(split_line)-1):
                data.append(float(split_line[i]))
            label = float(split_line[len(split_line) - 1])
            data_array.append([data,label])
    return data_array

def readFeature(filename):
    data_array = []
    with open(filename,'r') as feature:
        for eachline in feature:
            data_array.append(eachline)
    return data_array

def findSubsetpos(data_set, label):
    return_data = []
    for data in data_set:
        if data[1] == label[0]:
            return_data.append([data[0],1])
        elif data[1] == label[1]:
            return_data.append([data[0],-1])

    return return_data

def findSubsetOVA(data_set, label):
    return_data = []
    for data in data_set:
        if data[1] == label:
            return_data.append([data[0],1])
        elif data[1] != label:
            return_data.append([data[0],-1])

    return return_data


def perceptron(train_set, epoch, test_set):
    weights = np.zeros(len(train_set[0][0]))
    for j in range(epoch):
        train_error = 0
        test_error = 0
        for train_data in train_set:
            if(train_data[1]*(np.dot(train_data[0], weights)) <= 0):
                weights = weights + np.dot(train_data[1],train_data[0])
            else:
                weights = weights 
        for train_data in train_set:
            if (np.dot(train_data[0],weights)*train_data[1] <= 0):
                train_error += 1
        for test_data in test_set:
            if (np.dot(test_data[0],weights)*test_data[1] <= 0):
                test_error += 1
        print("epoch :", j+1, " train error: ", train_error/len(train_set))
        print("epoch :", j+1, " test error: ", test_error/len(test_set))

    return weights

def avg_perceptron(train_set, epoch, test_set):
    weights = np.zeros(len(train_set[0][0]))
    average = weights
    cm = 1  
    for j in range(epoch):
        train_error = 0
        test_error = 0
        for train_data in train_set:
            if(train_data[1]*(np.dot(train_data[0], weights)) <= 0):
                average = average + (cm*weights)
                weights = weights + np.dot(train_data[1],train_data[0])
                cm = 1
            else:
                cm = cm + 1
        for train_data in train_set:
            predict = 0
            if(np.dot(average,train_data[0])> 0 ):
                predict = 1
            elif (np.dot(average,train_data[0])< 0 ):
                predict = -1
            if predict <= 0 and train_data[1] > 0:
                train_error += 1
            elif predict > 0 and train_data[1] < 0:
                train_error += 1
        for test_data in test_set:
            predict = 0
            if(np.dot(average,test_data[0])> 0 ):
                predict = 1
            elif (np.dot(average,test_data[0])< 0 ):
                predict = -1
            if predict <= 0 and test_data[1] > 0:
                test_error += 1
            elif predict > 0 and test_data[1] < 0:
                test_error += 1
        print("epoch :", j+1, " train error: ", train_error/len(train_set))
        print("epoch :", j+1, " test error: ", test_error/len(test_set))

    average = average + (cm*weights)
    return average

def voted_perceptron(train_set, epoch, test_set):
    weights = np.zeros(len(train_set[0][0]))
    weight_cm = []
    cm = 1
    for j in range(epoch):
        train_error = 0
        test_error = 0
        cm = 1
        for train_data in train_set:
            if(train_data[1]*(np.dot(train_data[0], weights)) <= 0):
                weight_cm.append([weights,cm])
                weights = weights + np.dot(train_data[1],train_data[0])
                cm = 1
            else:
                weights = weights 
                cm = cm + 1
        for train_data in train_set:
            predict = 0
            for w in weight_cm:
                if(np.dot(w[0],train_data[0])> 0 ):
                    predict += w[1]*1
                elif (np.dot(w[0],train_data[0])< 0 ):
                    predict += w[1]*-1
            if predict < 0 and train_data[1] > 0:
                train_error += 1
            elif predict > 0 and train_data[1] < 0:
                train_error += 1
        for test_data in test_set:
            predict = 0
            for w in weight_cm:
                if(np.dot(w[0],test_data[0])> 0 ):
                    predict += w[1]*1
                elif (np.dot(w[0],test_data[0])< 0 ):
                    predict += w[1]*-1
            if predict < 0 and test_data[1] > 0:
                test_error += 1
            elif predict > 0 and test_data[1] < 0:
                test_error += 1
        print("epoch :", j+1, " train error: ", train_error/len(train_set))
        print("epoch :", j+1, " test error: ", test_error/len(test_set))

    weight_cm.append([weights,cm])
    return weight_cm

def question_one(train_set, test_set, dictionary):
    label = [1.0,2.0]
    class_one_train = findSubsetpos(train_set, label)
    class_one_test = findSubsetpos(test_set, label)
    print("reg perceptron: ")
    weights = perceptron(class_one_train, 4, class_one_test)
    print("voted perceptron: ")
    weights_voted = voted_perceptron(class_one_train,4, class_one_test)
    print("average perceptron: ")
    weights_average = avg_perceptron(class_one_train,4,class_one_test)
    return

def question_two(train_set, test_set, dictionary):
    label = [1.0,2.0]
    class_one_train = findSubsetpos(train_set, label)
    class_one_test = findSubsetpos(test_set, label)
    weights_avg = avg_perceptron(class_one_train,3,class_one_test)
    arg_sorted_weight = np.argsort(weights_avg)

    print("Most positive three words :")
    print(dictionary[arg_sorted_weight[-1]], " weight : ", weights_avg[arg_sorted_weight[-1]])
    print(dictionary[arg_sorted_weight[-2]], " weight : ", weights_avg[arg_sorted_weight[-2]])
    print(dictionary[arg_sorted_weight[-3]], " weight : ", weights_avg[arg_sorted_weight[-3]])
    print(dictionary[arg_sorted_weight[0]], " weight : ", weights_avg[arg_sorted_weight[0]])
    print(dictionary[arg_sorted_weight[1]], " weight : ", weights_avg[arg_sorted_weight[1]])
    print(dictionary[arg_sorted_weight[2]], " weight : ", weights_avg[arg_sorted_weight[2]])

    return

def predict_one_data(weight, test_data,label):
    if(type(weight[0]) is list):
        prediction = 0
        for i in weight:
            if(np.dot(i[0],test_data) > 0):
                prediction += i[1]*1
            elif(np.dot(i[0],test_data) < 0):
                prediction += i[1]*-1
        if prediction > 0:
            return label
        elif prediction < 0:
            return -1
    else:
        if(np.dot(weight,test_data) >= 0):
            return label
        elif(np.dot(weight,test_data) < 0):
            return -1


def construct_Confusion_Matrix(weights, test_set):
    confusion_matrix = []
    for i in range(0,7):
        confusion_matrix.append([])
        for j in range(0,6):
            confusion_matrix[i].append(0)
    test_data_has_label = []
    for i in range(0,6):
        test_data_has_label.append(0)

    for test_data in test_set:
        predict_matrix = []
        test_data_has_label[int(test_data[1])-1] += 1
        for weight in weights:
            predict_matrix.append(predict_one_data(weight[0],test_data[0], weight[1]))
        if(len(np.unique(predict_matrix))==2):
            for i in range(len(predict_matrix)):
                if predict_matrix[i] != -1:
                    confusion_matrix[i][int(test_data[1]-1)] += 1
        else:
            confusion_matrix[6][int(test_data[1]-1)] += 1
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[i])):
            confusion_matrix[i][j] = confusion_matrix[i][j]/test_data_has_label[j]
    
    return confusion_matrix
    
def question_three(train_set, test_set, dictionary):
    
    classes = []
    for i in range(1,7):
        classes.append(i)

    print(classes)
    subsets = []

    for i in classes:
        subsets.append([findSubsetOVA(train_set,i),i])

    reg_weights = []
    # avg_weights = []
    # voted_weights = []
    for subset in subsets:
        print("class :", subset[1], " train error : ")
        print("Reg perceptron : ")
        reg_weights.append([perceptron(subset[0],1,test_set), subset[1]])
        # print("Average perceptron : ")
        # avg_weights.append([avg_perceptron(subset[0],1,test_set), subset[1]])
        # print("Voted perceptron : " )
        # voted_weights.append([voted_perceptron(subset[0],1,test_set),subset[1]])
    confusion_matrix_reg = []
    confusion_matrix_reg = construct_Confusion_Matrix(reg_weights, test_set)
    # confusion_matrix_avg = []
    # confusion_matrix_avg = construct_Confusion_Matrix(avg_weights, test_set)
    # confusion_matrix_vot = []
    # confusion_matrix_vot = construct_Confusion_Matrix(voted_weights, test_set)

    print("resulting confusion reg: ")
    k = 0
    for i in confusion_matrix_reg:
        if k < 6:
            k += 1
            print("classifier ", k)
        else:
            print("classifier don't know ")
        print(i)
    print("resulting confusion avg: ")
    # k = 0
    # for i in confusion_matrix_avg:
    #     if k < 6:
    #         k += 1
    #         print("classifier ", k)
    #     else:
    #         print("classifier don't know ")
    #     print(i)
    # print("resulting confusion voted: ")
    # k = 0
    # for i in confusion_matrix_vot:
    #     if k < 6:
    #         k += 1
    #         print("classifier ", k)
    #     else:
    #         print("classifier don't know ")
    #     print(i)
    print("done")
    return


def main():
    train_set = readFile('pa3train.txt')
    test_set = readFile('pa3_testing.txt')
    dictionary = readFeature('pa3_dictionary.txt')
    # print("Question one")
    # question_one(train_set, test_set, dictionary)
    print("Question two")
    question_two(train_set, test_set, dictionary)
    # print("Question three")
    # question_three(train_set, test_set, dictionary)

    # question_multi(train_set, test_set, dictionary)
    
    return


if __name__== "__main__" :
    main()