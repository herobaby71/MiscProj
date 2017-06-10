import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.special import expit
import numpy as np

#input data
X = np.array(pd.read_csv('data_X.csv', sep=',', header= None))
y = np.array(pd.read_csv('data_y.csv', sep=',', header = None))
m = X.shape[0]
#check shape of input datas
print("X-shape:",X.shape)
print("y-shape:", y.shape)

#we are doing a multiclass classification so
#change y into array of numbers(10 is zero in this dataset)
y_matrix = np.zeros([m, 10])
print("y-matrix shape:", y_matrix.shape)
for i in range(m):
    y_matrix[i][y[i]-1] = 1

#Visualize the data (the first 20 image of X)
#randomly pick 100 images and display them
randno = X[np.random.choice(X.shape[0],100, replace=False), :]
display_array = None
display_row = np.reshape(randno[0],(20,20), order = 'F')
count = 1
while(count<100):
    #concatinate to the right
    while(not(count%10 ==0)):
        display_row = np.concatenate((display_row, np.reshape(randno[count], (20,20), order = 'F')), axis = 1)
        count+=1
    if(display_array is None):
        display_array = display_row
    else:
        display_array= np.concatenate((display_array, display_row), axis = 0)
    if (count == 100):
        break
    display_row = np.reshape(randno[count],(20,20), order = 'F')
    count+=1
plt.set_cmap("gray")
plt.imshow(display_array)
print("Program paused. waiting for continue...")
plt.show(block=False)



#create the sigmoid function
def sigmoid(z, derive= False):
    if(derive):
        return z*(1-z)
    return expit(z)
#gradient sigmoid (derivative)
def gradientsigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

#predict the digit given the input data
def predict(Theta1, Theta2, X):
    m = X.shape[0]
    n = Theta2.shape[0]
    pred = np.array([m, 1])
    h1= sigmoid(np.dot(np.append(np.ones([m,1]),X, axis = 1),Theta1.T))
    h2 = sigmoid(np.dot(np.append(np.ones([m,1]),h1, axis = 1),Theta2.T))
    pred = np.argmax(h2, axis=1)+1
    return pred
    

#so far so good. Let's implement backpropagation algorithm to learn our own weights
#inputs: X
#w/ label: y_matrix

#Cost Function (Use logistic function to calculate cost)
#k: number of attributes want to classify
def CostFunction(Thetas, input_layer_size, hidden_layer_size, k, X, y, lamb = 0):
    m = y.shape[0]
    #unrolling the parameters the '+1' because bias weights is included in Thetas
    Theta1 = np.reshape(Thetas[0:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1), order = 'F')
    #print(Theta1.shape)
    Theta2 = np.reshape(Thetas[hidden_layer_size*(input_layer_size+1):], (k,hidden_layer_size+1), order = 'F')
    #print(Theta2.shape)

    #forward Propagation
    a0 = np.column_stack((np.ones([m, 1]), X))
    z1 = sigmoid(np.dot(a0, Theta1.T))
    a1 = np.append(np.ones([m,1]),z1, axis = 1)
    a2 = sigmoid(np.dot(a1, Theta2.T))

    #Calculate cost
    #remove bias in the regularize term
    J = -(1/m)*(np.sum(y*np.log(a2) +(1-y)*np.log(1-a2))) + lamb/(2*m)*(np.sum(np.power(Theta1[:,1:], 2)) + np.sum(np.power(Theta2[:,1:],2)))


    #BackProb
    Delta1 = np.zeros(Theta1.shape)
    Delta2 = np.zeros(Theta2.shape)
    # for each training example
    for t in range(m):

        x = a0[t]
        a2 = sigmoid( np.dot(x,Theta1.T) )
        a2 = np.concatenate((np.array([1]), a2))
        a3 = sigmoid( np.dot(a2,Theta2.T) )
        delta3 = a3.ravel()-y[t]
        delta2 = (np.dot(Theta2[:,1:].T, delta3).T) * gradientsigmoid( np.dot(x, Theta1.T) )
        Delta1 += np.outer(delta2, x)
        Delta2 += np.outer(delta3, a2)

    Grad2 = (1/m)*Delta2
    Grad2[:, 2:]= Grad2[:,2:] + (lamb/m)*Theta2[:, 2:]
    Grad1 = (1/m)*Delta1
    Grad1[:, 2:]= Grad1[:,2:] + (lamb/m)*Theta1[:, 2:]

    ThetaGrad = np.append(Grad1.ravel(order= 'F'), Grad2.ravel(order = 'F'))
    
    return [J, ThetaGrad]



#Randomly initializing param Theta
def InitRandWeights(in_layer_size, out_layer_size):
    weights = np.random.random((out_layer_size, in_layer_size+1))*2-1
    return weights


#Check if our CostFunction is working correctly by applying advance optimization algorithm (BFGS)
input_layer_size = 400
hidden_layer_size = 25
lamb = 1
k = 10

#Initialize the weights
Theta1_init = InitRandWeights(input_layer_size,hidden_layer_size)
Theta2_init = InitRandWeights(hidden_layer_size, k)
Thetas_init = np.append(Theta1_init.ravel(order = 'F'),Theta2_init.ravel(order = 'F'))
#print (Thetas_init.shape)
arguments = (input_layer_size, hidden_layer_size, k, X, y_matrix, lamb)
results = optimize.minimize(CostFunction, x0=Thetas_init, args=arguments, options={'disp': True, 'maxiter':400}, method="L-BFGS-B", jac=True)
Trained_Thetas = results['x']
Trained_Theta1 = np.reshape(Trained_Thetas[0:hidden_layer_size*(input_layer_size+1)],(hidden_layer_size,input_layer_size+1), order = 'F')
Trained_Theta2 = np.reshape(Trained_Thetas[hidden_layer_size*(input_layer_size+1):], (k,hidden_layer_size+1), order = 'F')
pred= predict(Trained_Theta1,Trained_Theta2, X)
print("accuracy:", np.mean(list(map(int,np.equal(pred,y.ravel())))))
#99.5% is pretty good. Can try different lambda to get a better results
#output into file
#np.savetxt("Trained_Theta1.csv", Trained_Theta1, delimiter=",")
#np.savetxt("Trained_Theta2.csv", Trained_Theta2, delimiter=",")
