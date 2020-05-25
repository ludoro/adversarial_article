using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using BSON: @save

imgs = MNIST.images()
labels = MNIST.labels()

#Stack images into one big X (1 batch)
X = hcat(float.(reshape.(imgs, :))...)

labels = MNIST.labels()

# One-hot-encode the labels, it means that I create a big 10x60000 matrix, every column corresponds to an image
# every column corresponds to a labelling of an image, 9 false and 1 true corresponding to label. (1->0, 2->1 , ... 10->9)
Y = onehotbatch(labels, 0:9)

# Neural Network: multi-layer-perceptron con relu e softmax,
#Avrei potuto anche scrivere:
#=
m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10,softmax))
=#
# Vado da R^784 -> R^32 con relu: relu(x) = max(0,x),
#da R^32 a R^10 con softmax: comprime il vettore di 32 dimensioni in uno di 10 con valori tra 0 e 1 che sommano ad 1 (probabilità)
m = Chain(
  Dense(28^2, 32, relu),
  Dense(32, 10),
  softmax)


#I define this loss function:
loss(x, y) = crossentropy(m(x), y)

#onecold is the opposite of onehot,
#faccio la media su quante volte ho azzeccato con il modello m rispetto al vero valore y
accuracy(x, y) = mean(onecold(m(x)) .== onecold(y))

#Rendo il data set più grande artificialmente
dataset = repeated((X, Y), 200)
evalcb = () -> @show(loss(X, Y))

opt = ADAM()

#callback ferma iterazione
Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10))
