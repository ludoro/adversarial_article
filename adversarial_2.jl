eta = 1
maxiters = 100
λ = 0.03
#Trovo gli indici di tutti i 5 dai labels
indexs = findall(x->x==5, labels)
#x_goal è il primo 5 del dataset
x_start = X[:,indexs[1]]
function grad_descent_with_lambda_img_input(eta,maxiters,λ,x_start)
    y_goal = zeros(10,1)
    y_goal[4] = 1.0
    #Trucco veloce per avere il vettore che funziona per tracker.gradient senza
    #dover convertire da grayscale image a vettore di float
    xadv = zeros(N)
    xadv = xadv + x_start
    for i = 1:maxiters
        grads = Tracker.data(Tracker.gradient((xadv) -> loss(xadv, y_goal), xadv)[1])
        xadv -= eta* (grads + λ*(xadv - x_start))
    end
    return xadv

end
