N = 784
function loss(x,y_goal)
    C = 1/2*norm(y_goal-m(x),2)^2
    return C
end

eta = 1
maxiters = 2000
function grad_descent_no_lambda(eta,maxiters)
    N = 784
    y_goal = zeros(10,1)
    y_goal[3] = 1.0
    gaussian = Normal(.5,.1)
    #xadv di partenza è un un vettore di 784 elementi casuale
    xadv = rand(gaussian,N)
    for i = 1:maxiters
        grads = Tracker.data(Tracker.gradient((xadv) -> loss(xadv, y_goal), xadv)[1])
        xadv -= eta*grads
    end
    return xadv
end
