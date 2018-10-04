function optproblem = rtrmc(lrmcproblem, use_precon)
% Modified from RTRMC code to just return a Manopt optimization problem, so
% that we can run multiple solvers on it. use_precon is false by default.
    
    % Gather values from the problem description
    m = lrmcproblem.m;
    n = lrmcproblem.n;
    r = lrmcproblem.r;
    k = lrmcproblem.k;
	X = lrmcproblem.X;
    C = lrmcproblem.C;
    I = lrmcproblem.I;
    J = lrmcproblem.J;
    mask = lrmcproblem.mask;
	Chat = lrmcproblem.Chat;
	lambda = lrmcproblem.lambda;
    
    if m > n
        warning(['RTRMC is optimized for m <= n (matrices with more ' ...
                 'columns than rows). Please transpose the problem.']); %#ok<WNTAG>
    end
    
    % Obtain a description of the Grassmann manifold for Manopt
    grass = grassmannfactory(m, r);
    optproblem.M = grass;
    
    % Specify the cost and the gradient functions (see below)
    optproblem.cost = @cost;
    optproblem.grad = @gradient;
    optproblem.hess = @hessian;
    
    % Preconditioner for the Hessian.
    % Be careful: if W becomes rank-deficient or close, this will crash.
    % A rank-deficient W is the sign that r should be lowered. This is not
    % currently monitored in the algorithm, but could easily be.
    if nargin >= 2 && use_precon
        optproblem.precon = @precon;
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %% Define the cost function
    function [f, store] = cost(U, store)
        
        if ~isfield(store, 'val')
            
            [W, store] = lsqfit(lrmcproblem, U, store);
            store.W = W;
            
            UW = spmaskmult(U, W, I, J);
            store.UW = UW;
            
            store.val = ( .5*sum((C.*(UW-X)).^2)    ...
                        + .5*lambda.^2*sum(W(:).^2) ...
                        - .5*lambda.^2*sum(UW.^2)   ) / k;
        end
        f = store.val;
        
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Define the gradient function
    function [grad, store] = gradient(U, store)
        
        if ~isfield(store, 'grad')

            % If the cost was never computed for this point before, call it
            % so we have the necessary ingredients to compute the gradient.
            if ~isfield(store, 'val')
                [~, store] = cost(U, store);
            end
        
            W = store.W;
            WWt = W*W.';
            sqlaWWt = lambda.^2*WWt;
            store.WWt = WWt;
            store.sqlaWWt = sqlaWWt;

            UW = store.UW;
            RU = Chat.*(UW-X) - (lambda.^2)*X;
            store.RU = RU;
            
            store.grad = (multsparsefull(RU, W.', mask) + U*sqlaWWt)/k;
            
        end
        grad = store.grad;
        store = incrementcounter(store, 'gradhesscalls');
        
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Define the Hessian function
    function [hess, store] = hessian(U, H, store)
        
        [W, dW, store] = lsqfit(lrmcproblem, U, H, store);
        UdW = spmaskmult(U, dW, I, J);
        
        HW  = spmaskmult(H, W, I, J);
        
        if ~isfield(store, 'grad')
            [~, store] = gradient(U, store);
        end
        RU = store.RU;
        sqlaWWt = store.sqlaWWt;

        hess = multsparsefull(Chat.*(HW + UdW), W.', mask);
        hess = hess - U*(U.'*hess);
        hess = hess + multsparsefull(RU, dW.', mask);
        hess = hess + H*sqlaWWt;
        hess = hess + U*(lambda.^2*(W*dW.'));
        hess = hess/k;
        
        store = incrementcounter(store, 'hesscalls');
        store = incrementcounter(store, 'gradhesscalls');
            
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Define the preconditioner
    function [pH, store] = precon(U, H, store)
        
        if ~isfield(store, 'WWt')
            [W, store] = lsqfit(lrmcproblem, U, store);
            store.W = W;
            WWt = W*W.';
            store.WWt = WWt;
        end
        WWt = store.WWt;
        
        pH = H / (WWt/k);
        
    end


end
