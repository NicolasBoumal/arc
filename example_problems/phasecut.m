function problem = phasecut(N, M)
% Random phase-retrieval problem with N unknowns and M measurements.

    A = randn(M, N) + 1i*randn(M, N);
    x = randn(N, 1) + 1i*randn(N, 1);
    b = abs(A*x);

    sigma = 1; % should be scaled with typical b
    b = max(0.01, b + sigma*randn(M, 1));

    [Q, ~] = qr(A, 0);
    P = eye(M) - Q*Q'; % = eye(M)-A*pinv(A)

    Mat = diag(b) * P * diag(b);
    
    p = ceil(sqrt(M));
    
    manifold = obliquecomplexfactory(p, M, true);
    problem.M = manifold;
    inner = @(A, B) real(A(:)'*B(:));
    % we could save on computation of M*Y for cost & grad, but most of the
    % time is spent in the Hessian calls so we wouldn't gain much.
    problem.cost  = @(Y) .5*inner(Mat*Y, Y)/M;
    problem.egrad = @(Y) (Mat*Y)/M;
    problem.ehess = @ehess;
    function [H, store] = ehess(Y, Ydot, store) %#ok<INUSL>
        H = (Mat*Ydot)/M;
        store = incrementcounter(store, 'hesscalls');
    end
    
%     if exist('b', 'var') && ~isempty(b)
%         problem.precon = @(Y, Ydot) bsxfun(@times, Ydot, M./b.^2);
%     end

    problem.name = sprintf('PhaseCut in Burer-Monteiro form, %d unknowns, %d measurements', N, M);
    
end
