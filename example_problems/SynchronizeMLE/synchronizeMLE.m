function optiproblem = synchronizeMLE(synchroproblem)
% Returns a Manopt synchronization problem based on maximum likelihood approach.
%
% function optiproblem = synchronizeMLE(synchroproblem)
%
% SynchronizeMLE uses optimization on a manifold via Manopt. The Manopt
% toolbox can be downloaded from http://www.manopt.org. You need to install
% it before using this function. The method used here is described in:
%
% N. Boumal, A. Singer and P.-A. Absil, 2013,
%   Robust estimation of rotations from relative measurements
%   by maximum likelihood,
% in the proceedings of the 52nd Conference on Decision and Control (CDC).
%
% A synchronization of rotations problem instance must be described using a
% problem structure (first input argument). To create this structure from
% your problem data, use the buildproblem function. See also main.m for an
% example.
%
% R0 is an initial guess for the rotations. If none is provided (or [] is
% provided) then the initialguess function will be used to generate one.
%
% The options structure will be passed to Manopt's trustregions algorithm.
% See the help for that solver for details.
%
% See also: buildproblem initialguess
%
% Nicolas Boumal, UCLouvain, Jan. 16, 2013.
    
    if ~exist('synchroproblem', 'var') || ~isstruct(synchroproblem)
        error('The first input parameter (synchroproblem) must be a structure. See buildproblem.');
    end

    % Extract the problem parameters and anchors
    n = synchroproblem.n;
    N = synchroproblem.N;
    A = synchroproblem.A;
    Ra = synchroproblem.Ra;

    % Obtain a Manopt description of the manifold: product of rotation
    % groups, with anchors indexed in A and given by Ra.
    manifold = anchoredrotationsfactory(n, N, A, Ra);
    
    optiproblem = struct();
    
    optiproblem.M = manifold;

    % Specify the cost function and its derivatives. Notice that we use the
    % store caching capability of Manopt to cut down on redundant
    % computations.
    optiproblem.cost = @mycost;
    function [f, store] = mycost(R, store)
        [f, store] = funcost(synchroproblem, R, store);
    end
    optiproblem.grad = @gradient;
    function [G, store] = gradient(R, store)
        [G, store] = fungrad(synchroproblem, R, store);
        store = incrementcounter(store, 'gradhesscalls');
    end
    optiproblem.hess = @hessian;
    function [H, store] = hessian(R, Omega, store)
        [H, store] = funhess(synchroproblem, R, Omega, store);
        store = incrementcounter(store, 'hesscalls');
        store = incrementcounter(store, 'gradhesscalls');
    end

end
