function [optim_problem, X0] = SOn_regression()

%% Discrete regression curves on rotation group SO(n)
%
% Example script for code accompanying the paper:
%
% https://link.springer.com/chapter/10.1007/978-3-642-40020-9_37
%
% SO(n) is the set of orthogonal matrices of size n x n and determinant +1.
%
% This code requires Manopt, freely available at http://www.manopt.org.
%
% Nicolas Boumal, Oct. 2017

%% Define a regression problem by defining N control points on SO(n)

% Example 1: explicit construction
% n = 3;
% N = 4;
% p = zeros(n, n, 4);
% p(:, :, 1) = eye(n);
% p(:, :, 2) = [0 1 0 ; -1 0 0 ;  0  0  1];
% p(:, :, 3) = [0 0 1 ; -1 0 0 ;  0 -1  0];
% p(:, :, 4) = [0 0 1 ;  0 1 0 ; -1  0  0];

% Example 2: load from mat file
data = load('SOn_regression/controlpoints.mat');
n = data.n;
N = data.N;
p = data.p;

% For each control point, pick a weight (positive number). A larger value
% means the regression curve will pass closer to that control point.
w = ones(N, 1);

%% Define parameters of the discrete regression curve

% The curve has Nd points on SO(n)
Nd = 97;

% Each control point attracts one particular point of the regression curve.
% Specifically, control point k (in 1:N) attracts curve point s(k).
% The vector s of length N usually satsifies:
% s(1) = 1, s(end) = Nd and s(k+1) > s(k).
s = round(linspace(1, Nd, N));

% Time interval between two discretization points of the regression curve.
% This is only used to fix a scaling. It is useful in particular so that
% other parameter values such as w, lambda and mu (see below) have the same
% sense even when the discretization parameter Nd is changed.
delta_tau = 1/(Nd-1);

% Weight of the velocity regularization term (nonnegative). The larger it
% is, the more velocity along the discrete curve is penalized. A large
% value usually results in a shorter curve.
lambda = 0;

% Weight of the acceleration regularization term (nonnegative). The larger
% it is, the more acceleration along the discrete curve is penalized. A
% large value usually results is a 'straighter' curve (closer to a
% geodesic.)
mu = 1e-2;

%% Pack all data defining the regression problem in a problem structure.
regression_problem.n = n;
regression_problem.N = N;
regression_problem.Nd = Nd;
regression_problem.p = p;
regression_problem.s = s;
regression_problem.w = w;
regression_problem.delta_tau = delta_tau;
regression_problem.lambda = lambda;
regression_problem.mu = mu;

%% Call the optimization procedure to compute the regression curve.

% Compute an initial guess for the curve. If this step is omitted, digress
% (below) will compute one itself. X0 is a 3D matrix of size n x n x Nd,
% such that each slice X0(:, :, k) is a rotation matrix.
%
X0 = initguess(regression_problem);

% optim_problem is a Manopt optimization problem structure
%
optim_problem = digress(regression_problem);

optim_problem.name = sprintf('Discrete regression on SO(3)');

end

