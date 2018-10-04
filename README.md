# Numerical experiments for Riemannian ARC project

This repository contains the Matlab codes to reproduce our numerical experiments comparing our implementation of Riemannian Adaptive Regularization with Cubics (Riemannian ARC) to other Manopt solvers.

The main script is compare_solvers_on_problems.m.

It is necessary to download Manopt (and to add it to Matlab's path) to run these experiments. Preferably obtain the latest version from https://github.com/NicolasBoumal/manopt. Please run with a version of Manopt dating from Oct. 4, 2018 or more recent.

The code pdf_print_code.m produces PDFs from Matlab figures. It relies on the utility pdfcrop being installed. If this is an issue, simply remove the call to pdf_print_code in the main script.

Authors:
Naman Agarwal, Nicolas Boumal, Brian Bullins, Coralia Cartis

Oct. 4, 2018
