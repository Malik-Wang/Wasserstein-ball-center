# Wasserstein Ball Center: IPM Algorithm Implementation

## Project Overview

This project implements the algorithm proposed in the ICML25 paper "Finding Wasserstein Ball Center: Efficient Algorithm and The Applications in Fairness" for computing the Wasserstein ball center with fixed support. 

## Usage

### Basic Usage

```matlab
% Load data
load('testw0.mat'); % or other test data

% Prepare data structure
stride = m*ones(1,N); % Support size of each distribution
supp = repmat((1:m),1,N); % Support points

% Initial distribution
c0 = struct();
c0.w = ones(1,m)./m;    % Uniform weights
c0.supp = (1:m);        % Support points

% Set options
options.ipmouttolog = 1;         % Output log
options.ipmtol_primal_dual_gap = 5e-4; % Convergence tolerance
options.largem = 1;              % Large-scale problem setting

% Compute Wasserstein ball center
[c, iter, optval, t, fea] = IPM_WBC(stride, supp, w, c0, options);
```

### Parameter Description

- **stride**: Array of support sizes for each distribution
- **supp**: Matrix storing all support points
- **w**: Array of weights
- **c0**: Initial distribution
  - **w**: Weights
  - **supp**: Support points
- **options**: Algorithm options
  - **ipmouttolog**: Whether to output log
  - **ipmtol_primal_dual_gap**: Convergence tolerance
  - **largem**: Large-scale problem setting

### Return Values

- **c**: Computed Wasserstein ball center
- **iter**: Number of iterations
- **optval**: Optimal value
- **t**: Computation time
- **fea**: Feasibility error
