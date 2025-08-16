
%%%%%%%%%%%%%%%%%%%%%%%% SLRM Solve the first equation %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The first SLRM in MAAIPM
% (A * d * A)' * dp = t2

% The following 1 row: reshaping the vector diag(D) into a m x M matrix
d_pile = reshape(d(1:Mm), m, M);

% The following 2 rows: formulating the diagonal matrix B_1, the vector y, and the scalar c; (line 1 of Algorithm 1)
B_1_diag = sum(d_pile);
B_1_diag_cell = mat2cell(B_1_diag, 1, m_vec);
cc= sum(d(n_col-m+1-N-1:n_col-N-1));
y = d(n_col-m+2-N-1: n_col-N-1)/cc;

% DdE1' N分块对角 每个元素为m横向量
pile_cell = mat2cell(d_pile, m, m_vec);
D_pile = reshape(D', m, M);
D_pile_cell = mat2cell(D_pile, m, m_vec);
DE1 = cellfun(@(x,y) sum(x.*y,1), D_pile_cell, pile_cell, 'UniformOutput',0);

% dE1'()E1d m*N分块对角 每个元素为m*m矩阵
%pile_sum_cell = mat2cell(B_1_diag, ones(m*N,1), 1)';
%M1 = cellfun(@(x,y) x*x'./y, pile_cell, B1_diag_cell, 'UniformOutput',0);

% DdE1'() N分块对角 每个元素为m横向量
T2 = cellfun(@(x,y) x./y, DE1, B_1_diag_cell, 'UniformOutput',0);

% DdD' N对角
DD = cellfun(@(x,y) sum(sum(x.*x.*y)), D_pile_cell, mat2cell(d_pile, m, m_vec) ,'UniformOutput',0);

% II',JJ'
II = diag(d(Mm+m+1:Mm+m+N));
JJ = d(end) * ones(N);

% The following 3 rows: computing T = B_2^T B_1^(-1) in the cell formulation; 
d_pile(1,:) = [];
d_pile_cell = mat2cell(d_pile, m-1, m_vec);
T = cellfun(@(x,y) x./y, d_pile_cell, B_1_diag_cell, 'UniformOutput',0);

% The following 1 row: computing the diagonal blocks of matrix B_2 
B3_diag_cell= cellfun(@(x) sum(x,2), d_pile_cell,  'UniformOutput',0);

% The following 1 row: computing the diagonal blocks of matrix B_2 * B_3^(-1)
B2_B3inv_cell = cellfun(@(x,y) x'./y' , d_pile_cell,B3_diag_cell, 'UniformOutput',false );

% The following 2 rows: computing the blocks (B_1i - B_2i*B_3i^{-1}*B_2i^T)^{-1}; (i= 1, ..., N)
center_inv_cell = cellfun(@(x,y,z) diag(x')-y*z, B_1_diag_cell, B2_B3inv_cell, d_pile_cell,  'UniformOutput',0  );
center_inv_cell = cellfun(@inv, center_inv_cell , 'UniformOutput',false );

% The following 2 rows: computing inv_sum2 = \Sum_{i=1}^N B_3i^{-1};
B3_inv_diag_cell = cellfun(@(x) 1./x, B3_diag_cell,  'UniformOutput',0);

% The following 1 row1: computing inv_sum = \Sum_{i=1}^N A_ii^{-1};
A_1_inv_cell = cellfun(@(x,y,z) diag(x)+y'*z*y, B3_inv_diag_cell, B2_B3inv_cell, center_inv_cell, 'UniformOutput',0 );
inv_sum = sum( cat(3,A_1_inv_cell{:} ),3 );

% The following 2 rows: computing trN =  Y^(-1) + A_11^(-1) + ... + A_NN^(-1) ; 
BB = diag( d(n_col-m+2-N-1: n_col-N-1) ) - d(n_col-m+2-N-1: n_col-N-1)*d(n_col-m+2-N-1: n_col-N-1)'/cc;
trN = inv_sum + inv(BB);  
trN_inv = inv(trN);

% DdE2' N分块对角 元素为m-1横向量
% 横向排列
D_pile(1,:) = [];
D_pile_cell = mat2cell(D_pile, m-1, m_vec);
DE2 = cellfun(@(x,y) sum(x.*y,2)', D_pile_cell, d_pile_cell, 'UniformOutput',0);

% T4
% D...D^T
tmp_cell = cellfun(@(x,y,z) x-y*z', DE2 ,T2 ,d_pile_cell, 'UniformOutput', 0);  %D()E2'
tmp1_cell = cellfun(@(x,y) y*x*y', A_1_inv_cell, tmp_cell, 'UniformOutput', false);
tmp1 = cat(1,tmp1_cell{:});
tmp2_cell = cellfun(@(x,y) y*x, A_1_inv_cell, tmp_cell, 'UniformOutput', false);
tmp3_cell = cellfun(@(x) trN_inv*x', tmp2_cell, 'UniformOutput', false);
T4_1 = zeros(N);
for i=1:N
    for j=1:N
        ti = cell2mat(tmp2_cell(i));
        tj = cell2mat(tmp3_cell(j));
        T4_1(i,j) = ti * tj;
    end
end
T4_1 = diag(tmp1) - T4_1;
%D(I-M)D^T
T4_2_cell = cellfun(@(x,y,z) z-x./y*x', DE1, B_1_diag_cell, DD, 'UniformOutput',0);
T4_2 = diag(cat(1,T4_2_cell{:}));
T4 = T4_2 - T4_1 + II + JJ;

ys = kron( ones(N,1), y ); % to be used

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Start solving
xx=t2;

% The following 4 rows: computing xx = V_1 * xx ; 
tmp = xx(1 : M);
tmp_cell = mat2cell(tmp, m_vec, 1)';
tmp_cell = cellfun(@mtimes, T, tmp_cell, 'UniformOutput', false);
xx(M+1 : n_row-1-N) = xx(M+1 : n_row-1-N)  - cat(1,tmp_cell{:} );
tmp2_cell = cellfun(@mtimes, T2, mat2cell(tmp, m_vec, 1)', 'UniformOutput', false);
xx(n_row-N+1 : n_row) = xx(n_row-N+1 : n_row) - cat(1,tmp2_cell{:} );

%The following 1 row: computing  xx = V_2 * xx;   
xx(M+1: n_row-1-N) = xx(M+1: n_row-1-N) + kron( ones(N,1),y )*xx(n_row-N);

% xx = V_3 * xx
tmp3 = xx(M+1 : M+(m-1)*N);
tmp3_cell = mat2cell(tmp3, (m-1)*ones(N,1), 1)';
tmp3_cell = cellfun(@mtimes, A_1_inv_cell, tmp3_cell, 'UniformOutput', false);
tmp3 = cat(1,tmp3_cell{:});
tmp3_1 = tmp3;
tmp3_2 = zeros( N*(m-1),1 );
tmp3_2( 1:(N-1)*(m-1) )=  tmp3( 1:(N-1)*(m-1) );
tmp3_2( (N-1)*(m-1)+1: N*(m-1) ) = sum(reshape(tmp3, m-1, N), 2);
tmp3_2 = [zeros((N-1)*(m-1) ,1 ); trN_inv*tmp3_2((N-1)*(m-1)+1: N*(m-1))];
tmp3_2( 1: (N-1)*(m-1) ) = kron( ones(N-1,1), tmp3_2( (N-1)*(m-1)+1: N*(m-1) ) );
tmp3_2_cell = mat2cell(tmp3_2, (m-1)*ones(N,1),1 )';
tmp3_2_cell = cellfun(@mtimes, A_1_inv_cell, tmp3_2_cell, 'UniformOutput',false );
tmp3_2 = cat(1,tmp3_2_cell{:});
tmp3 = tmp3_1 - tmp3_2;
tmp3_3_cell = mat2cell(tmp3, (m-1)*ones(N,1), 1)';
tmp3_4_cell = cellfun(@(x,y,z,w) x*y-z*w'*y, DE2 ,tmp3_3_cell ,T2 ,d_pile_cell, 'UniformOutput',false );
xx(n_row-N+1 : n_row) = xx(n_row-N+1 : n_row) - cat(1,tmp3_4_cell{:});

% The following 1 row: computing  xx(1:M) = B_1^(-1) xx(1:M);  
xx(1:M) =  xx(1:M)./B_1_diag';

% The following 1 row: 
xx(n_row-N) = xx(n_row-N)/cc;

% T4
xx(n_row-N+1:n_row) = T4\xx(n_row-N+1 : n_row);

% The following 4 rows: computing xx(M+1: n_row-1) = A_1^(-1) * xx(M+1: n_row-1); 
tilde_x = xx(M+1: n_row-1-N);
tilde_x_cell = mat2cell(tilde_x, (m-1)*ones(N,1), 1)';
tilde_x_cell = cellfun(@mtimes, A_1_inv_cell, tilde_x_cell, 'UniformOutput', 0);
tilde_x = cat(1,tilde_x_cell{:});

tilde_x_1 = tilde_x;
% The following 3 rows: computing  tilde_x_2 = U^T * tilde_x; 
tilde_x_2 = zeros( N*(m-1),1 );
tilde_x_2( 1:(N-1)*(m-1) )=  tilde_x( 1:(N-1)*(m-1) );
tilde_x_2( (N-1)*(m-1)+1: N*(m-1) ) = sum(reshape(tilde_x, m-1, N), 2);

%The following 2 rows: 
tilde_x_2 = [zeros((N-1)*(m-1) ,1 ); trN_inv*tilde_x_2((N-1)*(m-1)+1: N*(m-1))];

%The following 1 row: computing  tilde_x_2 = U*tilde_x_2; 
tilde_x_2( 1: (N-1)*(m-1) ) = kron( ones(N-1,1), tilde_x_2( (N-1)*(m-1)+1: N*(m-1) ) );

%The following 3 rows: computing  tilde_x_2 = A_1^(-1) * tilde_x_2; 
tilde_x_2_cell = mat2cell(tilde_x_2, (m-1)*ones(N,1),1 )';
tilde_x_2_cell = cellfun(@mtimes, A_1_inv_cell, tilde_x_2_cell, 'UniformOutput',false );
tilde_x_2 = cat(1,tilde_x_2_cell{:});

%The following 1 row: 
tilde_x = tilde_x_1 - tilde_x_2;

xx(M+1: n_row-1-N) = tilde_x;

% V3^T * xx
tmp_cell = mat2cell(xx(n_row-N+1:n_row), ones(N,1), 1)';
tmp_cell = cellfun(@(x,y,z,w) x'.*y-w*z'.*y, DE2 ,tmp_cell ,T2 ,d_pile_cell, 'UniformOutput',false );
tmp2_cell = cellfun(@mtimes, A_1_inv_cell, tmp_cell, 'UniformOutput', false);
tmp2 = cat(1,tmp2_cell{:});
tmp2_1 = tmp2;
tmp2_2 = zeros(N*(m-1),1);
tmp2_2(1 : (N-1)*(m-1)) =  tmp2(1 : (N-1)*(m-1));
tmp2_2((N-1)*(m-1)+1 : N*(m-1)) = sum(reshape(tmp2, m-1, N), 2);
tmp2_2 = [zeros((N-1)*(m-1) ,1); trN_inv*tmp2_2((N-1)*(m-1)+1 : N*(m-1))];
tmp2_2(1: (N-1)*(m-1)) = kron(ones(N-1,1), tmp2_2((N-1)*(m-1)+1 : N*(m-1)));
tmp2_2_cell = mat2cell(tmp2_2, (m-1)*ones(N,1), 1)';
tmp2_2_cell = cellfun(@mtimes, A_1_inv_cell, tmp2_2_cell, 'UniformOutput', 0);
tmp2_2 = cat(1,tmp2_2_cell{:});
tmp2 = tmp2_1 - tmp2_2;
xx(M+1:n_row-1-N) = xx(M+1:n_row-1-N) - tmp2;

%The following 1 row: computing xx = V_2^T * xx ; 
xx(n_row-N) = xx(n_row-N) + dot(ys, xx(M+1 : n_row-1-N));

%The following 4 rows: computing xx = V_1^T * xx ; 
tmp_cell = mat2cell(xx(M+1 : n_row-1-N), (m-1)*ones(N,1),1 )';
tmp_cell = cellfun(@(x,y) x'*y, T, tmp_cell, 'UniformOutput', 0);
tmp = cat(1,tmp_cell{:});
xx(1:M) = xx(1:M) - tmp;
tmp2_cell = mat2cell(xx(n_row-N+1:n_row), ones(N,1), 1)';
tmp2_cell = cellfun(@(x,y) x' .* y, T2, tmp2_cell, 'UniformOutput', false);
tmp2 = cat(1, tmp2_cell{:});
xx(1:M) = xx(1:M) - tmp2;

dp = xx;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End solving %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
