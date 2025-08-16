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

%The following 2 rows: (line 5 and 6 of Algorithm 2)
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
tmp2_2 = [zeros((N-1)*(m-1) ,1); trN\tmp2_2((N-1)*(m-1)+1 : N*(m-1))];
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
