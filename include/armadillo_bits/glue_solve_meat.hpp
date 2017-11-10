// Copyright 2008-2016 Conrad Sanderson (http://conradsanderson.id.au)
// Copyright 2008-2016 National ICT Australia (NICTA)
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ------------------------------------------------------------------------


//! \addtogroup glue_solve
//! @{



//
// glue_solve_gen


template<typename T1, typename T2>
inline
void
glue_solve_gen::apply(Mat<typename T1::elem_type>& out, const Glue<T1,T2,glue_solve_gen>& X)
  {
  arma_extra_debug_sigprint();
  
  const bool status = glue_solve_gen::apply( out, X.A, X.B, X.aux_uword );
  
  if(status == false)
    {
    arma_stop_runtime_error("solve(): solution not found");
    }
  }



template<typename eT, typename T1, typename T2>
inline
bool
glue_solve_gen::apply(Mat<eT>& out, const Base<eT,T1>& A_expr, const Base<eT,T2>& B_expr, const uword flags)
  {
  arma_extra_debug_sigprint();
  
  typedef typename get_pod_type<eT>::result T;
  
  const bool fast        = bool(flags & solve_opts::flag_fast       );
  const bool equilibrate = bool(flags & solve_opts::flag_equilibrate);
  const bool no_approx   = bool(flags & solve_opts::flag_no_approx  );
  const bool allow_band  = bool(flags & solve_opts::flag_allow_band );
  
  arma_extra_debug_print("glue_solve_gen::apply(): enabled flags:");
  
  if(fast       )  { arma_extra_debug_print("fast");        }
  if(equilibrate)  { arma_extra_debug_print("equilibrate"); }
  if(no_approx  )  { arma_extra_debug_print("no_approx");   }
  if(allow_band )  { arma_extra_debug_print("allow_band");  }
  
  T    rcond  = T(0);
  bool status = false;
  
  Mat<eT> A = A_expr.get_ref();
  
  if(A.n_rows == A.n_cols)
    {
    arma_extra_debug_print("glue_solve_gen::apply(): detected square system");
    
    uword KL = 0;
    uword KU = 0;
    
    // for development purposes only: determining the cost factor; need to take into account conversion from dense to band format
    const bool band = (allow_band) ? glue_solve_gen::get_band_counts(KL, KU, A) : false;
    
    ////const bool band = (glue_solve_band::get_band_counts(KL, KU, A)) ? ((KL*KU+uword(1)) < A.n_rows) : false;   // TODO: adjust cost factor
    
    if(fast)
      {
      if(equilibrate)  { arma_debug_warn("solve(): option 'equilibrate' ignored, as option 'fast' is enabled"); }
      
      if(true)
      //if(band == false)
        {
        arma_extra_debug_print("glue_solve_gen::apply(): fast + dense");
      
        status = auxlib::solve_square_fast(out, A, B_expr.get_ref());  // A is overwritten
        }
      else
        {
        arma_extra_debug_print("glue_solve_gen::apply(): fast + band");
        
        Mat<eT> AB;
        
        glue_solve_gen::get_band_format(AB, KL, KU, A, true);
        
        status = auxlib::solve_band_fast(out, AB, KL, KU, B_expr.get_ref());  // AB is overwritten
        }
      }
    else
      {
      if(band == false)
        {
        arma_extra_debug_print("glue_solve_gen::apply(): refine + dense");
        
        status = auxlib::solve_square_refine(out, rcond, A, B_expr, equilibrate);  // A is overwritten
        }
      else
        {
        arma_extra_debug_print("glue_solve_gen::apply(): refine + band");
        
        Mat<eT> AB;
        
        glue_solve_gen::get_band_format(AB, KL, KU, A, false);
        
        status = false;
        
        arma_check(true , "solve(): TODO: implement handling of 'refine + band'");  // TODO
        }
      }
    
    
    if( (status == false) && (no_approx == false) )
      {
      arma_extra_debug_print("glue_solve_gen::apply(): solving rank deficient system");
      
      if(rcond > T(0))
        {
        arma_debug_warn("solve(): system seems singular (rcond: ", rcond, "); attempting approx solution");
        }
      else
        {
        arma_debug_warn("solve(): system seems singular; attempting approx solution");
        }
      
      Mat<eT> AA = A_expr.get_ref();
      status = auxlib::solve_approx_svd(out, AA, B_expr.get_ref());  // AA is overwritten
      }
    }
  else
    {
    arma_extra_debug_print("glue_solve_gen::apply(): detected non-square system");
    
    if(equilibrate)  { arma_debug_warn( "solve(): option 'equilibrate' ignored for non-square matrix" ); }
    
    if(fast)
      {
      status = auxlib::solve_approx_fast(out, A, B_expr.get_ref());  // A is overwritten
      
      if(status == false)
        {
        Mat<eT> AA = A_expr.get_ref();
        
        status = auxlib::solve_approx_svd(out, AA, B_expr.get_ref());  // AA is overwritten
        }
      }
    else
      {
      status = auxlib::solve_approx_svd(out, A, B_expr.get_ref());  // A is overwritten
      }
    }
  
  
  if(status == false)  { out.soft_reset(); }
  
  return status;
  }



template<typename eT>
inline
bool
glue_solve_gen::get_band_counts(uword& out_KL, uword& out_KU, const Mat<eT>& A)
  {
  arma_extra_debug_sigprint();
  
  // NOTE: assuming that A has a square size
  
  const uword N = A.n_rows;
  
  if(N <= 4)  { return false; }
  
  // first, quickly check bottom-right and top-left corners
  
  const eT eT_zero = eT(0);
  
  const eT* A_col0 = A.memptr();
  const eT* A_col1 = A_col0 + N;
  
  if( (A_col0[N-2] != eT_zero) || (A_col0[N-1] != eT_zero) || (A_col1[N-1] != eT_zero) )  { return false; }
  
  const eT* A_colNm2 = A.colptr(N-2);
  const eT* A_colNm1 = A_colNm2 + N;
  
  if( (A_colNm2[0] != eT_zero) || (A_colNm1[0] != eT_zero) || (A_colNm1[1] != eT_zero) )  { return false; }
  
  
  // if we reached this point, go through the entire matrix to work out number of subdiagonals and superdiagonals
  
  uword KL = 0;  // number of   subdiagonals
  uword KU = 0;  // number of superdiagonals
  
  const uword nonzero_threshold = (N*N) / 4;
  
  for(uword col=0; col < N; ++col)
    {
    uword first_nonzero_row = col;
    uword  last_nonzero_row = col;
    
    const eT* A_colptr = A.colptr(col);
    
    for(uword row=0; row < col; ++row)
      {
      if( A_colptr[row] != eT_zero )  { first_nonzero_row = row; break; }
      }
    
    for(uword row=(col+1); row < N; ++row)
      {
      last_nonzero_row = (A_colptr[row] != eT_zero) ? row : last_nonzero_row;
      }
    
    const uword L_count = last_nonzero_row - col;
    const uword U_count = col - first_nonzero_row;
    
    if( (L_count > KL) || (U_count > KU) )
      {
      KL = L_count;
      KU = U_count;
      
      const uword nonzero = N*(1 + KL + KU) - (KL*(KL+1) + KU*(KU+1)) / 2;
      
      // return as soon as we know that it's not worth analysing the matrix any further
      
      if(nonzero > nonzero_threshold)  { return false; }
      }
    }
  
  out_KL = KL;
  out_KU = KU;
  
  return true;
  }



template<typename eT>
inline
void
glue_solve_gen::get_band_format(Mat<eT>& AB, const uword KL, const uword KU, const Mat<eT>& A, const bool use_offset)
  {
  arma_extra_debug_sigprint();
  
  // NOTE: assuming that A has a square size
  
  // band matrix storage format
  // http://www.netlib.org/lapack/lug/node124.html  
  
  // for ?gbsv,  matrix AB size: 2*KL+KU+1 x N; band representation of A stored in rows KL+1 to 2*KL+KU+1  (note: fortran counts from 1)
  // for ?gbsvx, matrix AB size:   KL+KU+1 x N; band representaiton of A stored in rows    1 to   KL+KU+1  (note: fortran counts from 1)
  //
  // the +1 in the above formulas is to take into account the main diagonal
  
  const uword AB_n_rows = (use_offset) ? uword(2*KL + KU + 1) : uword(KL + KU + 1);
  const uword N         = A.n_rows;
  
  AB.set_size(AB_n_rows, N);
  
  if(AB_n_rows == uword(1))
    {
    eT* AB_mem = AB.memptr();
    
    for(uword i=0; i<N; ++i)
      {
      AB_mem[i] = A.at(i,i);
      }
    }
  else
    {
    AB.zeros();  // paranoia
    
    for(uword j=0; j < N; ++j)
      {
      const uword A_row_start = (j > KU) ? uword(j - KU) : uword(0);
      const uword A_row_endp1 = (std::min)(N, j+KL+1);
      
      const uword length = A_row_endp1 - A_row_start;
      
      const uword AB_row_start = (KU > j) ? (KU - j) : uword(0);
      
      const eT*  A_colptr =  A.colptr(j) +  A_row_start;
            eT* AB_colptr = AB.colptr(j) + AB_row_start + ( (use_offset) ? KL : uword(0) );
      
      arrayops::copy( AB_colptr, A_colptr, length );
      }
    }
  }



//
// glue_solve_tri


template<typename T1, typename T2>
inline
void
glue_solve_tri::apply(Mat<typename T1::elem_type>& out, const Glue<T1,T2,glue_solve_tri>& X)
  {
  arma_extra_debug_sigprint();
  
  const bool status = glue_solve_tri::apply( out, X.A, X.B, X.aux_uword );
  
  if(status == false)
    {
    arma_stop_runtime_error("solve(): solution not found");
    }
  }



template<typename eT, typename T1, typename T2>
inline
bool
glue_solve_tri::apply(Mat<eT>& out, const Base<eT,T1>& A_expr, const Base<eT,T2>& B_expr, const uword flags)
  {
  arma_extra_debug_sigprint();
  
  const bool fast        = bool(flags & solve_opts::flag_fast       );
  const bool equilibrate = bool(flags & solve_opts::flag_equilibrate);
  const bool no_approx   = bool(flags & solve_opts::flag_no_approx  );
  const bool triu        = bool(flags & solve_opts::flag_triu       );
  const bool tril        = bool(flags & solve_opts::flag_tril       );
  
  arma_extra_debug_print("glue_solve_tri::apply(): enabled flags:");
  
  if(fast       )  { arma_extra_debug_print("fast");        }
  if(equilibrate)  { arma_extra_debug_print("equilibrate"); }
  if(no_approx  )  { arma_extra_debug_print("no_approx");   }
  if(triu       )  { arma_extra_debug_print("triu");        }
  if(tril       )  { arma_extra_debug_print("tril");        }
  
  bool status = false;
  
  if(equilibrate)  { arma_debug_warn("solve(): option 'equilibrate' ignored for triangular matrices"); }
  
  const unwrap_check<T1> U(A_expr.get_ref(), out);
  const Mat<eT>& A     = U.M;
  
  arma_debug_check( (A.is_square() == false), "solve(): matrix marked as triangular must be square sized" );
  
  const uword layout = (triu) ? uword(0) : uword(1);
  
  status = auxlib::solve_tri(out, A, B_expr.get_ref(), layout);  // A is not modified
  
  if( (status == false) && (no_approx == false) )
    {
    arma_extra_debug_print("glue_solve_tri::apply(): solving rank deficient system");
    
    arma_debug_warn("solve(): system seems singular; attempting approx solution");
    
    Mat<eT> triA = (triu) ? trimatu( A_expr.get_ref() ) : trimatl( A_expr.get_ref() );
    
    status = auxlib::solve_approx_svd(out, triA, B_expr.get_ref());  // triA is overwritten
    }
  
  
  if(status == false)  { out.soft_reset(); }
  
  return status;
  }



//! @}
