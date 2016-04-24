//[[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadilloExtensions/sample.h>
#include<RcppArmadillo.h>
#include<Rcpp.h>

using namespace Rcpp;
using namespace arma;


//==========================================================================================================
// functions for generate random number from truncated normal distribution
//==========================================================================================================


// norm_rs(a, b)
// generates a sample from a N(0,1) RV restricted to be in the interval
// (a,b) via rejection sampling.
// ======================================================================

// [[Rcpp::export]]

double norm_rs(double a, double b)
{
  double  x;
  x = Rf_rnorm(0.0, 1.0);
  while( (x < a) || (x > b) ) x = norm_rand();
  return x;
}

// half_norm_rs(a, b)
// generates a sample from a N(0,1) RV restricted to the interval
// (a,b) (with a > 0) using half normal rejection sampling.
// ======================================================================

// [[Rcpp::export]]

double half_norm_rs(double a, double b)
{
  double   x;
  x = fabs(norm_rand());
  while( (x<a) || (x>b) ) x = fabs(norm_rand());
  return x;
}

// unif_rs(a, b)
// generates a sample from a N(0,1) RV restricted to the interval
// (a,b) using uniform rejection sampling. 
// ======================================================================

// [[Rcpp::export]]

double unif_rs(double a, double b)
{
  double xstar, logphixstar, x, logu;
  
  // Find the argmax (b is always >= 0)
  // This works because we want to sample from N(0,1)
  if(a <= 0.0) xstar = 0.0;
  else xstar = a;
  logphixstar = R::dnorm(xstar, 0.0, 1.0, 1.0);
  
  x = R::runif(a, b);
  logu = log(R::runif(0.0, 1.0));
  while( logu > (R::dnorm(x, 0.0, 1.0,1.0) - logphixstar))
  {
    x = R::runif(a, b);
    logu = log(R::runif(0.0, 1.0));
  }
  return x;
}

// exp_rs(a, b)
// generates a sample from a N(0,1) RV restricted to the interval
// (a,b) using exponential rejection sampling.
// ======================================================================

// [[Rcpp::export]]

double exp_rs(double a, double b)
{
  double  z, u, rate;
  
  //  Rprintf("in exp_rs");
  rate = 1/a;
  //1/a
  
  // Generate a proposal on (0, b-a)
  z = R::rexp(rate);
  while(z > (b-a)) z = R::rexp(rate);
  u = R::runif(0.0, 1.0);
  
  while( log(u) > (-0.5*z*z))
  {
    z = R::rexp(rate);
    while(z > (b-a)) z = R::rexp(rate);
    u = R::runif(0.0,1.0);
  }
  return(z+a);
}




// rnorm_trunc( mu, sigma, lower, upper)
//
// generates one random normal RVs with mean 'mu' and standard
// deviation 'sigma', truncated to the interval (lower,upper), where
// lower can be -Inf and upper can be Inf.
//======================================================================

// [[Rcpp::export]]
double rnorm_trunc (double mu, double sigma, double lower, double upper)
{
  int change;
  double a, b;
  double logt1 = log(0.150), logt2 = log(2.18), t3 = 0.725;
  double z, tmp, lograt;
  
  change = 0;
  a = (lower - mu)/sigma;
  b = (upper - mu)/sigma;
  
  // First scenario
  if( (a == R_NegInf) || (b == R_PosInf))
  {
    if(a == R_NegInf)
    {
      change = 1;
      a = -b;
      b = R_PosInf;
    }
    
    // The two possibilities for this scenario
    if(a <= 0.45) z = norm_rs(a, b);
    else z = exp_rs(a, b);
    if(change) z = -z;
  }
  // Second scenario
  else if((a * b) <= 0.0)
  {
    // The two possibilities for this scenario
    if((R::dnorm(a, 0.0, 1.0,1.0) <= logt1) || (R::dnorm(b, 0.0, 1.0, 1.0) <= logt1))
    {
      z = norm_rs(a, b);
    }
    else z = unif_rs(a,b);
  }
  // Third scenario
  else
  {
    if(b < 0)
    {
      tmp = b; b = -a; a = -tmp; change = 1;
    }
    
    lograt = R::dnorm(a, 0.0, 1.0, 1.0) - R::dnorm(b, 0.0, 1.0, 1.0);
    if(lograt <= logt2) z = unif_rs(a,b);
    else if((lograt > logt1) && (a < t3)) z = half_norm_rs(a,b);
    else z = exp_rs(a,b);
    if(change) z = -z;
  }
  double output;
  output = sigma*z + mu;
  return (output);
}


//===============================================================================================
//===============================================================================================



//=====================================================================
// generate from multivariate normal
//=====================================================================


// multivariate normal

// [[Rcpp::export]]
mat mvrnorm(int n, colvec mu, mat sigma) {
  int ncols = sigma.n_cols;
  mat Y = randn(n, ncols);
  return (repmat(mu, 1, n).t() + Y * chol(sigma));
}



//===========================================================================
// generate from wishart and inverse wishart
//===========================================================================

// wishart and inverse wishart

//[[Rcpp::export]]
mat rwish(double nu, mat S) {
  //Rcpp::NumericVector nu(NU);
  //arma::mat S = as<arma::mat>(s);
  mat CC = chol(S);
  int n = S.n_cols;
  mat x = randn(n, n);
  
  for (int i = 0; i < n; i++) {
    x.diag()[i] = sqrt(R::rchisq(nu-i));
  }
  x = trimatu(x);
  x = x * CC;
  x = trans(x) * x;
  return x;
}



//[[Rcpp::export]]
mat riwish(double nu, mat S){
  int n = S.n_cols;
  mat X(n,n);
  X = inv(rwish(nu,inv(S)));
  return X;
}




//======================================================================
// generate from multivariate student t
//======================================================================


//multivariate student t

// [[Rcpp::export]]
mat mvrt(int n, colvec mu, mat sigma, double df){
  
  int m = mu.n_elem;
  colvec muz = zeros<colvec>(m); 
  mat Z = mvrnorm(n,muz,sigma);
  
  
  colvec u(n);
  
  for(int i=0; i<n; i++){
    u(i) = R::rchisq(df);
  }
  
  
  mat Y = repmat(mu, 1, n)+trans(Z)*diagmat(1/sqrt(u/df));
  
  return Y;
}


//=============================================================================
// draw from multivariate truncated normal with gibbs sampler (Geweke 1991)
//=============================================================================


//[[Rcpp::export]]

int mod(int a,int b){
  int r = a- floor(a/b)*b;
  return r;
}


// [[Rcpp::export]]
mat rtruncmvnorm(int n, colvec mu, mat Sigma, colvec lb, colvec ub,
                 double thin, double nburn){
  
  int p = mu.n_elem;
  
  colvec z = zeros<colvec>(p),
    x = zeros<colvec>(p);
  
  mat T = inv(Sigma);  
  
  colvec lalpha = lb-mu,
    ubeta = ub-mu;
  
  double h2_i, h_i;
  rowvec t_i, c_i;
  //colvec c_i;
  
  
  colvec z_ij; // z_ij is the z vector without i
  
  double a_i, b_i, eps_i, z_i;
  
  double niter = thin*n+nburn, idx=0;
  
  mat out(n,p);
  
  for(int iter=0; iter<niter; iter++){
    
    for(int i=0; i<p; i++){
      
      
      h2_i = 1/T(i,i);
      h_i = sqrt(h2_i);
      t_i = T.row(i);
      t_i.shed_col(i);
      c_i = -h2_i*t_i;
      
      z_ij = z;
      z_ij.shed_row(i);
      
      
      a_i = as_scalar((lalpha(i)-c_i*z_ij)/h_i);
      b_i = as_scalar((ubeta(i)-c_i*z_ij)/h_i);
      
      eps_i = rnorm_trunc(0,1,a_i,b_i);
      
      z_i = as_scalar(c_i*z_ij+h_i*eps_i);
      
      z(i) = z_i;
      
      x(i) = mu(i)+z_i;
      
    }
    
    if(iter>nburn-1 && mod(iter-(nburn-1),thin)==0){
      out.row(idx) = trans(x);
      idx +=1;
    }
  }
  
  return out;
}



//=========================================================================
// get upper/lower triangular without diagonal (get correlations vector)
//=========================================================================

// [[Rcpp::export]]

colvec corvec(mat V){
  int l = V.n_rows;
  colvec x(1);
  
  for(int j=0; j<l-1; j++){
    
    colvec temp = trans(V(j, span(j+1,l-1)));
    x = join_cols(x,temp);
    
  }
  
  x.shed_row(0);
  
  return x;
}


//=========================================================================
// get upper/lower triangular with diagonal (get var-covariance vector)
//=========================================================================

// [[Rcpp::export]]

colvec varvec(mat V){
  int l = V.n_rows;
  colvec x(1);
  
  for(int j=0; j<l; j++){
    
    colvec temp = trans(V(j, span(j,l-1)));
    x = join_cols(x,temp);
    
  }
  
  x.shed_row(0);
  
  return x;
}


//========================================================
// Real Root calculation
//========================================================

// [[Rcpp::export]]
NumericVector realRoot(double a, double b, double c){
  
  NumericVector out(2);
  
  double D = pow(b,2)-4*a*c;
  
  if(D<=0){
    cout << "Error: no real roots";
    return false;
  }
  
  out[0] = (-b-sqrt(D))/(2*a);
  out[1] = (-b+sqrt(D))/(2*a);
  
  return out; 
}



//=============================================================
// check positive definite matrix
//=============================================================


// [[Rcpp::export]]
bool ispd(mat X, double tol=1e-8){
  colvec eigenval = eig_sym(X);
  
  int n = X.n_rows;
  //double maxeig = max(eigenval);
  //if(!all(eigenval>=-tol*abs(maxeig))){
  //  return false;
  //}
  colvec abseig = abs(eigenval);
  
  for(int i=0;i<n;i++){
    if(abseig(i) < tol){
      eigenval(i) = 0;
    }
  }
  
  if(any(eigenval<=0)){
    return false;
  }else{
    return true;
  }
  
}


//=============================================================================
// Simulation Update
//=============================================================================


//================================================================
// log likelihood for y_i, i=1,...,n
//================================================================


//[[Rcpp::export]]

double yiloglik(colvec y_i, mat& X1_i, mat& X2_i, mat Z1_i, 
                mat Z2_i, colvec b1_i, colvec b2_i, 
                colvec alphadraw, colvec betadraw, double sig2y){
  
  int ni = y_i.n_elem;
  
  colvec mu1 = X1_i*alphadraw+Z1_i*b1_i;
  colvec mu2 = X2_i*betadraw+Z2_i*b2_i;
  
  
  double prob_1, prob_2; 
  
  colvec logyi(ni);
  //d_i(ni),
  
  for(int j=0; j<ni; j++){
    
    prob_1 = R::pnorm(mu1(j),0,1,1,0);
    
    if(y_i(j)>0){
      
      prob_2 = R::dlnorm(y_i(j),mu2(j),sqrt(sig2y),0);
      
      logyi(j) = log(prob_1)+log(prob_2);
      
    }else{
      
      logyi(j) = log(1-prob_1);
      
    }
    
  }
  
  return sum(logyi);
  
}



//==================================================================
// log-likelihood for all yi, i = 1,...,n
//==================================================================


//[[Rcpp::export]]
double yloglik(colvec& ID, colvec& y, mat& X1, mat& X2, 
               mat& Z1, mat& Z2, mat& b1, mat& b2,
               colvec& alphadraw, colvec& betadraw, double sig2y){
  
  
  //int n = y.n_elem;
  
  colvec subid = unique(ID);
  
  int nid = subid.n_elem;
  
  int id;
  
  colvec logy(nid);
  
  colvec y_i, b1_i, b2_i;
  
  mat X1_i, X2_i, Z1_i, Z2_i;
  
  
  for(int i=0; i<nid;i++){
    
    id = subid(i);
    
    y_i = y(find(ID==id));
    X1_i = X1.rows(find(ID==id));
    X2_i = X2.rows(find(ID==id));
    Z1_i = Z1.rows(find(ID==id));
    Z2_i = Z2.rows(find(ID==id));
    b1_i = trans(b1.rows(find(subid==id)));
    b2_i = trans(b2.rows(find(subid==id)));
    
    logy(i)= yiloglik(y_i, X1_i, X2_i, Z1_i, Z2_i, b1_i, 
         b2_i, alphadraw, betadraw, sig2y);
  } 
  
  return sum(logy);
  
}



//=================================================================
// Update latent variable w_i using multivariate probit
//=================================================================


//[[Rcpp::export]]

colvec wiUpdate(colvec y_i, mat& X1_i, mat& Z1_i, colvec b1_i,
                 colvec alphadraw, mat Rdraw){
  
  int ni = y_i.n_elem;
  
  colvec mu1 = X1_i*alphadraw+Z1_i*b1_i;
  
  //mat invR = inv(Rdraw);
  
  
  colvec lb = zeros<colvec>(ni),
         ub = zeros<colvec>(ni);
  
  
  for(int j=0; j<ni; j++){
    
    
    if(y_i(j)>0){
      
      ub(j) =  R_PosInf;
      
    }else{
      lb(j) = R_NegInf;
    }
  }
    
  colvec wdraw_i = vectorise(rtruncmvnorm(1,mu1,Rdraw,lb,ub,1,0));
  
  return wdraw_i;
}



//=================================================================
// Update latent variable w (all w_i)
//=================================================================



//[[Rcpp::export]]

colvec wUpdate(colvec& ID, colvec& y, mat& X1, mat& Z1, 
              mat& b1, colvec& alphadraw, mat Rdraw){
  
  int n = y.n_elem;
  
  colvec subid = unique(ID);
  
  int nid = subid.n_elem;
  
  colvec wdraw = zeros<colvec>(n);
  
  colvec y_i, b1_i;
  
  mat X1_i, Z1_i;
  
  int id;
  
  for(int i=0; i<nid; i++){
   
    id = subid(i);
    y_i = y(find(ID==id));
   // wdraw_i = wdraw(find(ID==id));
    X1_i = X1.rows(find(ID==id));
    Z1_i = Z1.rows(find(ID==id));
    b1_i = trans(b1.rows(find(subid==id)));
    
    wdraw.rows(find(ID==id)) = wiUpdate(y_i,X1_i,Z1_i,b1_i,alphadraw,Rdraw);
    
  } 
  
  return wdraw;
  
}



//===================================================
// Update alphadraw, (probit parameters)
//===================================================

//[[Rcpp::export]]

colvec alphaUpdate(colvec& ID, colvec& wdraw, mat& X1, mat& Z1, 
                   mat& b1, mat& Rdraw, colvec& A0, mat& sigA){
  
  colvec subid = unique(ID);
  
  int nid = subid.n_elem,
    p = X1.n_cols;
  
  mat invR = inv(Rdraw);
  
  mat Xsum = zeros<mat>(p,p);
  colvec Xwsum = zeros<colvec>(p);
  
  int id;
  colvec wdraw_i, b1_i, zb1i, resi;
  mat X1_i, Z1_i;
  
  for(int i=0;i<nid;i++){
    
    id = subid(i);
    
    wdraw_i = wdraw(find(ID==id));
    X1_i = X1.rows(find(ID==id));
    Z1_i = Z1.rows(find(ID==id));
    b1_i = trans(b1.rows(find(subid==id)));
    
    zb1i = Z1_i*b1_i;
    resi = wdraw_i-zb1i;
    
    Xsum += trans(X1_i)*invR*X1_i;
    Xwsum += trans(X1_i)*invR*resi;
    
  } 
  
  mat VA = inv(inv(sigA)+Xsum);
  
  colvec muA = VA*(inv(sigA)*A0+Xwsum);
  
  colvec alphadraw  = vectorise(mvrnorm(1, muA, VA));
  
  return alphadraw;
}



//========================================================================
// Update the probit Correlation matrix Rdraw, rho_12, rho_13, rho_23
//========================================================================

//=====================================================================
// compute the determine of R by change the rho_ij to -1,0,1
//=====================================================================

// [[Rcpp::export]]
double detR(mat& Rdraw, int rowi, int colj, int change){
  
  mat Rtemp = Rdraw;
  
  Rtemp(rowi,colj) = change;
  Rtemp(colj,rowi) = change;
  
  return (det(Rtemp));
}



//==========================================================
// Update each correlation element rho_ij
//==========================================================

// [[Rcpp::export]]
double rhoUpdate(mat Rdraw, double rhodraw_k, 
                 int rowi, int colj, double division_k){
  
  
  //int nrow = Rdraw.n_rows, ncol = Rdraw.n_cols;
  
  double d1 = detR(Rdraw, rowi, colj, 1),
    d2 = detR(Rdraw, rowi, colj,-1),
    d0 = detR(Rdraw, rowi, colj, 0);
  
  double a = 0.5*(d1+d2-2*d0), b = 0.5*(d1-d2), c = d0;
  
  NumericVector sol = realRoot(a,b,c);
  
  NumericVector diff = abs(rhodraw_k-sol);
  
  double mindiff = min(diff)/division_k;
  
  double rhonew_k = rhodraw_k + R::runif(-mindiff,mindiff);
  
  
  return rhonew_k;
  
}




//=============================================================
// posterior distribuion of R, with prior p(R)~1
//=============================================================


//[[Rcpp::export]]

double Rpost(colvec& ID, colvec& wdraw, mat& X1, mat& Z1, 
             mat& b1, colvec& alphadraw, mat& Rdraw, bool Log){
  
  colvec subid = unique(ID);
  
  int nid = subid.n_elem, k= Rdraw.n_rows;
  
  
  mat invR = inv(Rdraw);
  
  double empss = 0;
  
  int id;
  
  colvec wdraw_i, b1_i, mu1i, resi;
  mat X1_i, Z1_i;
  
  for(int i=0;i<nid;i++){
    
    id = subid(i);
    
    wdraw_i = wdraw(find(ID==id));
    X1_i = X1.rows(find(ID==id));
    Z1_i = Z1.rows(find(ID==id));
    b1_i = trans(b1.rows(find(subid==id)));
    
    mu1i = X1_i*alphadraw + Z1_i*b1_i;
    
    resi = wdraw_i-mu1i;
    
    empss += as_scalar(trans(resi)*invR*resi);
    
  } 
  
  double logw = as_scalar(-0.5*k*log(2*M_PI)-0.5*nid*log(det(Rdraw))-0.5*empss);
  
  if(Log){
    return logw;
  }else{
    return exp(logw);
  }
  
  
}



//=================================================================
// Update correlation matrix R by updating rho_ij seperately
//=================================================================

// [[Rcpp::export]]
List RUpdate1(colvec& ID, colvec& wdraw, mat& X1, mat& Z1, 
             mat& b1, colvec& alphadraw, mat& Rdraw, 
             colvec& rhodraw, mat& division, mat& rhocount){
  
  
  mat Rnew = Rdraw;
  
  double rhonew, Rnewpost, Rdrawpost, accprob;
  
  int k = Rdraw.n_rows, i, j;
  
  double u = log(R::runif(0,1));
  
  
  for(i =0;i<k-1;i++){
    
    for(j=i+1;j<k;j++){
      
      rhonew = rhoUpdate(Rdraw,Rdraw(i,j),i,j,division(i,j));
      
      Rnew(i,j) = rhonew;
      Rnew(j,i) = rhonew;
      Rnew.diag().ones();
      
      //cout<<Rnew<<endl;
      
      Rnewpost = Rpost(ID,wdraw,X1,Z1,b1,alphadraw,Rnew,1);
      Rdrawpost = Rpost(ID,wdraw,X1,Z1,b1,alphadraw,Rdraw,1);
      
      accprob = Rnewpost-Rdrawpost;
      
      if(u < accprob){
        //rhodraw = rhonew;
        Rdraw = Rnew;    
        rhocount(i,j) += 1;
      }
      // cout<< Rdraw<<endl;
    }
  }
  
  rhodraw = corvec(Rdraw);
  
  
  
  
  return List::create(Named("rhodraw")=rhodraw,
                      Named("Rdraw")=Rdraw,   
                      Named("rhocount")=rhocount);
  
}




// [[Rcpp::export]]
List RUpdate(colvec& ID, colvec& wdraw, mat& X1, mat& Z1, mat& b1, 
              colvec& alphadraw, mat& Rdraw, double division, double Rcount){
  
  
  mat Rnew = Rdraw;
  
  double rhonew, Rnewpost, Rdrawpost, accprob;
  
  int k = Rdraw.n_rows, i, j;
  
  double u = log(R::runif(0,1));
  
  
  for(i =0;i<k-1;i++){
    
    for(j=i+1;j<k;j++){
      
      rhonew = Rdraw(i,j)+R::rnorm(0, division);
      
      Rnew(i,j) = rhonew;
      Rnew(j,i) = rhonew;
    }
  }
      Rnew.diag().ones();
      
      //cout<<Rnew<<endl;
      if(ispd(Rnew)){
        Rnewpost = Rpost(ID,wdraw,X1,Z1,b1,alphadraw,Rnew,1);
      }else{
        Rnewpost = R_NegInf;
      }
        
      
      Rdrawpost = Rpost(ID,wdraw,X1,Z1,b1,alphadraw,Rdraw,1);
      
      accprob = Rnewpost-Rdrawpost;
      
      if(u < accprob){
        //rhodraw = rhonew;
        Rdraw = Rnew;    
        Rcount += 1;
      }
      // cout<< Rdraw<<endl;
  
  
  colvec rhodraw = corvec(Rdraw);

  return List::create(Named("rhodraw")=rhodraw,
                      Named("Rdraw")=Rdraw,   
                      Named("Rcount")=Rcount);
  
}






//=======================================================
// Update betadraw, lognormal parameters, y_ij>0
//=======================================================

//[[Rcpp::export]]

colvec betaUpdate(colvec& ID, colvec& y, mat& X2, mat& Z2, 
                  mat& b2, double sig2y, colvec B0, mat sigB){
  
  
  int n = y.n_elem;
  
  colvec subid = unique(ID);
  
  int nid = subid.n_elem;
  
  
  colvec zb2 = zeros<colvec>(n);
  
  mat Z2_i;
  colvec b2_i;
  
  int id;
  
  for(int i=0; i<nid;i++){
    
    id = subid(i);
    
    Z2_i = Z2.rows(find(ID==id));
    b2_i = trans(b2.rows(find(subid==id)));
    
    zb2.rows(find(ID==id)) = Z2_i*b2_i;
    
  } 
  
  colvec logysub = log(y.rows(find(y>0)));
  mat X2sub = X2.rows(find(y>0));
  colvec zb2sub = zb2.rows(find(y>0));
  
  mat VB = inv(inv(sigB)+trans(X2sub)*X2sub/sig2y);
  
  colvec muB = VB*(inv(sigB)*B0+trans(X2sub)*(logysub-zb2sub)/sig2y);
  
  // cout<<"muB "<<muB<<endl;
  
  colvec betadraw = vectorise(mvrnorm(1, muB, VB));
  
  return betadraw;
  
}


//=============================================================
// Update sig2y, lognormal model
//=============================================================

//[[Rcpp::export]]

double sig2yUpdate(colvec& ID, colvec& y, mat& X2, mat& Z2, mat& b2,
                   colvec& betadraw, double nu0, double tau0){
  
  int n = y.n_elem;
  
  colvec subid = unique(ID);
  
  int nid = subid.n_elem;
  
  colvec zb2 = zeros<colvec>(n);
  
  mat Z2_i;
  colvec b2_i;
  
  int id;
  
  for(int i=0; i<nid; i++){
    
    id = subid(i);
    
    Z2_i = Z2.rows(find(ID==id));
    b2_i = trans(b2.rows(find(subid==id)));
    
    zb2.rows(find(ID==id)) = Z2_i*b2_i;
    
  } 
  
  colvec logysub = log(y.rows(find(y>0)));
  mat X2sub = X2.rows(find(y>0));
  colvec zb2sub = zb2.rows(find(y>0));
  
  // cout<<"zbsub "<<zb2sub<<endl;
  
  int nsub = logysub.n_elem;
  
  colvec res = logysub-X2sub*betadraw-zb2sub;
  
  double nu1 = 0.5*nsub+nu0;
  
  double tau1 = as_scalar(0.5*trans(res)*res+tau0);
  
  double sig2ydraw = 1/(R::rgamma(nu1,1/tau1));
  
  
  return sig2ydraw;
  
}



//=========================================================
// Update Sigb, variance-covariance matrix for (b1,b2)
//=========================================================

//[[Rcpp::export]]
mat SigbUpdate(int nid, mat& b1, mat& b2, double nub, mat& Sb){
  
  mat b = join_rows(b1,b2);
  
  double nub1 = nid+nub;
  
  mat Phib = trans(b)*b+Sb;
  
  mat Sigbdraw = riwish(nub1, Phib);
  
  return Sigbdraw;
}



//===========================================================
//  Update bi=c(b1i,b2i), using M-H step
//===========================================================

//======================
// bi likelihood
//======================

//[[Rcpp::export]]
double biloglik(colvec b1_i, colvec b2_i, mat Sigbdraw){
  
  colvec b_i = join_cols(b1_i,b2_i);
  
  int k = Sigbdraw.n_rows;
  
  mat invSigb = inv(Sigbdraw);
  
  double logbi = as_scalar(-0.5*k*log(2*M_PI)-0.5*log(det(Sigbdraw))-0.5*trans(b_i)*invSigb*b_i);
  
  return logbi;
  
}


//[[Rcpp::export]]
double bloglik(mat& b1, mat& b2, mat Sigbdraw){
  
  int nid = b1.n_rows;
  
  double logb = 0;
  colvec b1_i, b2_i;
  
  for(int i=0; i<nid; i++){
    
    b1_i = trans(b1.row(i));
    b2_i = trans(b2.row(i));
    
    logb += biloglik(b1_i, b2_i, Sigbdraw);
  }
  
  return logb;
}




//=================================
// bi posterior
//=================================

//[[Rcpp::export]]

double bilogpost(colvec y_i, mat& X1_i, mat& X2_i, mat Z1_i, mat Z2_i,
                  colvec b1_i, colvec b2_i, colvec alphadraw, 
                  colvec betadraw, double sig2y, mat Sigbdraw){
  
  double logyi, logbi, logbipost;
  
  logyi = yiloglik(y_i,X1_i,X2_i,Z1_i,Z2_i,b1_i,
                    b2_i,alphadraw,betadraw,sig2y);
  
  logbi = biloglik(b1_i,b2_i,Sigbdraw);
  
  logbipost = logyi+logbi;
  
  return logbipost;
  
}



//====================
// bi Update
//====================

//[[Rcpp::export]]

List biUpdate(colvec y_i, mat& X1_i, mat& X2_i, mat Z1_i, 
               mat Z2_i, colvec b1_i, colvec b2_i,
               colvec alphadraw, colvec betadraw, 
               double sig2y, mat Sigbdraw, double delta, 
               mat propSigb, double df, colvec bicount){
  
  colvec bidraw = join_cols(b1_i,b2_i);
  
  int k1 = b1_i.n_elem, k = bidraw.n_elem;
  
  mat propSb = delta*propSigb;
  
  colvec binew = vectorise(mvrt(1,bidraw,propSb, df));
  
  colvec b1inew = binew.subvec(0,k1-1),
    b2inew = binew.subvec(k1,k-1);
  
  double logbipostnew, logbipostdraw, logu, accprob;
  
  logu = log(R::runif(0,1));
  
  logbipostnew = bilogpost(y_i,X1_i,X2_i,Z1_i,Z2_i,b1inew,b2inew,
                            alphadraw,betadraw,sig2y,Sigbdraw);
  
  
  logbipostdraw = bilogpost(y_i,X1_i,X2_i,Z1_i,Z2_i,b1_i,b2_i,
                             alphadraw,betadraw,sig2y,Sigbdraw);
  
  accprob = logbipostnew-logbipostdraw;
  
  
  if(logu<accprob){
    
    b1_i = b1inew;
    b2_i = b2inew;
    bicount += 1;
    
  }
  
  return List::create(Named("b1i")=b1_i,
                      Named("b2i")=b2_i,
                      Named("bicount")=bicount);
  
}




//===============================
// Update latent variable b
//===============================

//[[Rcpp::export]]

List bUpdate(colvec& ID, colvec& y, mat& X1, mat& X2, 
              mat& Z1, mat& Z2, mat& b1, mat& b2,
              colvec alphadraw, colvec betadraw, double sig2y,
              mat Sigbdraw, double delta, mat propSigb, 
              double df, colvec& bcount){
  
  
  colvec subid = unique(ID);
  
  int nid = subid.n_elem;
  //int q1 = b1.n_cols, q2 = b2.n_cols;
  
  //cout <<"nid "<<nid
  
  int id;
  
  
  mat X1_i, X2_i, Z1_i, Z2_i;
  colvec y_i, b1_i, b2_i;
  colvec bicount;
  
  //double delta_i;
  
  
  
  //mat b1mat = zeros<mat>(n,q1), b2mat = zeros<mat>(n,q2);
  
  List bout;
  
  for(int i=0; i<nid; i++){
    
    id = subid(i);
    
    y_i = y.rows(find(ID==id));
    X1_i = X1.rows(find(ID==id));
    X2_i = X2.rows(find(ID==id));
    Z1_i = Z1.rows(find(ID==id));
    Z2_i = Z2.rows(find(ID==id));   
    b1_i = trans(b1.rows(find(subid==id)));
    b2_i = trans(b2.rows(find(subid==id)));
    
    bicount = bcount.rows(find(subid==id));
    
    
    bout = biUpdate(y_i,X1_i,X2_i,Z1_i,Z2_i,b1_i,b2_i,
                     alphadraw,betadraw,sig2y,Sigbdraw,
                     delta,propSigb,df,bicount);
    
    b1_i = as<colvec>(bout["b1i"]);
    b2_i = as<colvec>(bout["b2i"]);
    bicount = as<colvec>(bout["bicount"]);
    
    b1.rows(find(subid==id)) = trans(b1_i);  
    b2.rows(find(subid==id)) = trans(b2_i);
    bcount.rows(find(subid==id)) = bicount;
    
    
  } 
  
  return List::create(Named("b1")=b1,
                      Named("b2")=b2,
                      Named("bcount")=bcount);
}





//=============================================================
// full likelihood
//=============================================================

//[[Rcpp::export]]
double fullloglik(colvec& ID, colvec& y, mat& X1, mat& X2, 
                   mat& Z1, mat& Z2, mat& b1, mat& b2, 
                   colvec alphadraw, 
                   colvec betadraw, double sig2y, mat Sigbdraw,
                   colvec& A0, mat& sigA,
                   colvec& B0, mat& sigB, 
                   double nu0, double tau0, 
                   double nub, mat Sb){
  //colvec rhodraw,
  //colvec& rho0, mat& Sigrho0,  
  double logy, logb, logalpha, logbeta, logsig2y, logsigb; //logrho,
  
  
  logy = yloglik(ID,y,X1,X2,Z1, Z2, b1, b2,
                  alphadraw,betadraw,sig2y);
  
  logb = bloglik(b1,b2,Sigbdraw);
  
  // prior likelihood
  int ka = sigA.n_rows, kb = sigB.n_rows, psigb = Sigbdraw.n_rows;
  
  
  logalpha = as_scalar(-0.5*ka*log(2*M_PI)-0.5*log(det(sigA))-0.5*trans(alphadraw-A0)*inv(sigA)*(alphadraw-A0));
  
  //logrho = logrhoprior(rhodraw, rho0, Sigrho0, true);
  
  logbeta = as_scalar(-0.5*kb*log(2*M_PI)-0.5*log(det(sigB))-0.5*trans(betadraw-B0)*inv(sigB)*(betadraw-B0));
  
  logsig2y = as_scalar(nu0*log(tau0)-log(Rf_gammafn(nu0))-(nu0+1)*log(sig2y)-tau0/sig2y);
  
  
  logsigb = as_scalar(0.5*nub*log(det(Sb))-0.5*nub*psigb*log(2)
                      -log(pow(M_PI,0.5)*R::gammafn(nub/2)*R::gammafn(nub/2-0.5))
                      -0.5*(nub+psigb+1)*log(det(Sigbdraw))-0.5*trace(Sb*inv(Sigbdraw)));
                        
                        
  double floglik = logy+logb+logalpha+logbeta+logsig2y+logsigb;
                        
  return floglik;
                        
}





//==================================================================================
// MCMC Update for two-part model with MVP, no variable selection
//===================================================================================

//[[Rcpp::export]]
List cppmcmcUpdate(colvec& ID, colvec& y, mat& X1, mat& X2, mat& Z1, mat& Z2, 
                   colvec& A0, mat& sigA, colvec& B0, mat& sigB, 
                   double nu0, double tau0, double nub, mat& Sb, 
                   double delta, mat& propSigb, double df, 
                   double division, double niter){
  
  
  // y,X1,X2,Z1,Z2,b1,b2 need to be sorted by ID
  
  colvec subid = unique(ID);
  
  int n = y.n_elem, nid = subid.n_elem,
    p = X1.n_cols, q = X2.n_cols,
    pz = Z1.n_cols, qz = Z2.n_cols, nb = pz+qz,
    k = n/nid, nrho = k*(k-1)/2;
  
  colvec wdraw, alphadraw, betadraw, rhodraw;
  mat Sigbdraw, b1, b2, Rdraw;
  double sig2y, deviance;
  
  //initial values 
  wdraw = randn<colvec>(n);
  alphadraw = zeros<colvec>(p);
  betadraw = zeros<colvec>(q);
  b1 = randn<mat>(nid,pz);
  b2 = randn<mat>(nid,qz);
  sig2y = 1;
  
  Sigbdraw = zeros<mat>(nb,nb);
  Sigbdraw.diag().ones();
  
  Rdraw = zeros<mat>(k,k);
  Rdraw.diag().ones();
  rhodraw = corvec(Rdraw);
  
 // colvec Sigbvec = vectorise(Sigbdraw);
  colvec Sigbvec = varvec(Sigbdraw);
  int nsigb = Sigbvec.n_elem;
  
  double Rcount = 0;
  //mat rhocount = zeros<mat>(k,k);
  colvec bcount = zeros<colvec>(nid);
  
  List Rout, bout;
  // results matrix
  
  mat Wmat = zeros<mat>(niter,n),
    alphamat = zeros<mat>(niter,p),
    betamat = zeros<mat>(niter,q),
    Sigbmat = zeros<mat>(niter,nsigb),
    //b1mat = zeros<mat>(niter,nid),
    //b2mat = zeros<mat>(niter,nid),
    rhomat = zeros<mat>(niter,nrho);
  
  cube b1mat = zeros<cube>(nid,pz,niter),
       b2mat = zeros<cube>(nid,qz,niter);
  
  colvec sig2ymat = zeros<colvec>(niter),
    devmat = zeros<colvec>(niter);
  
  for(int iter=0; iter<niter; iter++){
    
    //wdraw = wUpdate(ID,y,wdraw,X1,Z1,b1,alphadraw,Rdraw);
    wdraw = wUpdate(ID,y,X1,Z1,b1,alphadraw,Rdraw);
    Wmat.row(iter) = trans(wdraw);
    
    alphadraw = alphaUpdate(ID,wdraw,X1,Z1,b1,Rdraw,A0,sigA);
    alphamat.row(iter) = trans(alphadraw);
    
    Rout = RUpdate(ID,wdraw,X1,Z1,b1,alphadraw,Rdraw,division,Rcount); 
    
    Rdraw = as<mat>(Rout["Rdraw"]);
    rhodraw = as<colvec>(Rout["rhodraw"]);
    Rcount = as<double>(Rout["Rcount"]);
    rhomat.row(iter) = trans(rhodraw);
    
    betadraw = betaUpdate(ID,y,X2,Z2,b2,sig2y,B0,sigB);
    betamat.row(iter) = trans(betadraw);
    
    sig2y = sig2yUpdate(ID,y,X2,Z2,b2,betadraw,nu0,tau0);
    sig2ymat(iter) = sig2y;
    
    Sigbdraw = SigbUpdate(nid,b1,b2,nub,Sb);
    Sigbvec = varvec(Sigbdraw);
    Sigbmat.row(iter) = trans(Sigbvec);
    
    bout = bUpdate(ID,y,X1,X2,Z1,Z2,b1,b2,alphadraw,betadraw,
                   sig2y,Sigbdraw,delta,propSigb, df,bcount);
    
    b1 = as<mat>(bout["b1"]);
    b2 = as<mat>(bout["b2"]);
    bcount = as<colvec>(bout["bcount"]);
    b1mat.slice(iter) = b1;
    b2mat.slice(iter) = b2;
    
    deviance = -2*fullloglik(ID,y,X1,X2,Z1,Z2,b1,b2,alphadraw,
                          betadraw,sig2y,Sigbdraw,A0,sigA, 
                          B0,sigB,nu0,tau0,nub,Sb);
    devmat(iter) = deviance;
    
  }
  
  return List::create(Named("Wmat")=Wmat,
                      Named("alphamat")=alphamat,
                      Named("betamat")=betamat,
                      Named("sig2ymat")=sig2ymat,
                      Named("Sigbmat")=Sigbmat,
                      Named("b1mat")=b1mat,
                      Named("b2mat")=b2mat,
                      Named("rhomat")=rhomat,
                      Named("devmat")=devmat,
                      Named("Rcount")=Rcount,
                      Named("bcount")=bcount);
}






//=================================================================
// fix R = I
//=================================================================


//==================================================================================
// MCMC Update for two-part model with MVP, no variable selection
//===================================================================================

//[[Rcpp::export]]
List cppmcmcUpdate1(colvec& ID, colvec& y, mat& X1, mat& X2, mat& Z1, mat& Z2, 
                   colvec& A0, mat& sigA, colvec& B0, mat& sigB, 
                   double nu0, double tau0, double nub, mat& Sb, 
                   double delta, mat& propSigb, double df, 
                   double division, double niter){
  
  
  // y,X1,X2,Z1,Z2,b1,b2 need to be sorted by ID
  
  colvec subid = unique(ID);
  
  int n = y.n_elem, nid = subid.n_elem,
    p = X1.n_cols, q = X2.n_cols,
    pz = Z1.n_cols, qz = Z2.n_cols, nb = pz+qz,
    k = n/nid;// nrho = k*(k-1)/2;
  
  colvec wdraw, alphadraw, betadraw, rhodraw;
  mat Sigbdraw, b1, b2, Rdraw;
  double sig2y, deviance;
  
  //initial values 
  wdraw = randn<colvec>(n);
  alphadraw = zeros<colvec>(p);
  betadraw = zeros<colvec>(q);
  b1 = randn<mat>(nid,pz);
  b2 = randn<mat>(nid,qz);
  sig2y = 1;
  
  Sigbdraw = zeros<mat>(nb,nb);
  Sigbdraw.diag().ones();
  
  Rdraw = zeros<mat>(k,k);
  Rdraw.diag().ones();
  //rhodraw = corvec(Rdraw);
  
  //colvec Sigbvec = vectorise(Sigbdraw);
  colvec Sigbvec = varvec(Sigbdraw);
  int nsigb = Sigbvec.n_elem;
  
  //double Rcount = 0;
  //mat rhocount = zeros<mat>(k,k);
  colvec bcount = zeros<colvec>(nid);
  
  List Rout, bout;
  // results matrix
  
  mat Wmat = zeros<mat>(niter,n),
    alphamat = zeros<mat>(niter,p),
    betamat = zeros<mat>(niter,q),
    Sigbmat = zeros<mat>(niter,nsigb);
    //b1mat = zeros<mat>(niter,nid),
    //b2mat = zeros<mat>(niter,nid);
    //rhomat = zeros<mat>(niter,nrho);
  
  cube b1mat = zeros<cube>(nid,pz,niter),
       b2mat = zeros<cube>(nid,qz,niter);
  
  colvec sig2ymat = zeros<colvec>(niter),
    devmat = zeros<colvec>(niter);
  
  for(int iter=0; iter<niter; iter++){
    
    //wdraw = wUpdate(ID,y,wdraw,X1,Z1,b1,alphadraw,Rdraw);
    wdraw = wUpdate(ID,y,X1,Z1,b1,alphadraw,Rdraw);
    
    alphadraw = alphaUpdate(ID,wdraw,X1,Z1,b1,Rdraw,A0,sigA);
    
    //Rout = RUpdate(ID,wdraw,X1,Z1,b1,alphadraw,Rdraw,division,Rcount); 
    
    //Rdraw = as<mat>(Rout["Rdraw"]);
    //rhodraw = as<colvec>(Rout["rhodraw"]);
    //Rcount = as<double>(Rout["Rcount"]);
    
    betadraw = betaUpdate(ID,y,X2,Z2,b2,sig2y,B0,sigB);
    
    sig2y = sig2yUpdate(ID,y,X2,Z2,b2,betadraw,nu0,tau0);
    
    Sigbdraw = SigbUpdate(nid,b1,b2,nub,Sb);
    
    Sigbvec = varvec(Sigbdraw);
    
    bout = bUpdate(ID,y,X1,X2,Z1,Z2,b1,b2,alphadraw,betadraw,
                   sig2y,Sigbdraw,delta,propSigb, df,bcount);
    
    b1 = as<mat>(bout["b1"]);
    b2 = as<mat>(bout["b2"]);
    bcount = as<colvec>(bout["bcount"]);
    
    deviance = fullloglik(ID,y,X1,X2,Z1,Z2,b1,b2,alphadraw,
                          betadraw,sig2y,Sigbdraw,A0,sigA, 
                          B0,sigB,nu0,tau0,nub,Sb);
    
    Wmat.row(iter) = trans(wdraw);
    alphamat.row(iter) = trans(alphadraw);
    betamat.row(iter) = trans(betadraw);
    sig2ymat(iter) = sig2y;
    Sigbmat.row(iter) = trans(Sigbvec);
    b1mat.slice(iter) = b1;
    b2mat.slice(iter) = b2;
   // rhomat.row(iter) = trans(rhodraw);
    devmat(iter) = deviance;
    
  }
  
  return List::create(Named("Wmat")=Wmat,
                      Named("alphamat")=alphamat,
                      Named("betamat")=betamat,
                      Named("sig2ymat")=sig2ymat,
                      Named("Sigbmat")=Sigbmat,
                      Named("b1mat")=b1mat,
                      Named("b2mat")=b2mat,
                     // Named("rhomat")=rhomat,
                      Named("devmat")=devmat,
                     // Named("Rcount")=Rcount,
                      Named("bcount")=bcount);
}













//==============================================================
// Prediction ybin, and y|y>0
//==============================================================

//============================================
// predict y_i, i=1,...,t, t=3
//============================================

//[[Rcpp::export]]

List yipred(mat& X1new_i, mat& X2new_i, mat& Z1new_i, mat& Z2new_i, 
              colvec& b1_i, colvec& b2_i, colvec& alphadraw, 
              mat& Rdraw, colvec& betadraw, double sig2y){
  int ni = X1new_i.n_rows;
  
  colvec mu1_i, mu2_i, wnew_i, ybin_i(ni), ypos_i(ni);
  
  mu1_i = X1new_i*alphadraw+Z1new_i*b1_i;
 
  mu2_i = X2new_i*betadraw+Z2new_i*b2_i;
  
  wnew_i = vectorise(mvrnorm(1,mu1_i,Rdraw));

  for(int j=0; j<ni; j++){
    if(wnew_i(j)<0){
      ybin_i(j) = 0;
      ypos_i(j) = 0;   
    }
    if(wnew_i(j)>0){
      ybin_i(j) = 1;
      ypos_i(j) = R::rlnorm(mu2_i(j),sqrt(sig2y));
     }
  }
  
  return List::create(Named("ybin_i")=ybin_i,
                      Named("ypos_i")=ypos_i);
}


//========================================================================================
// predict y=(y1,...yi,...yn), i=1,...n, yi= y(i1,...,yij,...yit) for each iteration
//========================================================================================

//[[Rcpp::export]]
List yprediction(colvec& IDnew, mat& X1new, mat& X2new, mat& Z1new,
                 mat& Z2new, mat& b1, mat& b2, colvec& alphadraw, 
                 mat& Rdraw, colvec& betadraw, double sig2y){
  
  colvec subid = unique(IDnew);
  
  int nid = subid.n_elem, 
      n = X1new.n_rows; // X1new and X2new, Z1new, Z2new should have same number of rows
  
  colvec ybin = zeros<colvec>(n), ypos = zeros<colvec>(n);
  
  int id;
  
  mat X1new_i, X2new_i, Z1new_i, Z2new_i, ybin_i, ypos_i;
  
  colvec b1_i, b2_i;
  
  List yipredout;
  
  for(int i=0; i<nid;i++){
    
    id = subid(i);
    
    X1new_i = X1new.rows(find(IDnew==id));
    X2new_i = X2new.rows(find(IDnew==id));
    Z1new_i = Z1new.rows(find(IDnew==id));
    Z2new_i = Z2new.rows(find(IDnew==id));
    
    b1_i = trans(b1.rows(find(subid==id)));
    b2_i = trans(b2.rows(find(subid==id)));
    
    yipredout = yipred(X1new_i,X2new_i,Z1new_i,Z2new_i,b1_i,
                       b2_i,alphadraw,Rdraw,betadraw,sig2y);
    
    ybin_i = as<colvec>(yipredout["ybin_i"]);
    ypos_i = as<colvec>(yipredout["ypos_i"]);
    
    ybin.rows(find(IDnew==id)) = ybin_i;
    ypos.rows(find(IDnew==id)) = ypos_i;
  }
  
  return List::create(Named("ybin")=ybin,
                      Named("ypos")=ypos);
}
  
  
  
  
//===============================================================================
// predict y for all iterations
//===============================================================================

//=========================================
// convert rhodraw to Rdraw
//=========================================

//[[Rcpp::export]]

mat corrmat(colvec v){
  
  int p = v.n_elem;
  
  int k = (1+sqrt(1+4*2*p))/2;
 
  int ki;
  
  mat V = zeros<mat>(k,k);
  
  colvec vtemp = v, vs;
  
  for(int i=0; i<k-1; i++){
    
    ki = k-(i+1);
   
    vs = vtemp.rows(0,ki-1);
    
    V(i,span(i+1,k-1)) = trans(vs);
    V(span(i+1,k-1),i) = vs;
    
    vtemp.shed_rows(0,ki-1);

  }
  
  V.diag().ones();
  
  return V;
}





//[[Rcpp::export]]

List ypredmat(colvec& IDnew, mat& X1new, mat& X2new, mat& Z1new,
              mat& Z2new, cube& b1mat, cube& b2mat, mat& alphamat, 
              mat& rhomat, mat& betamat, colvec& sig2ymat){
  
  int niter = sig2ymat.n_elem, n = X1new.n_rows;
  
  mat b1, b2, Rdraw;
  
  colvec alphadraw, rhodraw, betadraw, ybin, ypos;
  
  double sig2y;
  
  List ypredout;
  
  mat ybinmat = zeros<mat>(niter,n),
      yposmat = zeros<mat>(niter,n);
  
  for(int iter=0; iter<niter; iter++){
    
    b1 = b1mat.slice(iter);
    b2 = b2mat.slice(iter);
    alphadraw = trans(alphamat.row(iter));
    
    rhodraw = trans(rhomat.row(iter));
    Rdraw = corrmat(rhodraw);
    
    betadraw = trans(betamat.row(iter));
    
    sig2y = sig2ymat(iter);
    
    ypredout = yprediction(IDnew,X1new,X2new,Z1new,Z2new,b1,b2,
                           alphadraw,Rdraw,betadraw,sig2y);
    
    ybin = as<colvec>(ypredout["ybin"]);
    ypos = as<colvec>(ypredout["ypos"]);
    
    ybinmat.row(iter) = trans(ybin);
    yposmat.row(iter) = trans(ypos);
    
  }
  
  return List::create(Named("ybinmat")=ybinmat,
                      Named("yposmat")=yposmat);
  
}
