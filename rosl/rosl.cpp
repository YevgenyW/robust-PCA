/***************************************************************************

	C++ Robust Orthonormal Subspace Learning

	Author:	Tom Furnival	
	Email:	tjof2@cam.ac.uk

	Copyright (C) 2015 Tom Furnival

	This is a C++ implementation of Robust Orthonormal Subspace Learning (ROSL)
	and its fast variant, ROSL+, based on the MATLAB implementation by Xianbiao
	Shu et al. [1].

	References:
	[1] 	"Robust Orthonormal Subspace Learning: Efficient Recovery of 
			Corrupted Low-rank Matrices", (2014), Shu, X et al.
			http://dx.doi.org/10.1109/CVPR.2014.495  

***************************************************************************/

// ROSL header
#include "rosl.hpp"

void ROSL::runROSL(arma::mat *X) {
	int m = (*X).n_rows;
	int n = (*X).n_cols;	

	switch(method) {
		case 0:
			// For fully-sampled ROSL
			InexactALM_ROSL(X);
			break;
		case 1:		
			// For sub-sampled ROSL+
			arma::uvec rowall, colall;
			arma::arma_rng::set_seed_random();
			rowall = (Sh == m) ? arma::linspace<arma::uvec>(0, m-1, m) : arma::shuffle(arma::linspace<arma::uvec>(0, m-1, m));
			colall = (Sl == n) ? arma::linspace<arma::uvec>(0, n-1, n) : arma::shuffle(arma::linspace<arma::uvec>(0, n-1, n));			
			
			arma::uvec rowsample, colsample;
			rowsample = (Sh == m) ? rowall : arma::join_vert(rowall.subvec(0,Sh-1), arma::sort(rowall.subvec(Sh,m-1)));			
			colsample = (Sl == n) ? colall : arma::join_vert(colall.subvec(0,Sl-1), arma::sort(colall.subvec(Sl,n-1)));

			arma::mat Xperm;
			Xperm = (*X).rows(rowsample);
			Xperm = Xperm.cols(colsample);
	
			// Take the columns and solve the small ROSL problem
			arma::mat XpermTmp;
			XpermTmp = Xperm.cols(0,Sl-1);	
			InexactALM_ROSL(&XpermTmp);
	
			// Free some memory
			XpermTmp.set_size(Sh,Sl);
			
			// Now take the rows and do robust linear regression
			XpermTmp = Xperm.rows(0,Sh-1);		
			//InexactALM_RLR(&XpermTmp);
			
			// Free some memory
			Xperm.reset();
			XpermTmp.reset();
	
			// Calculate low-rank component			
			A = D * alpha;
			
			// Permute back
			A.cols(colsample) = A;
			A.rows(rowsample) = A;
	
			// Calculate error
			E = *X - A;
			break;	
	}	
	
	// Free some memory
	Z.reset();
	Etmp.reset();
	error.reset();
	A.reset();

	return;
};

void ROSL::InexactALM_ROSL(arma::mat *X) {
	using namespace std;
	int m = (*X).n_rows;
	int n = (*X).n_cols;
	int precision = (int)std::abs(std::log10(tol))+2;

	cout<<"m= "<<m<<", n = "<<n<<", precision = "<<precision<<std::endl;


	// Initialize A, Z, E, Etmp and error
	A.set_size(m, n);
	Z.set_size(m, n);
	E.set_size(m, n);
	Etmp.set_size(m, n);
	alpha.set_size(R, n);
	D.set_size(m, R);
	error.set_size(m, n);

	// Initialize alpha randomly
	arma::arma_rng::set_seed_random();
	alpha.randu();
	
	// Set all other matrices
	A = *X;		
	D.zeros();
	E.zeros();			
	Z.zeros();
	Etmp.zeros();
			
	double infnorm, fronorm;
	infnorm = arma::norm(arma::vectorise(*X), "inf");
	fronorm = arma::norm(*X, "fro");
			
	// These are tunable parameters
	double rho, mubar;
	mu = 10 * lambda / infnorm;
	rho = 1.5;
	mubar = mu * 1E7;
			
	double stopcrit;		
			
	for(int i = 0; i < maxIter; i++) {

		// Perform the shrinkage
		LowRankDictionaryShrinkage(X);

		// Error matrix and intensity thresholding
		Etmp = *X + Z - A;
		// cout<<"Etmp= "<<Etmp<<endl;
		E = arma::abs(Etmp) - lambda / mu;
		// cout<<"E before = "<<E<<endl;
		E.transform([](double val) { return (val > 0.) ? val : 0.; });
		E = E % arma::sign(Etmp);
		// cout<<"E after = "<<E<<endl;
			
		// Calculate stop criterion		
		stopcrit = arma::norm(*X - A - E, "fro") / fronorm;	
		cout << "stopcrit= " <<	stopcrit <<", fronorm = " << fronorm << endl;
		roslIters = i+1;
									
		// Exit if stop criteria is met		
		if(stopcrit < tol) {
			// Report progress
			if(verbose) {
				std::cout<<"---------------------------------------------------------"<<std::endl;
				std::cout<<"   ROSL iterations: "<<i+1<<std::endl;
				std::cout<<"    Estimated rank: "<<D.n_cols<<std::endl;
				std::cout<<"       Final error: "<<std::fixed<<std::setprecision(precision)<<stopcrit<<std::endl;
				std::cout<<"---------------------------------------------------------"<<std::endl;
			}			
			return;
		}

		// Update Z
		Z = (Z + *X - A - E) / rho;
		mu = (mu*rho < mubar) ? mu*rho : mubar;
	}
	
	// Report convergence warning
	std::cout<<"---------------------------------------------------------"<<std::endl;
	std::cout<<"   WARNING: ROSL did not converge in "<<roslIters<<" iterations"<<std::endl;
	std::cout<<"            Estimated rank:  "<<D.n_cols<<std::endl;
	std::cout<<"               Final error: "<<std::fixed<<std::setprecision(precision)<<stopcrit<<std::endl;
	std::cout<<"---------------------------------------------------------"<<std::endl;
	
	return;		
};

void ROSL::LowRankDictionaryShrinkage(arma::mat *X) {
	// Get current rank estimate
	rank = D.n_cols;
	std::cout<<"current rank = "<<rank<<std::endl;
	// Thresholding
	double alphanormthresh;
	arma::vec alphanorm(rank);
	alphanorm.zeros();	
	arma::uvec alphaindices;
	
	// Norms
	double dnorm;	
	
	// Loop over columns of D
	for(int i = 0; i < rank; i++) {			
		// Compute error and new D(:,i)
		D.col(i).zeros();
		error = ((*X + Z - E) - (D*alpha));
		D.col(i) = error * arma::trans(alpha.row(i));								
		dnorm = arma::norm(D.col(i)); 
		
		// Shrinkage
		if(dnorm > 0.) {				
			// Gram-Schmidt on D
			for(int j = 0; j < i; j++) {
				D.col(i) = D.col(i) - D.col(j) * (arma::trans(D.col(j)) * D.col(i));					
			}
		
			// Normalize
			D.col(i) /= arma::norm(D.col(i));
			
			// Compute alpha(i,:)
			alpha.row(i) = arma::trans(D.col(i)) * error;
					
			// Magnitude thresholding			
			alphanorm(i) = arma::norm(alpha.row(i));		
			alphanormthresh = (alphanorm(i) - 1/mu > 0.) ? alphanorm(i) - 1/mu : 0.;					
			alpha.row(i) *= alphanormthresh / alphanorm(i);									
			alphanorm(i) = alphanormthresh;
		}
		else {
			alpha.row(i).zeros();
			alphanorm(i) = 0.;
		}		
	}
			
	// Delete the zero bases
	alphaindices = arma::find(alphanorm != 0.);
	D = D.cols(alphaindices);
	alpha = alpha.rows(alphaindices);
	
	// Update A
	A = D * alpha;
	
	return;				
};

// This is the Python/C interface using ctypes
//		- Needs to be C-style for simplicity
int pyROSL(double *xPy, double *dPy, double *alphaPy, double *ePy, int m, int n, int R, double lambda, double tol, int iter, int method, int subsamplel, int subsampleh, bool verbose) {
	
	// Create class instance
	ROSL *pyrosl = new ROSL();
	
	// First pass the parameters (the easy bit!)
	pyrosl->Parameters(R, lambda, tol, iter, method, subsamplel, subsampleh, verbose);
	
	/////////////////////
	//                 //
	// !!! WARNING !!! //
	//                 //
	/////////////////////
		
	// This is the dangerous bit - we want to avoid copying, so set up the Armadillo
	// data matrix to DIRECTLY read from auxiliary memory, but be careful, this is
	// also writable!!!
	// Remember also that Armadillo stores in column-major order
	arma::mat X(xPy, m, n, false, false);
	
	// Time ROSL	
	auto timerS1 = std::chrono::steady_clock::now();
	
	// Run ROSL
	pyrosl->runROSL(&X);
		
	auto timerE1 = std::chrono::steady_clock::now();
	auto elapsed1 = std::chrono::duration_cast<std::chrono::microseconds>(timerE1 - timerS1);
	if(verbose) {		
		std::cout<<"Total time: "<<std::setprecision(5)<<(elapsed1.count()/1E6)<<" seconds"<<std::endl;
	}

	// Get the estimated rank
	int rankEst = pyrosl->getR();

	// Now copy the data back to return to Python	
	pyrosl->getD(dPy, m, n);
	pyrosl->getAlpha(alphaPy, m, n);
	pyrosl->getE(ePy);
	
	// Free memory
	delete pyrosl;
	
	// Return the rank
	return rankEst;
}



