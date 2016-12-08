// Derivation of European call option value using Black-Scholes and verification using Monte Carlo methods
// Heather Porter
// December 2016

// Include header files
#include "stdafx.h"
#include <iostream>			// Used for input/output
#include <vector>			// Used for providing detailed output of the Monte Carlo simulation

// Include Boost libraries (available from: http://www.boost.org/)
// N.B. Using Boost 1.62.0 for distributions and value of pi
#include <boost/math/distributions/normal.hpp>		// Use for cumulative normal distribution
#include <boost/random/normal_distribution.hpp>		// Use for normal distribution sampling
#include <boost/math/constants/constants.hpp>		// Use for pi
#include <boost/random.hpp>							// Use for random number generation


// Standard namespace
using namespace std;

// Define vector type -- used in Monte Carlo simulation
typedef vector<double> Vector;


// Seed and initialise normal random engine
boost::mt19937 seed;								// Set random seed
boost::math::normal normDistCDF(0.0, 1.0);			// Normal distribution for cumulative distribution function (Black-Scholes)
boost::normal_distribution<> normDist(0.0, 1.0);	// Normal distribution for sampling (Monte Carlo)
boost::variate_generator<boost::mt19937&,boost::normal_distribution<> > normDistSample(seed, normDist);		// Create variable generator (normal)

// Get value of pi from Boost
const double pi = boost::math::constants::pi<double>();


// Function for obtaining the cumulative probability of a N(0,1) random variable (uses the Boost libraries)
double cumulativeNormalDistribution(double alpha)
{
	return cdf(normDistCDF, alpha);
}

// Function for the randomisation of the input values (uses the Boost libraries)
// Uniform distribution between range [minValue, maxValue]
double uniformRandomBoost(double minValue, double maxValue)
{
	boost::random::uniform_real_distribution<> uni(minValue, maxValue);
	return uni(seed);
}

// Function for calculating the analytical value of the European call option by Black-Scholes
double blackScholes(double S0, double K, double r, double sigma, int T)
{
	// Original Black-Scholes formulae -- payoff at maturity is max(0,ST - K)
	/*
	double Vt, d1, d2;

	// Calculate d1 and d2
	d1 = (log(S0/K) + (r + (sigma*sigma)/2)*T) / (sigma*sqrt(T));
	d2 = d1 - sigma*sqrt(T);

	// Calculate the value of the call option
	Vt = S0*cumulativeNormalDistribution(d1) - K*cumulativeNormalDistribution(d2)*exp(-r*T);
	*/

	// Modified Black-Scholes formulae -- payoff at maturity is max(0,ln(ST) - ln(K))
	double Vt, d;

	// Calculate d -- see BlackScholes Exercise Heather Porter.pdf, Equation (7)
	d = (log(S0/K) + (r - (sigma*sigma)/2)*T) / (sigma*sqrt(T)); 

	// Calculate the value of the call option -- see BlackScholes Exercise Heather Porter.pdf, Equation (10)	
	Vt = exp(-r*T) * ((cumulativeNormalDistribution(d) * (log(S0/K) + (r - (sigma*sigma)/2)*T)) + sigma*sqrt(T)*exp(-(d*d)/2) / sqrt(2*pi));
	
	return Vt;
}

// Function for calculating the value of the European call option using Monte Carlo simulation
double monteCarlo(double S0, double K, double r, double sigma, int T, int numIter, bool detailedOutput)
{
	// Monte Carlo simulation
	// 1) Simulate final value of stock (ST)
	// 2) Given final value of stock, find VT = max(0, ln(ST) - ln(K))
	// 3) Loop (1) and (2) over number of iterations
	// 4) Find average payoff and discount with discount factor exp(-rT)

	// Final value of stock -- see BlackScholes Exercise Heather Porter.pdf, Equation (4)
	// ST = St * exp((r - (sigma*sigma)/2)*(T-t) - sigma*(sqrt(T-t))*N(0,1))

	// Loop over number of iterations -- find final value of stock (ST) given by above equation, then final value of option given by max(0, ln(ST) - ln(K))
	// In order to improve computation speed we can pre-calculate some sections of the formula for ST that are not dependent on sampling a N(0,1) random variable
	double a, b;
	a = S0 * exp((r - sigma*sigma / 2)*T);
	b = -sigma*sqrt(T);

	// Define variables for Monte Carlo simulation
	double tmp;							// ST in iteration
	double tmpTotal = 0;				// Total sum of VT over all iterations
	Vector monteCarloValues(numIter);	// Vector containing VT for each iteration

	// Perform the Monte Carlo simulation
	for (int i = 0; i < numIter; i++)
	{
		// Value of stock at time T in iteration
		tmp = a * exp(b*normDistSample());

		// Total payoff at maturity (VT) given ST (tmp)
		// Simplifiation of max(0, ln(ST) - ln(K)) using the property that ln(x) - ln(y) = ln(x/y)
		monteCarloValues[i] = max(0.0, log(tmp / K));
		tmpTotal += monteCarloValues[i];
	}

	// Find the mean payoff and discount to obtain the mean value of the call option
	double meanPrice;
	meanPrice = (tmpTotal / (double)numIter) * exp(-r*T);
	
	// For user defined inputs mode calculate and output additional payoff information 
	if (detailedOutput == 1)
	{
		int numZeroPayoff = 0;
		int numLessThanPrice = 0;
		int numGreaterThanPrice = 0;

		for (int i = 0; i < numIter; i++)
		{
			// Number of payoffs equal to 0
			if (monteCarloValues[i] == 0.0)
			{
				++numZeroPayoff;
			}
			// Number of payoffs greater than 0, but less than the mean call price
			else if (monteCarloValues[i] > 0.0 & monteCarloValues[i] < meanPrice)
			{
				++numLessThanPrice;
			}
			// Number of payoffs greater than or equal to the mean call price
			else if (monteCarloValues[i] >= meanPrice)
			{
				++numGreaterThanPrice;
			}
		}

		// Output the value of the call option
		cout << "Price of call option: " << meanPrice << "\n";

		cout << "Detailed output: \n";
		cout << "  " << ((double)numZeroPayoff / (double)numIter) * 100.0 << "% have zero payoff \n";
		cout << "  " << ((double)numLessThanPrice / (double)numIter) * 100.0 << "% have payoff less than the call price \n";
		cout << "  " << ((double)numGreaterThanPrice / (double)numIter) * 100.0 << "% have payoff greater than or equal to the call price \n";
	}
	
	return meanPrice;
}

// We implement two modes for the Monte Carlo verification:
// (1) User specified inputs
// (2) Randomly generated inputs

// (1) Function to define the user specified inputs
void userDefinedVerification()
{
	// Ask user for input values
	cout << "To start please enter the input values for the European option. \n \n";

	double S0, K, r, sigma;
	unsigned int T, numIter;

	// Starting value of the stock (S0)
	cout << "Enter starting value of the stock (S0): \n";
	cin >> S0;
	while (cin.fail() || S0 <= 0) 
	{ 
		cout << "ERROR -- please enter a value greater than 0: \n";
		cin.clear();
		cin.ignore();
		cin >> S0;
	}
	cout << "\n";

	// Strike price (K)
	cout << "Enter strike price (K): \n";
	cin >> K;
	while (cin.fail() || K <= 0) 
	{
		cout << "ERROR -- please enter a value greater than 0: \n";
		cin.clear();
		cin.ignore();
		cin >> K;
	}
	cout << "\n";

	// Volatility (sigma)
	cout << "Enter volatility (sigma) [e.g. for 20% please enter 0.2]: \n";
	cin >> sigma;
	while (cin.fail() || sigma <= 0) 
	{
		cout << "ERROR -- please enter a value greater than 0: \n";
		cin.clear();
		cin.ignore();
		cin >> sigma;
	}
	cout << "\n";

	// Risk-free rate of return (r)
	cout << "Enter risk-free rate of return (r) [e.g. for 1% please enter 0.01]: \n";
	cin >> r;
	while (cin.fail()) 
	{
		cout << "ERROR -- please enter a value between -infinity and infinity: \n";
		cin.clear();
		cin.ignore();
		cin >> r;
	}
	cout << "\n";

	// Number of years (T)
	cout << "Enter number of years (T) [integer] -- non-integers will be truncated: \n";
	cin >> T;
	while (cin.fail() || T < 1) 
	{
		cout << "ERROR -- please enter an integer greater than or equal to 1: \n";
		cin.clear();
		cin.ignore(256, '\n');
		cin >> T;
	}
	cout << "\n";

	// Number of Monte Carlo iterations (numIter) 
	cout << "Enter the number of Monte Carlo iterations [integer] -- non-integers will be truncated: \n";
	cin >> numIter;
	while (cin.fail() || numIter < 1)
	{
		cout << "ERROR -- please enter an integer greater than or equal to 1: \n";
		cin.clear();
		cin.ignore(256, '\n');
		cin >> numIter;
	}
	cout << "\n";

	cout << "Thanks! \n" << "Calculating... \n \n";

	// Output the Black-Scholes analytical solution and the verification by Monte Carlo simulation
	double analyticalResult, monteCarloResult;

	// Black-Scholes (analytical)
	cout << "Black-Scholes (analytical solution) \n";
	analyticalResult = blackScholes(S0, K, r, sigma, T);
	cout << "Price of call option: " << analyticalResult << "\n \n";

	// Monte Carlo verification
	cout << "Monte Carlo verification of Black-Scholes \n";
	// Detailed output option enabled -- function will print results to console
	monteCarloResult = monteCarlo(S0, K, r, sigma, T, numIter, 1);
	cout << "\n";

	// Absolute difference between analytical solution and Monte Carlo solution
	cout << "Absolute difference between analytical solution and Monte Carlo solution: " << abs(analyticalResult - monteCarloResult) << "\n";
	cout << "\n \n";
}

// (2) Randomly generated inputs
void randomInputVerification()
{
	// Ask user for number of random input simulations
	unsigned int numSims = 1000;

	cout << "Enter the number of random input simulations -- the larger the number the more thorough the verification (default 1,000): \n";
	cin >> numSims;
	while (cin.fail() || numSims < 1)
	{
		cout << "ERROR -- please enter an integer greater than or equal to 1: \n";
		cin.clear();
		cin.ignore(256, '\n');
		cin >> numSims;
	}
	cout << "\n";

	// Ask user for number of Monte Carlo iterations
	unsigned int numIter = 100000; 
	
	cout << "Enter the number of Monte Carlo iterations -- the larger the number the more accurate the simulation (default 100,000): \n";
	cin >> numIter;
	while (cin.fail() || numIter < 1)
	{
		cout << "ERROR -- please enter an integer greater than or equal to 1: \n";
		cin.clear();
		cin.ignore(256, '\n');
		cin >> numIter;
	}
	cout << "\n";

	cout << "Thanks! \n" << "Calculating... \n \n";

	// Compare the Black-Scholes analytical solutions to the Monte Carlo simulations across the random input simulations
	double totDiff = 0;

	// Loop over random generation of inputs (all sampled from uniform distributions with arbitrary bounds) and sum the absolute difference
	cout << "Progress: \n";
	for (int i = 0; i < numSims; i++)
	{
		// Draw random input values from uniform distributions
		double S0 = uniformRandomBoost(0.0, 10000.0);
		double K = uniformRandomBoost(0.0, 10000.0);
		double r = uniformRandomBoost(0.0, 0.1);
		double sigma = uniformRandomBoost(0.00001, 1.0);
		int T = uniformRandomBoost(1, 50);

		// Calculate the total absolute difference across all random input simulations between the analytical solutions and Monte Carlo simulations
		totDiff += abs(blackScholes(S0, K, r, sigma, T) - monteCarlo(S0, K, r, sigma, T, numIter, 0));

		// Output progress
		if (i % (numSims / 10) == 0) 
		{
			if (i > 0)
			{
				cout << "|---";
			}
		}
	}
	cout << "|---| 100% complete \n \n";

	// Return the mean absolute difference between the analytical solution and the Monte Carlo simulation
	// We expect this difference to tend to zero as the number of Monte Carlo iterations tends to infinity 
	cout << "Mean absolute difference following random input generation with " << numSims << " simulations of " << numIter << " Monte Carlo iterations: " << totDiff / (double)numSims << "\n \n \n";
}


// Program begins here
int main()
{
	// Welcome user
	cout << "Welcome! \n \n";
	cout << "This is a Black-Scholes European call option pricing calculator developed by Heather Porter (December 2016) \n \n";

	cout << "This program will calculate the price of European call options by two methods: \n";
	cout << "  (1) Black-Scholes \n";
	cout << "  (2) Monte Carlo simulation";

	cout << "\n \n";

	// Prompt user to select mode
	cout << "Please select user defined input values or random input verification mode: \n";
	cout << "  0: User defined inputs \n";
	cout << "  1: Random input verification \n";

	bool mode;

	// Read in user input and verify that it is in the correct format
	cin >> mode;
	while (cin.fail())
	{
		cout << "ERROR -- please select 0 or 1: \n";
		cin.clear();
		cin.ignore(256, '\n');
		cin >> mode;
	}
	cout << "\n";

	// Run functions dependent on user mode choice
	if (mode == 0)
	{
		// User defined input values
		cout << "User defined input mode selected \n \n";
		// Call the user defined input procedure
		// This function prompts the user for inputs then calculates the price of the call option by Black-Scholes and verifies using Monte Carlo simulation
		userDefinedVerification();
	}
	else 
	{
		// Random input verification
		cout << "Random input value verification selected \n \n";
		// Call the random input procedure and output the mean absolute difference between the analytical solutions and the Monte Carlo simulations
		// We expect this difference to tend to zero as the number of Monte Carlo iterations tends to infinity 
		randomInputVerification();
	}

	// Ask user if they want to restart
	cout << "Start over? \n";
	cout << "  0: Exit \n";
	cout << "  1: Return to start menu \n";

	bool restart;

	cin >> restart;
	while (cin.fail())
	{
		cout << "ERROR -- please select 0 or 1: \n";
		cin.clear();
		cin.ignore(256, '\n');
		cin >> restart;
	}
	cout << "\n";

	if (restart == 0)
	{
		exit(0);
	}
	else 
	{
		cout << "\n \n \n";
		main();
	}

	return 0;
}

// End of program