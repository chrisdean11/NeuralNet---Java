/*
	Chris Dean
	November 15, 2016
*/

import java.util.Arrays;

public class Neuron{
	private int 		inputSize;
	private double [] 	weight;
	private double 		learningRate;

	/**
	 * Creates a neuron with a specific number of inputs and weights
	 * @param size 		Number of inputs
	 * @param rate 		Learning rate
	 */
	Neuron(int size, double rate){
		inputSize 	= size;
		learningRate = rate;
		weight 	= new double[inputSize];

		// Initialize the weights to a random value between -1 and 1
		for(int i = 0; i < inputSize; i++){
			weight[i] = (Math.random() * 2) - 1;
		}
	}

	/**
	 * Computes the output of this neuron based on new inputs
	 * @param input 	Inputs from previous layer of neurons
	 * @return 			Output of this neuron
	 */
	public double calculate(double [] input){
		double output = 0;
		for(int i = 0; i < inputSize; i++){
			output += input[i] * weight[i];
		}

		output = 1./(1. + Math.exp(-output));

		return output;
	}

	/**
	 * Perform back propagation to compute the new weights
	 * @param input 	Inputs from previous layer
	 * @param delta 	Computed delta value
	 */
	public void backProp(double [] input, double delta){
		// Compute the delta using the derivative of the sigmoid
		// double delta = (output*error)*(1 - output*error);

		// Compute each new weight
		for(int i = 1; i < inputSize; i++){
			weight[i] += learningRate * delta * input[i];
		}
	}

	/**
	 * Returns the weight of the specified input connection
	 * @param i 	Index of the desired weight 	
	 */
	public double getWeight(int i){
		return weight[i];
	}
}