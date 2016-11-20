/*
	Class implementation of a single layer of Neurons.
	A user can instantiate a Layer with any number of neurons, calculate the output given an input,
	and perform backpropagation given an expected output.

	Chris Dean
	November 15, 2016
*/

public class Layer{
	private int         numInputs;
	private int         size;
	private Neuron []   neuron;

	/**
	 * Creates instance of a Layer of neurons
	 * @param rate          Learning rate
	 * @param inputs 		Number of inputs to this layer
	 * @param num 			Number of neurons in this layer
	 */
	Layer(double rate, int inputs, int num){
		numInputs = inputs;
		size 	= num;
		neuron = new Neuron[size];
		for(int i = 0; i < size; i++){
			neuron[i] = new Neuron(numInputs, rate);
		}
	}

	/**
	 * Computes output of this layer given a set of inputs
	 * @param input         Array of inputs from previous layer
	 * @param output 		Output of each neuron in this layer
	 */
	public void calculate(double [] input, double [] output){
		// For each neuron, compute its output given the set of inputs
		for(int i = 0; i < size; i++){
			output[i] = neuron[i].calculate(input);
		}
	}

	/**
	 * Performs backpropagation through this layer
	 * @param input 		Input to this layer from previous layer
	 * @param delta 		Computed delta from following layer's result
	 */
	public void backProp(double [] input, double [] delta){
		for(int i = 0; i < size; i++){
			neuron[i].backProp(input, delta[i]);
		}
	}

	/**
	 * Returns the weight value from the desired neuron
	 * @param i 			The neuron number
	 * @param index 		Index of the desired input's weight for this neuron
	 */
	public double getWeight(int i, int index){
		return neuron[i].getWeight(index);
	}
}