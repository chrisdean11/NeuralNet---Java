/*
	Class implementation of a neural network consisting of an input layer, hidden layer, and
	output layer of arbitrary sizes. Arrays holding the inputs and outputs of each layer "live" in
	this class, and are passed down to layers and neurons only to be read and filled, ie they do
	not keep their own copy of this information.

	A user may compute the output of the NN given a set of outputs, or give it a pair of input/
	output arrays to train with.

	Chris Dean
	November 15, 2016
*/

public class NeuralNetwork{
	private int     numI;
	private int     numH;
	private int     numO;
	private Layer   hLayer;
	private Layer   oLayer;

	private double [] 	outH; 	// Output from the hidden layer
	private double []   outO;	// Output from the output layer
	private double []   deltaO; // Deltas calculated during backprop
	private double []   deltaH; // Deltas calculated during backprop

	/**
	 * Creates instance of a Neural Network with one hidden layer
	 * @param numIn         Number of inputs
	 * @param numHidden     Number of neurons in the hidden layer
	 * @param numOut        Number of neurons in the output layer
	 * @param rate          Learning rate
	 */
	NeuralNetwork(int numIn, int numHidden, int numOut, double rate){
		numI = numIn;
		numH = numHidden;
		numO = numOut;

		//in = new double[numI];
		outH = new double[numH];
		for(int i = 0; i <numH; i++) outH[i] = 42;
		outO = new double[numO];

		hLayer = new Layer(rate, numI, numH);
		oLayer = new Layer(rate, numH, numO);

	}

	/**
	 * Performs forward propagation and returns the network's output
	 * @param input 	Input vector
	 * @param output 	Output vector
	 */
	public void test(double[] input, double[] output){

		// First calculate output from hidden layer
		hLayer.calculate(input, outH);
		// Then use the hidden layer's output to calculate the next layer
		oLayer.calculate(outH, output);
	}

	/**
	 * Performs supervised learning with an input expected result
	 * @param input 	Input vector
	 * @param output 	Expected output vector
	 */
	public void train(double [] input, double [] expected){

        /*System.out.println("INPUT: ");
        for(int i=0; i<785; i++) System.out.print(input[i] + " ");
            System.out.println();
        System.out.println("EXPECTED: ");
        for(int i=0; i<10; i++) System.out.print(expected[i] + " ");
            System.out.println();*/

		// First do a forward propagation to get outputs
		test(input, outO);

		// Then compute deltas for hidden and output layer
		double[] deltaO = new double[numO];
		for(int i = 0; i < numO; i++){
			double dSigmoid = outO[i] * (1 - outO[i]);
			double error 	=  expected[i] - outO[i];
			deltaO[i] = dSigmoid * error;
		}

		double[] deltaH = new double[numH];
		for(int i = 0; i < numH; i++){
			for(int j = 0; j < numO; j++){
				double dSigmoid = outH[i] * (1 - outH[i]);
				deltaH[i] += dSigmoid * deltaO[j] * oLayer.getWeight(j, i);
			}
		}
		// Then compute backpropagation results on each layer using these deltas
		hLayer.backProp(input, deltaH);
		oLayer.backProp(outH, deltaO);
	}
}