/*
	Main class that performs optical character recognition using the MNIST training set and a 
	my simple neural net with one hidden layer. Input vectors are 784 elements in size, each 
	representing one grayscale pixel. 

	Chris Dean
	November 15, 2016
*/

import java.io.File;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;

public class OCR{
	// Training/Testing Parameters
	private static final int NUM_TRAIN 	= 60000;
	private static final int NUM_TEST 	= 10000;
	private static final int NUM_EPOCHS = 100;

	// Neural Network Parameters
	private static final int NUM_IN 	= 785;
	private static final int NUM_HIDDEN = 40;
	private static final int NUM_OUT 	= 10;
	private static final double LEARNING_RATE = .01;

	// Local Fields
	private static NeuralNetwork net;
	private static double[] input = new double[NUM_IN];
	private static double[] output = new double[NUM_OUT];
	private static String trainFile = "mnist_train.csv";
	private static String testFile = "mnist_test.csv";


    /**
     * Entry point for this application
     * @param args - Command line arguments
     */
	public static void main(String[] args) throws FileNotFoundException {

		// Create neural net
		net = new NeuralNetwork(NUM_IN, NUM_HIDDEN, NUM_OUT, LEARNING_RATE);


		// TRAIN
		// Open training file and read one line at a time into input array
		int count;
		for(int epoch = 0; epoch < NUM_EPOCHS; epoch++){
			count = 0;
			try{
				Scanner file = new Scanner(new File(trainFile));
				file.useDelimiter(",|\\n");
				while (file.hasNext()){
					if(count > NUM_TRAIN) break;

					// First value in line is the actual digit in the image
					int answer = Integer.parseInt(file.next());
					makeOutput(answer);

					// Then go through this line and build the array
					for(int i = 0; i < NUM_IN - 1; i++) input[i] = Double.parseDouble(file.next())/255.0;
					input[NUM_IN - 1] = 1; // Bias

					// Train with this array 
					net.train(input, output);

					count++;
				}
				file.close();
			}
			catch(FileNotFoundException ex){
				System.out.println("Error: Training File Not Found");
			}
		}
			

		// TEST
		// Open test file and read one line at a time into input array
		count = 0;
		BufferedWriter writer = null;
		try{
			Scanner file = new Scanner(new File(testFile));
			file.useDelimiter(",|\\n");

			// File to store neural net's answer
			File log = new File("Guesses.csv");
			writer = new BufferedWriter(new FileWriter(log));

			while (file.hasNext()){
				if(count > NUM_TEST) break;

				// First value in line is the actual digit in the image
				int answer = Integer.parseInt(file.next());
				
				// Then go through this line and build the array
				for(int i = 0; i < NUM_IN - 1; i++) input[i] = Double.parseDouble(file.next())/255.0;
				input[NUM_IN - 1] = 1.0;  // Bias

				// Test this array
				net.test(input, output);
				int response = translateResponse();
				//writer.write("Answer: " + answer + " Guess: " + response + "\n");
				writer.write(answer + "," + response + "\n");

				count++;
			}
			file.close();
			writer.close();
		}
		catch(FileNotFoundException ex){
			System.out.println("Error: Test File Not Found");
		}
		catch (Exception e) {
			e.printStackTrace();
		}

	}

	/**
	 * Turns a digit 0-9 into a one-hot encoded array to train the NN with desired output for any digit
     * @param answer - The digit to translate
	 */
	private static void makeOutput(int answer){
		for(int i = 0; i < 10; i++){
			if(answer == i) output[i] = 1.;
			else output[i] = 0.;
		}
	}

	/**
	 * Interprets the neural net output into a guess of digit 0-9.
	 */
	private static int translateResponse(){
		double max = -1;
		int maxIndex = -1;
		for(int i = 0; i < 10; i++){
			if(max < output[i]) {
				max = output[i];
				maxIndex = i;
			}
		}
		return maxIndex;
	}
}