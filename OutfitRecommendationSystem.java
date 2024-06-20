package org.example;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.List;

public class OutfitRecommendationSystem {

    private static final int height = 64;
    private static final int width = 64;
    private static final int channels = 3;
    private static final int numClasses = 10; // Number of outfit types
    private static final int batchSize = 32;
    private static final int numEpochs = 5;

    public static void main(String[] args) throws IOException, InterruptedException {

        // Set up UI server for monitoring training progress
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage(); // Update with your path

        uiServer.attach(statsStorage);

        // Load and preprocess the dataset
        File baseDir = new File("C:\\Users\\pc\\OneDrive\\Desktop\\minor_project-1\\Slashed dataset\\Slashed dataset");
        FileSplit fileSplit = new FileSplit(baseDir, NativeImageLoader.ALLOWED_FORMATS, new Random(42));

        // Define transformations
        ImageTransform cropTransform = new CropImageTransform(10);
        ImageTransform flipTransform = new FlipImageTransform(123);
        ImageTransform rotateTransform = new RotateImageTransform(45);
        ImageTransform scaleTransform = new ScaleImageTransform(0.8F);

        // Create a list to hold your transformations
        List<ImageTransform> imageTransformList = new ArrayList<>();

        // Add your transformations to the list
        imageTransformList.add(cropTransform);
        imageTransformList.add(flipTransform);
        imageTransformList.add(rotateTransform);
        imageTransformList.add(scaleTransform);

        // Create a MultiImageTransform object and add the transformations to it
        Random random = new Random();
        int randomIndex = random.nextInt(imageTransformList.size());
        ImageTransform transform = imageTransformList.get(randomIndex);

        ImageRecordReader recordReader = new ImageRecordReader(height, width, channels, new ParentPathLabelGenerator());
        recordReader.initialize(fileSplit, transform);
        RecordReaderDataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        dataSetIterator.setPreProcessor(new ImagePreProcessingScaler(0, 1));

        // Define the neural network architecture
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(0, new ConvolutionLayer.Builder()
                        .nIn(3)
                        .nOut(64)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new DenseLayer.Builder()
                        .nOut(256)
                        .activation(Activation.RELU)
                        .build())
                .layer(2, new OutputLayer.Builder()
                        .nOut(numClasses)
                        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutional(height, width, channels))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // Train the model
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            model.fit(dataSetIterator);
        }

        // Evaluate the model
        Evaluation evaluation = model.evaluate(dataSetIterator);
        System.out.println("***** Evaluation *****");
        System.out.println(evaluation.stats());

        // Make predictions for outfit recommendation
        Map<String, List<Double>> clothingItemScores = new HashMap<>();
        Map<String, String> outfitTypeMap = new HashMap<>();
        outfitTypeMap.put("Bags", "Type 1");
        outfitTypeMap.put("Dress", "Type 2");
        outfitTypeMap.put("Hats", "Type 3");
        outfitTypeMap.put("Jumpsuits", "Type 4");
        outfitTypeMap.put("Necklace", "Type 5");
        outfitTypeMap.put("Pants", "Type 6");
        outfitTypeMap.put("Shoes", "Type 7");
        outfitTypeMap.put("Skirt", "Type 8");
        outfitTypeMap.put("Top", "Type 9");
        outfitTypeMap.put("Watches", "Type 10");

        // Assuming dataSetIterator contains the test dataset
        while (dataSetIterator.hasNext()) {
            DataSet testData = dataSetIterator.next();
            INDArray features = testData.getFeatures();
            INDArray predicted = model.output(features, false);

            int[] predictedLabels = Nd4j.argMax(predicted, 1).toIntVector();
            for (int i = 0; i < predictedLabels.length; i++) {
                int predictedLabel = predictedLabels[i];
                String predictedOutfitType = outfitTypeMap.get(String.valueOf(predictedLabel));

                String clothingItem = "ClothingItem" + i;
                double confidenceScore = predicted.getDouble(i, predictedLabel);

                // Store the confidence score for each clothing item
                if (!clothingItemScores.containsKey(clothingItem)) {
                    clothingItemScores.put(clothingItem, new ArrayList<>());
                }
                clothingItemScores.get(clothingItem).add(confidenceScore);

                System.out.println("Predicted Outfit Type for " + clothingItem + ": " + predictedOutfitType);
            }
        }

        // Recommend an outfit based on the top-scoring item from each category
        Map<String, Double> topScoringItems = new HashMap<>();
        Map<String, String> recommendedOutfitPaths = new HashMap<>();

        for (String clothingItem : clothingItemScores.keySet()) {
            List<Double> scores = clothingItemScores.get(clothingItem);
            double maxScore = scores.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);
            topScoringItems.put(clothingItem, maxScore);

            // Assuming your images are named ClothingItem0.jpg, ClothingItem1.jpg, etc.
            recommendedOutfitPaths.put(clothingItem, "C:\\Users\\pc\\OneDrive\\Desktop\\minor_project-1\\Slashed dataset\\Slashed dataset" + outfitTypeMap.get(clothingItem) + "\\" + clothingItem + ".jpg");
        }

        // Sort items by score in descending order
        List<Map.Entry<String, Double>> sortedItems = new ArrayList<>(topScoringItems.entrySet());
        sortedItems.sort(Map.Entry.comparingByValue(Comparator.reverseOrder()));



// Display the recommended outfit using JOptionPane
        String recommendedOutfit = "***** Recommended Outfit *****\n";
        for (Map.Entry<String, Double> entry : sortedItems) {
            String clothingItem = entry.getKey();
            String imagePath = recommendedOutfitPaths.get(clothingItem);

            recommendedOutfit += clothingItem + " - Score: " + entry.getValue() + "\n";
            recommendedOutfit += "Recommended Outfit: " + outfitTypeMap.get(clothingItem) + "\n";

            // Add image to the JOptionPane
            BufferedImage img = ImageIO.read(new File(imagePath));
            ImageIcon icon = new ImageIcon(img.getScaledInstance(200, 200, Image.SCALE_SMOOTH));
            recommendedOutfit += " ";
            recommendedOutfit += "![Recommended Outfit](" + imagePath + ")\n\n";
        }

        JOptionPane.showMessageDialog(null, recommendedOutfit, "Recommended Outfit", JOptionPane.PLAIN_MESSAGE);

        // Save the model
        model.save(new File("C:\\Users\\pc\\OneDrive\\Desktop\\minor_project-1\\saved_model.zip"));

        // Close the UIServer to avoid memory leaks
        uiServer.stop();
    }
}