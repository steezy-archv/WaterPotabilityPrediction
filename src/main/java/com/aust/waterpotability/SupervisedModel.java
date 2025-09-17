package com.aust.waterpotability;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import java.io.File;
import java.util.Random;

public class SupervisedModel {

    public static void main(String[] args) throws Exception {

        Instances data = DataSource.read("data/processed_water_potability.arff");
        data.setClassIndex(data.numAttributes() - 1);

        Vote voter = buildVotingClassifier();

        evaluateModel(voter, data);

        saveModel(voter, data, "models/water_quality_voting.model");

        testLoadModel("models/water_quality_voting.model", data);
    }

    /** Build Voting classifier */
    private static Vote buildVotingClassifier() {
        NaiveBayes nb = new NaiveBayes();
        J48 j48 = new J48();
        RandomForest rf = new RandomForest();
        rf.setNumIterations(100);

        Vote voter = new Vote();
        Classifier[] classifiers = { nb, j48, rf };
        voter.setClassifiers(classifiers);

        System.out.println("Voting classifier built with NaiveBayes + J48 + RandomForest");
        return voter;
    }

    /** Evaluate with 10-fold stratified CV */
    private static void evaluateModel(Classifier model, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));

        System.out.println("\n=== Voting Classifier (NaiveBayes + J48 + RandomForest) ===");
        System.out.printf("Accuracy: %.2f%%\n", eval.pctCorrect());
        System.out.printf("Precision: %.3f\n", eval.weightedPrecision());
        System.out.printf("Recall: %.3f\n", eval.weightedRecall());
        System.out.printf("F1-Score: %.3f\n", eval.weightedFMeasure());
        System.out.printf("ROC AUC: %.3f\n", eval.weightedAreaUnderROC());

        // Confusion matrix
        double[][] cm = eval.confusionMatrix();
        System.out.println("\nConfusion Matrix:");
        for (double[] row : cm) {
            for (double v : row) System.out.printf("%6.0f ", v);
            System.out.println();
        }
    }

    /** Train final model and save */
    private static void saveModel(Classifier model, Instances data, String path) throws Exception {
        model.buildClassifier(data);
        File dir = new File("models");
        if (!dir.exists()) dir.mkdirs();

        SerializationHelper.write(path, model);
        System.out.println("Saved ensemble model: " + path);
    }

    /** Test loading and predicting */
    private static void testLoadModel(String path, Instances data) throws Exception {
        Classifier loaded = (Classifier) SerializationHelper.read(path);
        if (data.numInstances() > 0) {
            Instance test = data.instance(0);
            double pred = loaded.classifyInstance(test);
            System.out.printf("Example prediction on first instance: %.0f (%s)\n",
                    pred, data.classAttribute().value((int) pred));
        }
    }
}
